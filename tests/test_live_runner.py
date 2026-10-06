"""Live loop: demo-only guard, closed-bar streaming, and loop state management."""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import app.main as cli
from agent.coordinator import TradingCoordinator
from core.audit import AuditTrail
from core.compliance import ComplianceGate, WeekendGuard
from core.market_data import Bar, resample
from core.risk_engine import RiskConfigError
from execution.kill_switch import KillSwitch
from execution.live_runner import (
    DemoAccountGuardError,
    LiveConfig,
    LiveRunner,
    Mt5BarFeed,
    RunnerState,
    asset_from_symbol_info,
    enforce_demo_account,
    parse_timeframe,
)
from execution.mt5_adapter import Mt5Adapter
from execution.mt5_bridge import HeartbeatConfig, HeartbeatMonitor, Mt5Error, PayloadError, ZmqMt5Transport
from tests.conftest import AlwaysLongStrategy, FakeMt5Transport, bars_from_closes, open_regime_filter

EA_SOURCE = (Path(__file__).resolve().parents[1] / "mql5" / "ZeroMQ_Listener.mq5").read_text(encoding="utf-8")
WED_NOON = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def hourly(n: int, start: datetime = WED_NOON - timedelta(hours=300), base: float = 2000.0) -> list[Bar]:
    return bars_from_closes([base + i * 0.5 for i in range(n)], wick=0.2, start=start, step=timedelta(hours=1))


class Clock:
    def __init__(self, t: datetime = WED_NOON) -> None:
        self.t = t
        self.mono = 1_000.0

    def __call__(self) -> datetime:
        return self.t

    def monotonic(self) -> float:
        return self.mono


def build(tmp_path, fake: FakeMt5Transport | None = None, compliance: ComplianceGate | None = None,
          with_monitor: bool = True) -> dict[str, Any]:
    fake = fake or FakeMt5Transport()
    history = hourly(300)
    fake.add_bars(history)
    clock = Clock()
    monitor = HeartbeatMonitor(fake, HeartbeatConfig(interval_s=1.0, max_missed=2), clock=clock.monotonic) \
        if with_monitor else None
    kill = KillSwitch(sentinel_path=tmp_path / "KILL", retry_delay_s=0)
    adapter = Mt5Adapter(fake, monitor=monitor, kill_guard=kill)
    adapter._connected = True  # connect() would start the heartbeat thread; tests drive beat() by hand
    if monitor is not None:
        monitor.beat()
    audit = AuditTrail()
    asset = asset_from_symbol_info("XAUUSD", fake.symbol_info("XAUUSD"), fake.account_info())
    coord = TradingCoordinator("XAUUSD", asset, adapter, kill, [AlwaysLongStrategy()],
                               regime_filter=open_regime_filter(), audit=audit, compliance=compliance)
    feed = Mt5BarFeed(fake, "XAUUSD", "1h", history=250)
    runner = LiveRunner(coord, feed, fake, kill, monitor, LiveConfig(poll_seconds=0), audit, clock=clock,
                        sleep=lambda s: None)
    return {"fake": fake, "clock": clock, "monitor": monitor, "kill": kill, "adapter": adapter, "audit": audit,
            "coord": coord, "feed": feed, "runner": runner, "history": history}


def next_bar(env: dict[str, Any], k: int = 1) -> list[Bar]:
    last = env["history"][-1] if not env.get("added") else env["added"][-1]
    new = bars_from_closes([last.close + 0.5 * (i + 1) for i in range(k)], wick=0.2,
                           start=last.ts + timedelta(hours=1), step=timedelta(hours=1))
    env.setdefault("added", []).extend(new)
    env["fake"].add_bars(new)
    env["clock"].t = new[-1].ts + timedelta(hours=1, seconds=5)
    return new


def events(env: dict[str, Any], name: str) -> list[Any]:
    return [r for r in env["audit"].tail(500) if r.event == name]


# ------------------------------------------------------------------ demo-only guard
class TestDemoGuard:
    @pytest.mark.parametrize("mode", [0, 1, "0"])
    def test_demo_and_contest_allowed(self, mode: Any) -> None:
        assert enforce_demo_account({"trade_mode": mode}) == int(mode)

    @pytest.mark.parametrize("info", [{"trade_mode": 2}, {"trade_mode": 3}, {"trade_mode": None}, {},
                                      {"trade_mode": "REAL"}, {"trade_mode": -1}])
    def test_real_unknown_or_missing_rejected(self, info: dict) -> None:
        with pytest.raises(DemoAccountGuardError, match="CRITICAL"):
            enforce_demo_account(info)

    def test_preflight_on_real_account_aborts_before_any_data_or_order(self, tmp_path) -> None:
        fake = FakeMt5Transport()
        fake.trade_mode = 2
        env = build(tmp_path, fake)
        fake.calls.clear()
        with pytest.raises(DemoAccountGuardError, match="REAL"):
            env["runner"].preflight()
        assert fake.calls == []  # no RATES, no SEND_BRACKET, no FLATTEN_ALL
        assert env["runner"].state is RunnerState.STOPPED
        assert events(env, "demo_guard_violation")

    def test_switch_to_real_mid_run_stops_without_touching_positions(self, tmp_path) -> None:
        env = build(tmp_path)
        runner, fake = env["runner"], env["fake"]
        runner.preflight()
        next_bar(env)
        assert runner.step().cycle.order.accepted  # one demo position open
        fake.trade_mode = 2  # operator logs the terminal into a real account
        env["monitor"].beat()
        next_bar(env)
        sends = fake.calls.count("SEND_BRACKET")
        with pytest.raises(DemoAccountGuardError):
            runner.step()
        assert fake.calls.count("SEND_BRACKET") == sends
        assert "FLATTEN_ALL" not in fake.calls  # never acts on a real account
        assert runner.state is RunnerState.STOPPED

    def test_guard_without_monitor_pings_each_step(self, tmp_path) -> None:
        env = build(tmp_path, with_monitor=False)
        env["runner"].preflight()
        env["fake"].trade_mode = 2
        with pytest.raises(DemoAccountGuardError):
            env["runner"].step()

    def test_cli_live_refuses_real_account(self, monkeypatch, capsys) -> None:
        fake = FakeMt5Transport()
        fake.trade_mode = 2
        fake.close = lambda: None  # type: ignore[attr-defined]
        monkeypatch.setattr(cli, "ZmqMt5Transport", lambda *a, **kw: fake)
        code = cli.main(["live", "--token", "t", "--news", "off"])
        out = capsys.readouterr().out
        assert code == 3 and "CRITICAL" in out and "REAL" in out
        assert fake.calls == []  # stopped after ACCOUNT_INFO: no symbol info, no bars, no orders

    def test_ea_has_independent_demo_guard(self) -> None:
        assert re.search(r"input bool\s+InpDemoOnly\s*=\s*true;", EA_SOURCE)
        assert "demo guard: REAL account - orders refused" in EA_SOURCE
        assert "refusing to start" in EA_SOURCE and "IsRealAccount()" in EA_SOURCE
        assert 'JInt("trade_mode", AccountInfoInteger(ACCOUNT_TRADE_MODE))' in EA_SOURCE

    def test_ping_permission_is_not_conflated_with_weekend_window(self) -> None:
        """A weekend must not look like a broken link: PING's trade_allowed is permissions only."""
        ping = EA_SOURCE[EA_SOURCE.index('if(action == "PING")'):EA_SOURCE.index('if(action == "ACCOUNT_INFO")')]
        assert "bool permitted = TradingPermitted(why);" in ping
        assert 'JBool("weekend_blackout", InWeekendBlackout())' in ping
        send = EA_SOURCE[EA_SOURCE.index("string SendBracket("):EA_SOURCE.index("string HandleRequest(")]
        assert "if(InWeekendBlackout())" in send  # orders are still refused over the weekend


# ------------------------------------------------------------------ bar streaming
class TestBarStreaming:
    @pytest.mark.parametrize(("tf", "code", "hours"), [("1h", "H1", 1), ("H4", "H4", 4), ("4h", "H4", 4),
                                                        ("1d", "D1", 24)])
    def test_parse_timeframe(self, tf: str, code: str, hours: int) -> None:
        assert parse_timeframe(tf) == (code, timedelta(hours=hours))

    def test_bad_timeframe(self) -> None:
        with pytest.raises(ValueError):
            parse_timeframe("7h")
        sent: list[str] = []
        t = ZmqMt5Transport(socket_factory=lambda: sent.append("socket"))  # type: ignore[arg-type,return-value]
        with pytest.raises(PayloadError):
            t.rates("XAUUSD", "H7", 10)
        assert sent == []

    def test_server_time_converted_to_utc(self) -> None:
        fake = FakeMt5Transport()
        bars = hourly(5)
        fake.add_bars(bars)
        got = Mt5BarFeed(fake, "XAUUSD", "1h").fetch(5)
        assert [b.ts for b in got] == [b.ts for b in bars]
        assert got[0].ts.tzinfo is UTC

    def test_poll_returns_only_new_closed_bars(self) -> None:
        fake = FakeMt5Transport()
        fake.add_bars(hourly(50))
        feed = Mt5BarFeed(fake, "XAUUSD", "1h", history=40)
        assert len(feed.prime()) == 40
        assert feed.poll() == []
        more = bars_from_closes([3000.0, 3001.0], start=feed.last_ts + timedelta(hours=1), step=timedelta(hours=1))
        fake.add_bars(more)
        assert [b.ts for b in feed.poll()] == [b.ts for b in more]
        assert feed.poll() == []

    def test_empty_history_is_an_error(self) -> None:
        with pytest.raises(Mt5Error, match="no H1 history"):
            Mt5BarFeed(FakeMt5Transport(), "XAUUSD", "1h").prime()

    def test_resample_to_4h(self) -> None:
        start = datetime(2026, 9, 21, 0, tzinfo=UTC)
        bars = bars_from_closes([10, 11, 9, 12, 13, 14, 15, 16, 17], wick=0.5, start=start,
                                step=timedelta(hours=1))
        out = resample(bars, timedelta(hours=4))
        assert len(out) == 2  # the 9th bar starts a bucket that is not complete yet
        first = out[0]
        assert (first.ts, first.open, first.close) == (start, 10, 12)
        assert first.high == max(b.high for b in bars[:4]) and first.low == min(b.low for b in bars[:4])
        mid = resample(bars[1:], timedelta(hours=4))  # starts mid-bucket
        assert mid[0].ts == start + timedelta(hours=4)

    def test_asset_from_broker_specs(self) -> None:
        fake = FakeMt5Transport()
        asset = asset_from_symbol_info("XAUUSD", fake.symbol_info("XAUUSD"), fake.account_info())
        assert (asset.contract_size, asset.lot_step, asset.min_lot, asset.price_tick) == (100.0, 0.01, 0.01, 0.01)
        assert asset.min_stop_distance == pytest.approx(0.1) and asset.leverage == 30
        fake.currency = "EUR"
        with pytest.raises(RiskConfigError, match="EUR"):
            asset_from_symbol_info("XAUUSD", fake.symbol_info("XAUUSD"), fake.account_info())


# ------------------------------------------------------------------ loop state management
class TestLiveLoop:
    def test_step_before_preflight_is_refused(self, tmp_path) -> None:
        with pytest.raises(RuntimeError):
            build(tmp_path)["runner"].step()

    def test_preflight_loads_history_and_audits(self, tmp_path) -> None:
        env = build(tmp_path)
        env["runner"].preflight()
        assert env["runner"].state is RunnerState.RUNNING and len(env["runner"].bars) == 250
        start = events(env, "live_start")[-1].payload
        assert start["trade_mode"] == "DEMO" and start["timeframe"] == "H1"

    def test_no_new_bar_means_no_decision(self, tmp_path) -> None:
        env = build(tmp_path)
        env["runner"].preflight()
        rep = env["runner"].step()
        assert rep.new_bars == 0 and rep.cycle is None and "SEND_BRACKET" not in env["fake"].calls

    def test_new_bar_trades_once_with_hard_bracket(self, tmp_path) -> None:
        env = build(tmp_path)
        env["runner"].preflight()
        next_bar(env)
        rep = env["runner"].step()
        assert rep.new_bars == 1 and rep.cycle.order.accepted
        sent = env["fake"].sent[-1]
        assert sent["sl"] > 0 and sent["tp"] > 0 and round(sent["sl"], 2) == sent["sl"]
        assert env["fake"].calls.count("SEND_BRACKET") == 1

    def test_gap_catch_up_decides_only_on_latest_bar(self, tmp_path) -> None:
        env = build(tmp_path)
        env["runner"].preflight()
        next_bar(env, k=3)
        rep = env["runner"].step()
        assert rep.new_bars == 3 and any("caught up 2" in n for n in rep.notes)
        assert env["fake"].calls.count("SEND_BRACKET") == 1
        assert env["runner"].bars[-1].ts == env["added"][-1].ts

    def test_link_down_waits_and_recovers(self, tmp_path) -> None:
        env = build(tmp_path)
        runner, fake, mon, clock = env["runner"], env["fake"], env["monitor"], env["clock"]
        runner.preflight()
        fake.fail_ping = True
        for _ in range(2):
            clock.mono += 1
            mon.beat()
        next_bar(env)
        fake.calls.clear()
        rep = runner.step()
        assert rep.state is RunnerState.WAITING_LINK and rep.cycle is None
        assert "RATES" not in fake.calls and "SEND_BRACKET" not in fake.calls
        fake.fail_ping = False
        mon.beat()
        rep = runner.step()
        assert rep.state is RunnerState.RUNNING and rep.cycle is not None
        assert events(env, "live_link_down") and events(env, "live_link_restored")

    def test_kill_switch_halts_loop(self, tmp_path) -> None:
        env = build(tmp_path)
        env["runner"].preflight()
        KillSwitch.write_sentinel(tmp_path / "KILL", "operator", source="cli")
        next_bar(env)
        rep = env["runner"].step()
        assert rep.state is RunnerState.HALTED and "operator" in rep.notes[0]
        assert "SEND_BRACKET" not in env["fake"].calls

    def test_weekend_flatten_happens_between_bars(self, tmp_path) -> None:
        env = build(tmp_path, compliance=ComplianceGate(WeekendGuard()))
        runner, fake, clock = env["runner"], env["fake"], env["clock"]
        runner.preflight()
        next_bar(env)
        assert runner.step().cycle.order.accepted and fake.open
        clock.t = datetime(2026, 9, 25, 21, 0, 30, tzinfo=UTC)  # Friday 21:00:30, no bar closes now
        rep = runner.step()
        assert rep.new_bars == 0 and fake.open == [] and "FLATTEN_ALL" in fake.calls
        assert events(env, "compliance_flatten")

    def test_daily_loss_limit_checked_between_bars(self, tmp_path) -> None:
        env = build(tmp_path)
        runner, fake = env["runner"], env["fake"]
        runner.preflight()
        next_bar(env)
        runner.step()
        fake.equity = 50_000.0 * (1 - 0.026)  # -2.6% intrabar
        rep = runner.step()
        assert rep.new_bars == 0 and "FLATTEN_ALL" in fake.calls and fake.open == []
        assert events(env, "daily_loss_freeze")

    def test_run_loop_survives_transport_errors_and_stops_cleanly(self, tmp_path) -> None:
        env = build(tmp_path)
        runner, fake = env["runner"], env["fake"]
        seen: list[Any] = []
        original = runner.step

        def flaky_step() -> Any:
            if len(seen) == 1:
                fake.fail_rates = True
            elif len(seen) == 2:
                fake.fail_rates = False
            return original()

        runner.step = flaky_step  # type: ignore[method-assign]
        runner.run(max_steps=4, on_step=seen.append)
        assert len(seen) == 4 and any("transport error" in n for r in seen for n in r.notes)
        assert runner.state is RunnerState.STOPPED and events(env, "live_stop")
