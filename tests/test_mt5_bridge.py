"""MT5 bridge: payload construction, transport, heartbeat/reconnect, adapter, EA parity."""
from __future__ import annotations

import itertools
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.coordinator import TradingCoordinator
from core.audit import AuditTrail
from core.risk_engine import AssetSpec, BracketError, Side, hard_bracket
from execution.base import BracketOrder, OrderRejected
from execution.kill_switch import KillSwitch
from execution.mt5_adapter import Mt5Adapter
from execution.mt5_bridge import (
    ACTIONS,
    IDEMPOTENT_ACTIONS,
    PROTOCOL_VERSION,
    HeartbeatConfig,
    HeartbeatMonitor,
    LinkState,
    Mt5Error,
    Mt5Timeout,
    PayloadError,
    SocketTimeout,
    ZmqMt5Transport,
    build_bracket_payload,
    encode,
    make_request,
    sanitize_comment,
)
from tests.conftest import AlwaysLongStrategy, FakeMt5Transport, open_regime_filter, trend_bars

ROOT = Path(__file__).resolve().parents[1]
EA_SOURCE = (ROOT / "mql5" / "ZeroMQ_Listener.mq5").read_text(encoding="utf-8")
GOLD = AssetSpec("XAUUSD", leverage=30, contract_size=100, lot_step=0.01, min_lot=0.01, price_tick=0.01)


def order(**kw: Any) -> BracketOrder:
    base: dict[str, Any] = {"symbol": "XAUUSD", "side": Side.LONG, "units": 0.25, "entry_price": 2000.0,
                            "stop_loss": 1980.0, "take_profit": 2040.0, "client_id": "qm-abc123"}
    base.update(kw)
    return BracketOrder(**base)


# ------------------------------------------------------------------ payload construction
class TestBracketPayload:
    def test_payload_carries_sl_and_tp(self) -> None:
        p = build_bracket_payload(order(), volume=0.25, digits=2, magic=7, deviation=20)
        assert p["sl"] == 1980.0 and p["tp"] == 2040.0
        assert p["side"] == "BUY" and p["volume"] == 0.25 and p["magic"] == 7
        assert p["client_id"] == p["comment"] == "qm-abc123"

    def test_sell_side(self) -> None:
        p = build_bracket_payload(order(side=Side.SHORT, stop_loss=2020.0, take_profit=1960.0), 0.1, 2, 1, 20)
        assert p["side"] == "SELL" and p["sl"] == 2020.0 and p["tp"] == 1960.0

    @pytest.mark.parametrize(("sl", "tp"), [
        (None, 2040.0), (1980.0, None), (0.0, 2040.0), (1980.0, 0.0), (math.nan, 2040.0),
        (1980.0, math.inf), (-5.0, 2040.0), (2010.0, 2040.0), (1980.0, 1990.0),
    ])
    def test_sl_tp_never_missing_or_invalid(self, sl: Any, tp: Any) -> None:
        # Duck-typed order that bypasses BracketOrder's own validation: the payload
        # builder must still refuse.
        bad = SimpleNamespace(symbol="XAUUSD", side=Side.LONG, entry_price=2000.0, stop_loss=sl,
                              take_profit=tp, client_id="x")
        with pytest.raises(PayloadError):
            build_bracket_payload(bad, 0.1, 2, 1, 20)  # type: ignore[arg-type]

    def test_volume_must_be_positive(self) -> None:
        with pytest.raises(PayloadError):
            build_bracket_payload(order(), 0.0, 2, 1, 20)

    def test_prices_rounded_to_symbol_digits(self) -> None:
        p = build_bracket_payload(order(stop_loss=1980.123456, take_profit=2040.98765), 0.1, 2, 1, 20)
        assert p["sl"] == 1980.12 and p["tp"] == 2040.99

    def test_request_envelope_is_flat_versioned_and_authenticated(self) -> None:
        fields = build_bracket_payload(order(), 0.25, 2, 7, 20)
        req = make_request("SEND_BRACKET", "s3cret", **fields)
        assert req["v"] == PROTOCOL_VERSION and req["token"] == "s3cret" and req["action"] == "SEND_BRACKET"
        decoded = json.loads(encode(req))
        assert all(not isinstance(v, (dict, list)) for v in decoded.values())
        assert '"sl":1980.0' in encode(req) and '"tp":2040.0' in encode(req)
        assert make_request("PING", "t")["id"] != make_request("PING", "t")["id"]

    @pytest.mark.parametrize("fields", [{"nested": {"a": 1}}, {"arr": [1]}, {"s": 'quo"te'},
                                        {"s": "back\\slash"}, {"x": math.nan}])
    def test_encoder_rejects_what_the_ea_cannot_parse(self, fields: dict) -> None:
        with pytest.raises(PayloadError):
            make_request("PING", "t", **fields)

    def test_unknown_action_rejected(self) -> None:
        with pytest.raises(PayloadError):
            make_request("WITHDRAW", "t")

    def test_comment_sanitised(self) -> None:
        assert sanitize_comment('qm-"x"\\y' + "z" * 50) == "qm-xy" + "z" * 26
        assert len(sanitize_comment("a" * 100)) == 31


class TestHardBracket:
    def test_levels_rounded_conservatively(self) -> None:
        lv = hard_bracket(Side.LONG, 2000.0, 1980.123456, 2040.987654, GOLD)
        assert lv.stop == pytest.approx(1980.12)  # rounded away from entry
        assert lv.take_profit == pytest.approx(2040.98)  # rounded toward entry
        assert lv.adjusted
        short = hard_bracket(Side.SHORT, 2000.0, 2019.871, 1959.019, GOLD)
        assert short.stop == pytest.approx(2019.88) and short.take_profit == pytest.approx(1959.02)

    def test_min_stop_distance_widens(self) -> None:
        asset = AssetSpec("X", price_tick=0.01, min_stop_distance=5.0)
        lv = hard_bracket(Side.LONG, 100.0, 98.0, 101.0, asset)
        assert lv.stop == pytest.approx(95.0) and lv.take_profit == pytest.approx(105.0)

    @pytest.mark.parametrize(("stop", "tp"), [(None, 110.0), (90.0, None), (math.nan, 110.0), (0.0, 110.0),
                                              (101.0, 110.0), (90.0, 99.0)])
    def test_invalid_levels_raise(self, stop: Any, tp: Any) -> None:
        with pytest.raises(BracketError):
            hard_bracket(Side.LONG, 100.0, stop, tp, GOLD)


# ------------------------------------------------------------------ ZeroMQ transport
class FakeReqSocket:
    """Scripted REQ socket: each entry of ``script`` handles one send/recv pair."""

    def __init__(self, script: list[Any], log: list[dict]) -> None:
        self.script = script
        self.log = log
        self.pending: dict | None = None
        self.closed = False

    def send_string(self, data: str) -> None:
        self.pending = json.loads(data)
        self.log.append(self.pending)

    def recv_string(self) -> str:
        step = self.script.pop(0)
        req = self.pending or {}
        if step == "timeout":
            raise SocketTimeout("recv timeout")
        if step == "wrong_id":
            return json.dumps({"ok": True, "id": "stale", "data": {}})
        if step == "garbage":
            return "not json"
        if isinstance(step, str) and step.startswith("error:"):
            return json.dumps({"ok": False, "id": req["id"], "error": step[6:]})
        return json.dumps({"ok": True, "id": req["id"], "data": step})

    def close(self) -> None:
        self.closed = True


def transport(script: list[Any]) -> tuple[ZmqMt5Transport, list[dict], list[FakeReqSocket]]:
    log: list[dict] = []
    sockets: list[FakeReqSocket] = []

    def factory() -> FakeReqSocket:
        s = FakeReqSocket(script, log)
        sockets.append(s)
        return s

    return ZmqMt5Transport(token="tok", retries=2, socket_factory=factory), log, sockets


class TestZmqTransport:
    def test_round_trip(self) -> None:
        t, log, _ = transport([{"trade_allowed": True}])
        assert t.ping() == {"trade_allowed": True}
        assert log[0]["action"] == "PING" and log[0]["token"] == "tok"

    def test_idempotent_action_retried_on_fresh_socket(self) -> None:
        t, log, sockets = transport(["timeout", "timeout", {"trade_allowed": True}])
        assert t.ping()["trade_allowed"]
        assert len(log) == 3 and len(sockets) == 3  # lazy pirate: new socket after each timeout
        assert sockets[0].closed and sockets[1].closed

    def test_send_bracket_is_never_retried(self) -> None:
        t, log, _ = transport(["timeout", {"retcode": 10009}])
        with pytest.raises(Mt5Timeout):
            t.send_bracket(build_bracket_payload(order(), 0.25, 2, 1, 20))
        assert [r["action"] for r in log] == ["SEND_BRACKET"]
        assert "SEND_BRACKET" not in IDEMPOTENT_ACTIONS

    def test_flatten_all_is_retried(self) -> None:
        t, log, _ = transport(["timeout", {"closed": 2, "cancelled": 0, "failed": []}])
        assert t.flatten_all(None, 20)["closed"] == 2
        assert [r["action"] for r in log] == ["FLATTEN_ALL", "FLATTEN_ALL"]
        assert log[0]["magic"] == -1  # account-wide

    def test_exhausted_retries_raise_timeout(self) -> None:
        t, _, _ = transport(["timeout", "timeout", "timeout"])
        with pytest.raises(Mt5Timeout):
            t.ping()

    def test_stale_reply_rejected_and_socket_reset(self) -> None:
        t, _, sockets = transport(["wrong_id"])
        with pytest.raises(Mt5Error, match="id mismatch"):
            t.ping()
        assert sockets[0].closed

    def test_ea_error_and_malformed_reply(self) -> None:
        t, _, _ = transport(["error:SL and TP are mandatory", "garbage"])
        with pytest.raises(Mt5Error, match="SL and TP are mandatory"):
            t.account_info()
        with pytest.raises(Mt5Error, match="malformed"):
            t.account_info()


# ------------------------------------------------------------------ heartbeat & reconnect
class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


class TestHeartbeatMonitor:
    def make(self) -> tuple[HeartbeatMonitor, FakeMt5Transport, FakeClock, list]:
        fake, clock, changes = FakeMt5Transport(), FakeClock(), []
        mon = HeartbeatMonitor(fake, HeartbeatConfig(interval_s=1.0, max_missed=3, backoff_initial_s=0.5,
                                                     backoff_max_s=4.0), clock=clock,
                               on_state_change=lambda a, b: changes.append((a, b)))
        return mon, fake, clock, changes

    def test_up_and_healthy(self) -> None:
        mon, _, clock, _ = self.make()
        assert mon.beat() is LinkState.UP and mon.healthy()
        clock.t += 3.5  # no beats for longer than interval * max_missed
        assert not mon.healthy()

    def test_misses_degrade_then_down_with_reconnect_and_backoff(self) -> None:
        mon, fake, clock, changes = self.make()
        mon.beat()
        fake.fail_ping = True
        assert mon.beat() is LinkState.DEGRADED
        assert mon.beat() is LinkState.DEGRADED
        assert mon.beat() is LinkState.DOWN and fake.reconnects == 1
        clock.t += 0.2
        mon.beat()
        assert fake.reconnects == 1  # still inside the 0.5 s backoff
        delays = []
        for _ in range(6):
            before = fake.reconnects
            while fake.reconnects == before:
                clock.t += 0.1
                mon.beat()
            delays.append(round(clock.t, 1))
        gaps = [round(b - a, 1) for a, b in itertools.pairwise(delays)]
        assert gaps[-1] == pytest.approx(4.0, abs=0.11)  # exponential backoff capped at 4 s
        assert gaps == sorted(gaps)
        assert (LinkState.UP, LinkState.DEGRADED) in changes and (LinkState.DEGRADED, LinkState.DOWN) in changes

    def test_recovery_resets(self) -> None:
        mon, fake, _, changes = self.make()
        fake.fail_ping = True
        for _ in range(3):
            mon.beat()
        fake.fail_ping = False
        assert mon.beat() is LinkState.UP and mon.missed == 0 and mon.healthy()
        assert changes[-1] == (LinkState.DOWN, LinkState.UP)

    def test_trading_disabled_terminal_side_is_degraded(self) -> None:
        mon, fake, _, _ = self.make()
        fake.trade_allowed = False
        assert mon.beat() is LinkState.DEGRADED and not mon.healthy()

    def test_background_thread_beats(self) -> None:
        fake = FakeMt5Transport()
        mon = HeartbeatMonitor(fake, HeartbeatConfig(interval_s=0.01))
        mon.start()
        deadline = time.monotonic() + 2
        while fake.calls.count("PING") < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        mon.stop()
        assert fake.calls.count("PING") >= 3 and mon.state is LinkState.UP


# ------------------------------------------------------------------ adapter
class TestMt5AdapterSafety:
    def connected(self, **kw: Any) -> tuple[Mt5Adapter, FakeMt5Transport]:
        fake = FakeMt5Transport(**kw)
        adapter = Mt5Adapter(fake)
        adapter.connect()
        return adapter, fake

    def test_unhealthy_link_refuses_orders(self) -> None:
        fake = FakeMt5Transport()
        mon = HeartbeatMonitor(fake, HeartbeatConfig(max_missed=1))
        adapter = Mt5Adapter(fake, monitor=mon)
        adapter._connected = True  # connect() would start the background thread
        fake.fail_ping = True
        mon.beat()
        res = adapter.place_bracket(order())
        assert not res.accepted and "link not healthy" in res.message
        assert "SEND_BRACKET" not in fake.calls

    def test_timeout_reconciled_as_filled(self) -> None:
        adapter, fake = self.connected()
        fake.timeout_send, fake.fill_on_timeout = True, True
        res = adapter.place_bracket(order())
        assert res.accepted and "reconciled: position found" in res.message
        assert fake.calls.count("SEND_BRACKET") == 1  # never re-sent

    def test_timeout_reconciled_as_not_filled(self) -> None:
        adapter, fake = self.connected()
        fake.timeout_send = True
        res = adapter.place_bracket(order())
        assert not res.accepted and "no position opened" in res.message
        assert fake.calls.count("SEND_BRACKET") == 1

    def test_flatten_is_one_command_and_kill_switch_cancels_first(self, tmp_path) -> None:
        adapter, fake = self.connected()
        adapter.place_bracket(order())
        adapter.place_bracket(order(symbol="EURUSD", client_id="qm-2", entry_price=1.1, stop_loss=1.09,
                                    take_profit=1.12))
        kill = KillSwitch([adapter], sentinel_path=tmp_path / "K")
        rep = kill.trip("test")
        assert rep.clean and rep.flattened == 2
        assert fake.calls[-2:] == ["CANCEL_ALL", "FLATTEN_ALL"]
        assert fake.calls.count("FLATTEN_ALL") == 1 and fake.open == []

    def test_flatten_failure_is_reported(self) -> None:
        adapter, fake = self.connected()
        fake.flatten_all = lambda magic, deviation: {"closed": 0, "cancelled": 0, "failed": [42]}  # type: ignore
        with pytest.raises(Mt5Error, match="42"):
            adapter.flatten_all()
        rep = KillSwitch([adapter], retry_delay_s=0).trip("x")
        assert not rep.clean

    def test_amend_stop_never_loosens(self) -> None:
        adapter, fake = self.connected()
        adapter.place_bracket(order())
        assert adapter.amend_stop("XAUUSD", 1990.0)
        assert fake.open[0]["sl"] == 1990.0 and fake.open[0]["tp"] == 2040.0  # TP preserved
        assert not adapter.amend_stop("XAUUSD", 1985.0)

    def test_limit_entries_refused(self) -> None:
        adapter, _ = self.connected()
        with pytest.raises(OrderRejected):
            adapter.place_bracket(order(entry_type="limit"))


class TestCoordinatorEnrichment:
    def test_every_order_reaching_zmq_has_risk_engine_sl_tp(self, tmp_path) -> None:
        fake = FakeMt5Transport()
        adapter = Mt5Adapter(fake)
        adapter.connect()
        audit = AuditTrail()
        coord = TradingCoordinator("XAUUSD", GOLD, adapter, KillSwitch(sentinel_path=tmp_path / "K"),
                                   [AlwaysLongStrategy()], regime_filter=open_regime_filter(), audit=audit)
        bars = trend_bars()
        rep = coord.on_bar(bars)
        assert rep.order is not None and rep.order.accepted
        sent = adapter.last_payload
        assert sent is not None and fake.sent == [sent]
        close = bars[-1].close
        expected = hard_bracket(Side.LONG, close, close - 19.876543, close + 40.987654, GOLD)
        assert sent["sl"] == pytest.approx(expected.stop) and sent["tp"] == pytest.approx(expected.take_profit)
        assert sent["sl"] > 0 and sent["tp"] > 0
        # Raw strategy levels were NOT sent verbatim: they were rounded by the risk engine.
        assert sent["sl"] != round(close - 19.876543, 6)
        submitted = [r for r in audit.tail(50) if r.event == "bracket_submitted"][-1]
        assert submitted.payload["levels_adjusted"] is True


# ------------------------------------------------------------------ EA parity (static)
class TestEaSourceParity:
    def test_ea_handles_exactly_the_python_actions(self) -> None:
        handled = set(re.findall(r'action == "([A-Z_]+)"', EA_SOURCE))
        assert handled == set(ACTIONS)

    def test_ea_enforces_brackets_in_a_single_request(self) -> None:
        assert "SL and TP are mandatory" in EA_SOURCE
        assert "g_trade.Buy(volume, symbol, 0.0, sl, tp, comment)" in EA_SOURCE
        assert "g_trade.Sell(volume, symbol, 0.0, sl, tp, comment)" in EA_SOURCE
        assert "never strips a bracket" in EA_SOURCE

    def test_ea_defaults_match_python(self) -> None:
        assert '"tcp://127.0.0.1:5555"' in EA_SOURCE and '"tcp://127.0.0.1:5556"' in EA_SOURCE
        assert f"#define PROTOCOL_VERSION {PROTOCOL_VERSION}" in EA_SOURCE
        # Python weekday Fri=4 is MQL day_of_week 5; flatten time must equal the Python default.
        from core.compliance import FRIDAY_FLATTEN_UTC

        assert re.search(rf"InpFlattenMinuteUTC\s*=\s*{FRIDAY_FLATTEN_UTC.minute};", EA_SOURCE)
        for pat in (r"InpFlattenDow\s*=\s*5;", rf"InpFlattenHourUTC\s*=\s*{FRIDAY_FLATTEN_UTC.hour};",
                    r"InpReopenDow\s*=\s*0;",
                    r"InpReopenHourUTC\s*=\s*22;"):
            assert re.search(pat, EA_SOURCE), pat

    def test_ea_fails_closed_without_token(self) -> None:
        assert 'if(InpAuthToken == "")' in EA_SOURCE and "INIT_PARAMETERS_INCORRECT" in EA_SOURCE


class TestMqlZmqPatcher:
    def test_patch_text(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("patcher", ROOT / "mql5" / "tools" / "patch_mql_zmq.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = "int zmq_bind(intptr_t s,const char &addr[]);\n   char arr[]; uchar ok[]; // char in comment\ncharset x;\n"
        out, n = mod.patch_text(src)
        assert n == 2
        assert "const uchar &addr[]" in out and "uchar arr[]" in out and "uchar ok[]" in out
        assert "// char in comment" in out and "charset x" in out
        assert mod.patch_text(out)[1] == 0  # idempotent
