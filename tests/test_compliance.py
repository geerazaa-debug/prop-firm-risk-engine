"""Prop-firm session compliance: Friday auto-flatten / weekend block and news blackouts."""
from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
from typing import Any

import pytest

from agent.coordinator import TradingCoordinator
from core.audit import AuditTrail
from core.compliance import FRIDAY_FLATTEN_UTC, ComplianceGate, WeekendGuard, WeekendGuardConfig, parse_hhmm
from core.news_filter import (
    NewsEvent,
    NewsFilter,
    NewsFilterConfig,
    currencies_for,
    load_calendar,
    parse_ff_calendar,
)
from core.risk_engine import AssetSpec, RiskConfigError, Side
from execution.base import BracketOrder, PaperBroker
from execution.kill_switch import KillSwitch
from mcp_server.mcp_bridge import OrchestratorService
from tests.conftest import AlwaysLongStrategy, open_regime_filter, trend_bars

# 2026-09-25 is a Friday.
FRI = datetime(2026, 9, 25, tzinfo=UTC)
WED = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
SPOT = AssetSpec("XAUUSD", leverage=30, contract_size=1, lot_step=0.01, min_lot=0.01,
                 commission_bps=0, slippage_bps=0)


def at(day_offset: int, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return FRI + timedelta(days=day_offset, hours=hh, minutes=mm, seconds=ss)


# ------------------------------------------------------------------ weekend guard
class TestWeekendGuard:
    @pytest.mark.parametrize(("when", "blocked"), [
        (at(0, 19, 29, 59), False),   # Fri 19:29:59 - last second of trading
        (at(0, 19, 30, 0), True),     # Fri 19:30:00 - flatten instant (inclusive), 30 min before 20:00 close
        (at(0, 20, 0, 0), True),      # Fri 20:00:00 - broker close
        (at(0, 23, 30), True),        # Fri late
        (at(1, 12, 0), True),         # Saturday
        (at(2, 21, 59, 59), True),    # Sun just before reopen
        (at(2, 22, 0, 0), False),     # Sun 22:00 reopen (exclusive end)
        (at(3, 9, 0), False),         # Monday
        (WED, False),                 # mid-week
    ])
    def test_default_window(self, when: datetime, blocked: bool) -> None:
        d = WeekendGuard().evaluate(when)
        assert d.must_flatten is blocked and d.allow_new_orders is not blocked
        if blocked:
            assert "weekend guard" in d.reasons[0]

    def test_configurable_cutoff(self) -> None:
        g = WeekendGuard(WeekendGuardConfig(flatten_time=time(21, 0)))
        assert not g.in_blackout(at(0, 20, 59))
        assert g.in_blackout(at(0, 21, 0))

    def test_default_flatten_precedes_broker_close(self) -> None:
        # MetaQuotes-Demo XAUUSD closes Fri 23:00 server time = 20:00 UTC.
        assert FRIDAY_FLATTEN_UTC == time(19, 30)
        assert WeekendGuardConfig().flatten_time == FRIDAY_FLATTEN_UTC
        assert WeekendGuard().in_blackout(at(0, 19, 30)) and not WeekendGuard().in_blackout(at(0, 19, 29, 59))

    @pytest.mark.parametrize(("text", "expected"), [("19:30", time(19, 30)), ("7:05", time(7, 5)),
                                                    ("00:00", time(0, 0))])
    def test_parse_hhmm(self, text: str, expected: time) -> None:
        assert parse_hhmm(text) == expected

    @pytest.mark.parametrize("bad", ["1930", "25:00", "19:60", "", "ab:cd"])
    def test_parse_hhmm_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_hhmm(bad)

    def test_cli_friday_flatten_flag(self) -> None:
        import app.main as cli

        args = cli.build_parser().parse_args(["live", "--friday-flatten", "18:45"])
        assert args.friday_flatten == time(18, 45)
        assert cli.weekend_guard_from_args(args).config.flatten_time == time(18, 45)
        default = cli.build_parser().parse_args(["live"])
        assert default.friday_flatten == FRIDAY_FLATTEN_UTC
        dash = cli.build_parser().parse_args(["dashboard", "--weekend-guard"])
        assert cli.compliance_from_args(dash).weekend.config.flatten_time == FRIDAY_FLATTEN_UTC

    def test_non_utc_zone(self) -> None:
        # Fri 17:00 New York (EDT, UTC-4 in September) == Fri 21:00 UTC.
        g = WeekendGuard(WeekendGuardConfig(flatten_time=time(17, 0), reopen_time=time(17, 0),
                                            tz="America/New_York"))
        assert not g.in_blackout(at(0, 20, 59))
        assert g.in_blackout(at(0, 21, 0))

    def test_non_wrapping_window(self) -> None:
        g = WeekendGuard(WeekendGuardConfig(flatten_weekday=5, flatten_time=time(0, 0),
                                            reopen_weekday=6, reopen_time=time(23, 0)))
        assert g.in_blackout(at(1, 0, 0)) and not g.in_blackout(at(2, 23, 0)) and not g.in_blackout(at(0, 23))

    def test_naive_datetime_is_utc(self) -> None:
        assert WeekendGuard().in_blackout(at(0, 21, 0).replace(tzinfo=None))

    def test_disabled(self) -> None:
        assert not WeekendGuard(WeekendGuardConfig(enabled=False)).in_blackout(at(1, 12))

    @pytest.mark.parametrize("kw", [{"flatten_weekday": 7}, {"reopen_weekday": -1},
                                    {"flatten_weekday": 6, "flatten_time": time(22, 0)},
                                    {"tz": "Nowhere/Nope"}])
    def test_invalid_config(self, kw: dict) -> None:
        with pytest.raises(RiskConfigError):
            WeekendGuardConfig(**kw)


# ------------------------------------------------------------------ news filter
NFP = NewsEvent(datetime(2026, 10, 2, 12, 30, tzinfo=UTC), "USD", "Non-Farm Employment Change", "high")


def news(*events: NewsEvent, **cfg: Any) -> NewsFilter:
    return NewsFilter(list(events) or [NFP], NewsFilterConfig(**cfg))


class TestNewsFilter:
    @pytest.mark.parametrize(("offset_s", "blocked"), [
        (-301, False), (-300, True), (0, True), (300, True), (301, False),
    ])
    def test_five_minute_window_both_sides(self, offset_s: int, blocked: bool) -> None:
        d = news().check(NFP.ts + timedelta(seconds=offset_s), "XAUUSD")
        assert d.blocked is blocked
        if blocked:
            assert d.event == NFP and "Non-Farm" in d.reason

    def test_low_and_medium_impact_ignored(self) -> None:
        f = news(NewsEvent(NFP.ts, "USD", "Crude Oil Inventories", "medium"),
                 NewsEvent(NFP.ts, "USD", "Pending Home Sales", "low"))
        assert not f.check(NFP.ts, "XAUUSD").blocked

    def test_keyword_escalates_mislabelled_events(self) -> None:
        for title in ("CPI m/m", "FOMC Statement", "Federal Funds Rate", "Fed Chair Powell Speaks"):
            f = news(NewsEvent(NFP.ts, "USD", title, "medium"))
            assert f.check(NFP.ts, "XAUUSD").blocked, title

    def test_currency_relevance(self) -> None:
        ecb = NewsEvent(NFP.ts, "EUR", "Main Refinancing Rate", "high")
        f = news(ecb)
        assert f.check(NFP.ts, "EURUSD").blocked
        assert not f.check(NFP.ts, "XAUUSD").blocked
        assert not f.check(NFP.ts, "USDJPY").blocked
        usd = news()
        for sym in ("XAUUSD", "EURUSD", "USDJPY", "BTC/USDT:USDT"):
            assert usd.check(NFP.ts, sym).blocked, sym

    def test_currencies_for(self) -> None:
        assert currencies_for("XAUUSD") == {"XAU", "USD"}
        assert currencies_for("EURUSD.r") == {"EUR", "USD"}
        assert currencies_for("BTC/USDT:USDT") == {"BTC", "USD"}

    def test_stale_or_missing_calendar_fails_closed(self) -> None:
        assert NewsFilter([]).check(WED, "XAUUSD").blocked
        stale = news().check(NFP.ts + timedelta(days=2), "XAUUSD")
        assert stale.blocked and "stale" in stale.reason
        assert not news(fail_closed=False).check(NFP.ts + timedelta(days=2), "XAUUSD").blocked

    def test_explicit_coverage_keeps_calendar_fresh(self) -> None:
        f = NewsFilter([NFP], coverage_end=NFP.ts + timedelta(days=3))
        assert not f.check(NFP.ts + timedelta(days=2), "XAUUSD").blocked

    def test_parse_forex_factory_json(self, tmp_path) -> None:
        rows = [
            {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-10-02T08:30:00-04:00",
             "impact": "High"},
            {"title": "Bank Holiday", "country": "JPY", "date": "not-a-date", "impact": "Holiday"},
            {"title": "CPI y/y", "country": "EUR", "date": "2026-09-30T05:00:00-04:00", "impact": "High"},
        ]
        events = parse_ff_calendar(rows)
        assert [e.currency for e in events] == ["EUR", "USD"]  # sorted, bad row skipped
        assert events[1].ts == datetime(2026, 10, 2, 12, 30, tzinfo=UTC)  # offset converted to UTC
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        assert load_calendar(path) == events

    def test_csv_calendar(self, tmp_path) -> None:
        path = tmp_path / "cal.csv"
        path.write_text("ts,currency,impact,title\n2026-10-02T12:30:00+00:00,USD,High,NFP\n", encoding="utf-8")
        (ev,) = load_calendar(path)
        assert ev.ts == NFP.ts and ev.is_high_impact

    def test_upcoming(self) -> None:
        assert news().upcoming(NFP.ts - timedelta(hours=1), "XAUUSD") == [NFP]
        assert news().upcoming(NFP.ts + timedelta(minutes=1), "XAUUSD") == []


class TestComplianceGate:
    def test_combines_components(self) -> None:
        gate = ComplianceGate(WeekendGuard(), NewsFilter([NFP], coverage_end=at(10, 0)))
        assert gate.evaluate(WED, "XAUUSD").allow_new_orders
        weekend = gate.evaluate(at(1, 12), "XAUUSD")
        assert weekend.must_flatten and not weekend.allow_new_orders
        nfp = gate.evaluate(NFP.ts, "XAUUSD")
        assert not nfp.allow_new_orders and not nfp.must_flatten  # news blocks, never flattens

    def test_empty_gate_allows(self) -> None:
        assert ComplianceGate().evaluate(at(1, 12), "XAUUSD").allow_new_orders


# ------------------------------------------------------------------ coordinator integration
def coordinator(tmp_path, compliance: ComplianceGate, broker: PaperBroker | None = None,
                ) -> tuple[TradingCoordinator, PaperBroker, KillSwitch, AuditTrail]:
    kill = KillSwitch(sentinel_path=tmp_path / "KILL")
    broker = broker or PaperBroker(100_000.0, slippage_bps=0, commission_bps=0, guard=kill)
    audit = AuditTrail()
    coord = TradingCoordinator("XAUUSD", SPOT, broker, kill, [AlwaysLongStrategy()],
                               regime_filter=open_regime_filter(), audit=audit, compliance=compliance)
    return coord, broker, kill, audit


def with_position(broker: PaperBroker, bars: list) -> None:
    broker.mark("XAUUSD", bars[-1].close)
    c = bars[-1].close
    broker.place_bracket(BracketOrder("XAUUSD", Side.LONG, 1.0, c, c - 20, c + 40))


class TestFridayAutoFlatten:
    def test_flattens_at_cutoff_and_blocks_new_orders(self, tmp_path) -> None:
        coord, broker, kill, audit = coordinator(tmp_path, ComplianceGate(WeekendGuard()))
        bars = trend_bars()
        with_position(broker, bars)
        rep = coord.on_bar(bars, now=at(0, 21, 0))
        assert broker.positions() == []
        assert any(r.event == "compliance_flatten" for r in audit.tail(20))
        assert not kill.tripped and not (tmp_path / "KILL").exists()  # scheduled flatten is not a kill
        assert rep.decision.startswith("compliance veto")  # the always-long signal was blocked
        again = coord.on_bar(bars, now=at(1, 10, 0))
        assert again.order is None and broker.positions() == []

    def test_keeps_position_before_cutoff(self, tmp_path) -> None:
        coord, broker, _, audit = coordinator(tmp_path, ComplianceGate(WeekendGuard()))
        bars = trend_bars()
        with_position(broker, bars)
        coord.on_bar(bars, now=at(0, 19, 29))  # one minute before the 19:30 cutoff
        assert len(broker.positions()) == 1
        assert not any(r.event == "compliance_flatten" for r in audit.tail(20))

    def test_trading_resumes_after_reopen(self, tmp_path) -> None:
        coord, broker, _, _ = coordinator(tmp_path, ComplianceGate(WeekendGuard()))
        rep = coord.on_bar(trend_bars(), now=at(2, 22, 0))
        assert rep.order is not None and rep.order.accepted
        assert len(broker.positions()) == 1

    def test_failed_flatten_escalates_to_kill_switch(self, tmp_path) -> None:
        class StuckBroker(PaperBroker):
            def flatten_all(self) -> int:
                raise ConnectionError("terminal offline")

        coord, broker, kill, _ = coordinator(tmp_path, ComplianceGate(WeekendGuard()), StuckBroker(100_000.0))
        bars = trend_bars()
        with_position(broker, bars)
        rep = coord.on_bar(bars, now=at(0, 21, 5))
        assert kill.tripped and rep.decision.startswith("halted")

    def test_default_decision_time_is_bar_close(self, tmp_path) -> None:
        bars = trend_bars()
        assert TradingCoordinator.decision_time(bars) == bars[-1].ts + (bars[-1].ts - bars[-2].ts)


class TestNewsVeto:
    def test_blackout_vetoes_entry(self, tmp_path) -> None:
        gate = ComplianceGate(None, NewsFilter([NFP], coverage_end=NFP.ts + timedelta(days=3)))
        coord, broker, _, audit = coordinator(tmp_path, gate)
        rep = coord.on_bar(trend_bars(), now=NFP.ts - timedelta(minutes=2))
        assert broker.positions() == []
        assert rep.order is None and broker.positions() == []
        veto = [r for r in audit.tail(20) if r.event == "compliance_veto"][-1]
        assert "Non-Farm" in veto.payload["reasons"][0]

    def test_outside_blackout_trades(self, tmp_path) -> None:
        gate = ComplianceGate(None, NewsFilter([NFP], coverage_end=NFP.ts + timedelta(days=3)))
        coord, broker, _, _ = coordinator(tmp_path, gate)
        rep = coord.on_bar(trend_bars(), now=NFP.ts - timedelta(minutes=6))
        assert rep.order is not None and rep.order.accepted and len(broker.positions()) == 1

    def test_news_does_not_flatten_open_positions(self, tmp_path) -> None:
        gate = ComplianceGate(None, NewsFilter([NFP], coverage_end=NFP.ts + timedelta(days=3)))
        coord, broker, _, _ = coordinator(tmp_path, gate)
        bars = trend_bars()
        with_position(broker, bars)
        coord.on_bar(bars, now=NFP.ts)
        assert len(broker.positions()) == 1


class TestSoftFlatten:
    def test_does_not_latch(self, tmp_path) -> None:
        broker = PaperBroker(10_000)
        broker.mark("X", 100.0)
        broker.place_bracket(BracketOrder("X", Side.LONG, 1.0, 100.0, 95.0, 110.0))
        kill = KillSwitch([broker], sentinel_path=tmp_path / "KILL")
        rep = kill.soft_flatten("weekend", source="compliance")
        assert rep.flattened == 1 and rep.clean
        assert not kill.tripped and not (tmp_path / "KILL").exists()


class TestMcpCompliance:
    def service(self, tmp_path, now: datetime, gate: ComplianceGate | None = None) -> OrchestratorService:
        svc = OrchestratorService(tmp_path, clock=lambda: now, compliance=gate)
        svc.fetch_ohlcv("XAUUSD", limit=300)
        return svc

    def test_weekend_blocks_execute_order(self, tmp_path) -> None:
        svc = self.service(tmp_path, at(1, 12))
        px = svc._bars["XAUUSD"][-1].close
        res = svc.execute_order("long", px * 0.98, px * 1.04)
        assert not res["accepted"] and "weekend guard" in res["reason"]
        assert svc.get_risk_metrics()["compliance"]["must_flatten"] is True

    def test_news_blocks_execute_order(self, tmp_path) -> None:
        gate = ComplianceGate(WeekendGuard(), NewsFilter([NFP], coverage_end=NFP.ts + timedelta(days=3)))
        svc = self.service(tmp_path, NFP.ts, gate)
        px = svc._bars["XAUUSD"][-1].close
        res = svc.execute_order("long", px * 0.98, px * 1.04)
        assert not res["accepted"] and "news blackout" in res["reason"]

    def test_midweek_allows(self, tmp_path) -> None:
        svc = self.service(tmp_path, WED)
        px = svc._bars["XAUUSD"][-1].close
        assert svc.execute_order("long", px * 0.98, px * 1.04)["accepted"]
