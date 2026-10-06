"""Regression tests for boundary gaps found by the aegis-verifier contracts (aegis_manifest.py).

Each case used to crash with an untyped exception or, worse, be silently accepted.
"""
from __future__ import annotations

import math

import pytest

from core.risk_engine import (
    AssetSpec, BracketError, RatchetStop, RiskConfigError, Side, hard_bracket, ratchet_stop,
)

NAN, INF = math.nan, math.inf


@pytest.mark.parametrize("field", ["leverage", "contract_size", "lot_step", "min_lot", "quote_to_account",
                                   "commission_bps", "slippage_bps", "price_tick", "min_stop_distance"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_asset_spec_rejects_non_finite_fields(field, value):
    """inf leverage used to zero the margin per unit (ZeroDivisionError in sizing); NaN passed every check."""
    with pytest.raises(RiskConfigError):
        AssetSpec(symbol="XAUUSD", **{field: value})


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.0, -5.0, None])
def test_ratchet_rejects_unusable_stop(side, bad):
    """A NaN stop used to be accepted and could never be hit - a position with no stop at all."""
    with pytest.raises((ValueError, TypeError)):
        RatchetStop.open(side, 2_000.0, bad)


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, None])
def test_ratchet_rejects_unusable_entry(bad):
    with pytest.raises((ValueError, TypeError)):
        RatchetStop.open(Side.LONG, bad, 1_990.0)


def test_ratchet_from_a_nan_broker_fill_is_refused():
    """Mirrors agent/coordinator.py: the stop is shifted by (fill - planned entry)."""
    fill, planned_entry, planned_stop = NAN, 2_000.0, 1_990.0
    with pytest.raises(ValueError):
        RatchetStop.open(Side.LONG, fill, planned_stop + (fill - planned_entry))


@pytest.mark.parametrize("entry, stop, take_profit, tick", [
    (2_000.0, 1_990.0, 1.7976931348623157e308, 0.01),   # target / tick overflows
    (2_000.0, 1_990.0, 2_030.0, 5e-324),                 # subnormal tick overflows every level
])
def test_bracket_overflow_is_a_bracket_error(entry, stop, take_profit, tick):
    with pytest.raises(BracketError):
        hard_bracket(Side.LONG, entry, stop, take_profit, AssetSpec(symbol="XAUUSD", price_tick=tick))


def test_valid_brackets_and_ratchets_are_unchanged():
    levels = hard_bracket(Side.LONG, 2_000.0, 1_990.004, 2_030.006, AssetSpec(symbol="XAUUSD", price_tick=0.01))
    assert (levels.stop, levels.take_profit) == (1_990.0, 2_030.0)   # stop rounded away, target toward
    state = RatchetStop.open(Side.LONG, 2_000.0, 1_990.0)
    assert ratchet_stop(state, 2_030.0, 2_005.0, 5.0).stop >= state.stop


def test_short_target_stepped_below_zero_is_a_bracket_error():
    """Found by the on_bar golden master: a coarse tick collapsed a short target onto entry and the
    one-tick step-out landed at entry - tick < 0; hard_bracket only checked the stop."""
    asset = AssetSpec(symbol="XAUUSD", price_tick=5_000.0)
    with pytest.raises(BracketError, match="take-profit"):
        hard_bracket(Side.SHORT, 2_000.0, 2_010.0, 1_970.0, asset)


def test_invalid_order_is_a_veto_not_an_exception(monkeypatch):
    """Defense in depth: BracketOrder validation failures are audited, never raised out of on_bar."""
    from datetime import UTC, datetime, timedelta

    from agent import coordinator as coord_mod
    from core.market_data import Bar
    from core.risk_engine import BracketLevels
    from core.strategy_brain import build_strategy
    from execution.base import PaperBroker
    from execution.kill_switch import KillSwitch

    class Audit:
        def __init__(self):
            self.events = []

        def record(self, event, payload=None):
            self.events.append(event)

    def broken_bracket(side, entry, stop, take_profit, asset):  # levels that BracketOrder must refuse
        return BracketLevels(side, entry, stop, -1.0, True)

    monkeypatch.setattr(coord_mod, "hard_bracket", broken_bracket)
    audit, kill = Audit(), KillSwitch(audit=Audit().record)
    asset = AssetSpec(symbol="XAUUSD", leverage=100.0, contract_size=100.0, lot_step=0.01, min_lot=0.01)
    broker = PaperBroker(100_000.0, unit_value=100.0, guard=kill)
    coord = coord_mod.TradingCoordinator("XAUUSD", asset, broker, kill, [build_strategy("trend_breakout")],
                                         audit=audit)
    t0 = datetime(2026, 1, 5, tzinfo=UTC)
    bars = [Bar(t0 + timedelta(hours=i), 2_000 + i, 2_001 + i, 1_999 + i, 2_000.5 + i, 1.0) for i in range(400)]
    reports = [coord.on_bar(bars[: i + 1]) for i in range(len(bars))]
    vetoes = [r.decision for r in reports if r.decision.startswith("order invalid")]
    assert vetoes, "a breakout signal should have reached order construction"
    assert "order_invalid" in audit.events and not broker.positions()
