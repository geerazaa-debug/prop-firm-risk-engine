"""Invariant tests for the Hard Risk Boundary: DLL, trailing drawdown, sizing, ratchet stop."""
from __future__ import annotations

import dataclasses
import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from agent.coordinator import TradingCoordinator
from core.risk_engine import (
    HARD_DAILY_LOSS_LIMIT_PCT,
    HARD_MAX_TRAILING_DRAWDOWN_PCT,
    AccountState,
    AccountStatus,
    AssetSpec,
    ExposureSnapshot,
    RatchetConfig,
    RatchetStop,
    RiskAction,
    RiskConfigError,
    RiskLimits,
    Side,
    mark_to_market,
    pre_trade_check,
    ratchet_stop,
    size_position,
    trading_day_of,
)
from core.strategy_brain import TrendBreakoutStrategy
from execution.base import BracketOrder, PaperBroker
from execution.kill_switch import KillSwitch
from tests.conftest import bars_from_closes

T0 = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)
LIMITS = RiskLimits()
# Zero-cost asset: isolates the pure fixed-fractional math.
SPOT = AssetSpec("TEST", leverage=1.0, contract_size=1.0, lot_step=0.001, min_lot=0.001,
                 commission_bps=0.0, slippage_bps=0.0)
GOLD = AssetSpec("XAUUSD", leverage=30, contract_size=100, lot_step=0.01, min_lot=0.01)  # default costs


def fresh(equity: float = 100_000.0) -> AccountState:
    return AccountState.open(equity, T0, LIMITS)


# ------------------------------------------------------------------ hard ceilings
class TestHardCeilings:
    def test_defaults_are_the_hard_limits(self) -> None:
        assert LIMITS.daily_loss_limit_pct == pytest.approx(0.025)
        assert LIMITS.max_trailing_drawdown_pct == pytest.approx(0.06)

    @pytest.mark.parametrize("kwargs", [
        {"daily_loss_limit_pct": HARD_DAILY_LOSS_LIMIT_PCT + 1e-6},
        {"daily_loss_limit_pct": 0.05},
        {"max_trailing_drawdown_pct": HARD_MAX_TRAILING_DRAWDOWN_PCT + 1e-6},
        {"max_trailing_drawdown_pct": 0.10},
        {"risk_per_trade_pct": 0.02},
        {"max_open_risk_pct": 0.025},  # open risk may not reach the DLL
        {"headroom_buffer": 1.5},
        {"min_budget_floor_pct": 0.002},  # below the 0.25% hard minimum
        {"min_budget_floor_pct": 0.02},  # leaves no room for a standard trade under the DLL
        {"min_trade_risk_fraction": 0.0},
        {"day_rollover_tz": "Mars/Olympus_Mons"},
    ])
    def test_limits_can_never_be_loosened(self, kwargs: dict) -> None:
        with pytest.raises(RiskConfigError):
            RiskLimits(**kwargs)

    def test_limits_may_be_tightened(self) -> None:
        tight = RiskLimits(daily_loss_limit_pct=0.02, max_trailing_drawdown_pct=0.04,
                           risk_per_trade_pct=0.0025, max_open_risk_pct=0.01)
        assert tight.daily_loss_limit_pct == 0.02

    def test_state_and_limits_are_immutable(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            LIMITS.daily_loss_limit_pct = 0.05  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            fresh().status = AccountStatus.ACTIVE  # type: ignore[misc]


# ------------------------------------------------------------------ daily loss limit
class TestDailyLossLimit:
    def test_loss_just_below_limit_keeps_trading(self) -> None:
        ev = mark_to_market(fresh(), 100_000 * (1 - 0.0249), T0 + timedelta(hours=1), LIMITS)
        assert ev.action is RiskAction.NONE
        assert ev.state.trading_allowed

    def test_touching_limit_freezes_and_flattens(self) -> None:
        ev = mark_to_market(fresh(), 97_500.0, T0 + timedelta(hours=1), LIMITS)
        assert ev.action is RiskAction.FREEZE_AND_FLATTEN
        assert ev.state.status is AccountStatus.DAILY_FROZEN
        assert not ev.state.trading_allowed
        assert pre_trade_check(ev.state, LIMITS).approved is False

    def test_freeze_action_fires_once_per_day(self) -> None:
        ev1 = mark_to_market(fresh(), 97_000.0, T0 + timedelta(hours=1), LIMITS)
        ev2 = mark_to_market(ev1.state, 96_900.0, T0 + timedelta(hours=2), LIMITS)
        assert ev1.action is RiskAction.FREEZE_AND_FLATTEN
        assert ev2.action is RiskAction.NONE
        assert ev2.state.status is AccountStatus.DAILY_FROZEN

    def test_recovery_intraday_does_not_unfreeze(self) -> None:
        ev1 = mark_to_market(fresh(), 97_400.0, T0 + timedelta(hours=1), LIMITS)
        ev2 = mark_to_market(ev1.state, 99_900.0, T0 + timedelta(hours=3), LIMITS)
        assert ev2.state.status is AccountStatus.DAILY_FROZEN

    def test_new_day_unfreezes_with_new_baseline(self) -> None:
        ev1 = mark_to_market(fresh(), 97_400.0, T0 + timedelta(hours=1), LIMITS)
        ev2 = mark_to_market(ev1.state, 97_400.0, T0 + timedelta(days=1), LIMITS)
        assert ev2.state.status is AccountStatus.ACTIVE
        assert ev2.state.day_start_equity == pytest.approx(97_400.0)
        # The new day's DLL is measured from the new baseline.
        ev3 = mark_to_market(ev2.state, 97_400.0 * (1 - 0.025), T0 + timedelta(days=1, hours=1), LIMITS)
        assert ev3.action is RiskAction.FREEZE_AND_FLATTEN

    def test_profit_earlier_in_day_does_not_extend_dll(self) -> None:
        # DLL is measured from day-start equity, not from the intraday peak.
        s = mark_to_market(fresh(), 101_000.0, T0 + timedelta(hours=1), LIMITS).state
        ev = mark_to_market(s, 98_000.0, T0 + timedelta(hours=2), LIMITS)
        assert ev.action is RiskAction.NONE  # -2.0% vs day start 100k

    def test_rollover_respects_prop_firm_timezone(self) -> None:
        prague = RiskLimits(day_rollover_tz="Europe/Prague")
        summer_2230_utc = datetime(2024, 7, 1, 22, 30, tzinfo=UTC)  # 00:30 CEST next day
        assert trading_day_of(summer_2230_utc, "UTC").day == 1
        assert trading_day_of(summer_2230_utc, prague.day_rollover_tz).day == 2


# ------------------------------------------------------------------ trailing drawdown
class TestTrailingDrawdown:
    def test_drawdown_is_measured_from_high_water_mark(self) -> None:
        s = mark_to_market(fresh(), 110_000.0, T0 + timedelta(days=1), LIMITS).state
        s = mark_to_market(s, 106_000.0, T0 + timedelta(days=2), LIMITS).state
        ev = mark_to_market(s, 103_400.0, T0 + timedelta(days=3), LIMITS)  # -6.0% from 110k, still > initial
        assert ev.action is RiskAction.TERMINATE_AND_FLATTEN
        assert ev.state.status is AccountStatus.BREACHED

    def test_breach_is_terminal_across_days_and_recovery(self) -> None:
        s = mark_to_market(fresh(), 110_000.0, T0 + timedelta(days=1), LIMITS).state
        s = mark_to_market(s, 106_000.0, T0 + timedelta(days=2), LIMITS).state
        s = mark_to_market(s, 103_000.0, T0 + timedelta(days=3), LIMITS).state
        for d in range(4, 10):
            ev = mark_to_market(s, 120_000.0, T0 + timedelta(days=d), LIMITS)
            s = ev.state
            assert s.status is AccountStatus.BREACHED
            assert not pre_trade_check(s, LIMITS).approved
            assert not size_position(s, LIMITS, SPOT, 100.0, 99.0).approved

    def test_drawdown_takes_precedence_over_daily(self) -> None:
        ev = mark_to_market(fresh(), 93_000.0, T0 + timedelta(hours=1), LIMITS)
        assert ev.state.status is AccountStatus.BREACHED

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_equity_fails_closed(self, bad: float) -> None:
        ev = mark_to_market(fresh(), bad, T0 + timedelta(hours=1), LIMITS)
        assert ev.action is RiskAction.TERMINATE_AND_FLATTEN
        assert ev.state.status is AccountStatus.BREACHED


# ------------------------------------------------------------------ sizing
class TestPositionSizing:
    def test_fixed_fractional_risk(self) -> None:
        size = size_position(fresh(), LIMITS, SPOT, entry=100.0, stop=98.0)
        # 0.5% of 100k = $500 risk / $2 per unit = 250 units
        assert size.approved and size.units == pytest.approx(250.0)
        assert size.risk_amount == pytest.approx(500.0)
        assert size.binding_constraint == "risk_budget"

    def test_high_leverage_assets_are_scaled_down(self) -> None:
        low = AssetSpec("A", leverage=30.0, lot_step=0.001, min_lot=0.001)
        high = AssetSpec("B", leverage=100.0, lot_step=0.001, min_lot=0.001)
        s_low = size_position(fresh(), LIMITS, low, 100.0, 98.0)
        s_high = size_position(fresh(), LIMITS, high, 100.0, 98.0)
        assert s_high.units == pytest.approx(s_low.units * 30.0 / 100.0, rel=1e-3)

    def test_notional_cap_binds_for_tight_stops(self) -> None:
        size = size_position(fresh(), LIMITS, SPOT, entry=100.0, stop=99.99)
        assert size.binding_constraint in {"notional_cap", "margin_cap"}
        assert size.notional <= LIMITS.max_notional_exposure_multiple * 100_000 + 1e-6

    def test_margin_cap(self) -> None:
        limits = RiskLimits(max_margin_utilisation=0.1)
        size = size_position(AccountState.open(100_000.0, T0, limits), limits, SPOT, 100.0, 99.99)
        assert size.binding_constraint == "margin_cap"
        assert size.margin <= 10_000 + 1e-6

    def test_lot_step_rounds_down_and_min_lot_rejects(self) -> None:
        gold = AssetSpec("XAUUSD", leverage=30, contract_size=100, lot_step=0.01, min_lot=0.01,
                         commission_bps=0.0, slippage_bps=0.0)
        size = size_position(fresh(), LIMITS, gold, 2000.0, 1950.0)  # $500 / $5000 per lot = 0.1
        assert size.units == pytest.approx(0.10)
        tiny = size_position(fresh(1_000.0), LIMITS, gold, 2000.0, 1900.0)
        assert not tiny.approved and tiny.units == 0

    @pytest.mark.parametrize("entry,stop", [(100.0, 100.0), (0.0, 1.0), (math.nan, 1.0), (-5.0, -6.0)])
    def test_invalid_inputs_never_size(self, entry: float, stop: float) -> None:
        assert not size_position(fresh(), LIMITS, SPOT, entry, stop).approved

    def test_open_risk_consumes_budget(self) -> None:
        exp = ExposureSnapshot(open_risk=1_500.0)  # == max_open_risk_pct * 100k
        assert not size_position(fresh(), LIMITS, SPOT, 100.0, 98.0, exp).approved

    def test_budget_shrinks_with_daily_headroom_net_of_reserved_floor(self) -> None:
        s = mark_to_market(fresh(), 98_500.0, T0 + timedelta(hours=1), LIMITS).state  # -1.5% today
        approval = pre_trade_check(s, LIMITS)
        daily_headroom = s.equity - 100_000 * (1 - 0.025)  # $1,000
        floor = 0.005 * s.equity  # $492.50 reserved, never spendable
        assert approval.approved
        assert approval.max_risk == pytest.approx(0.8 * (daily_headroom - floor))

    @pytest.mark.parametrize("asset", [SPOT, GOLD, AssetSpec("FX", leverage=100, contract_size=100_000,
                                                              lot_step=0.01, min_lot=0.01,
                                                              commission_bps=0.7, slippage_bps=0.5)])
    def test_property_no_single_stop_out_can_breach(self, asset: AssetSpec) -> None:
        """Randomised invariant: open risk + the new trade's full loss (stop distance plus
        round-trip costs, recomputed here independently) never crosses either floor."""
        rng = random.Random(1234)
        c = (asset.commission_bps + asset.slippage_bps) / 10_000
        floor_pct = LIMITS.min_budget_floor_pct
        approved = 0
        for _ in range(3_000):
            hwm = rng.uniform(100_000, 130_000)
            equity = hwm * (1 - rng.uniform(0, 0.0599))
            day_start = equity * (1 + rng.uniform(-0.02, 0.0249))
            state = dataclasses.replace(fresh(), equity=equity, high_water_mark=max(hwm, equity),
                                        day_start_equity=day_start)
            open_risk = rng.uniform(0, 0.02) * equity
            entry = rng.uniform(1, 5_000) if asset.contract_size == 1 else rng.uniform(0.5, 2.5)
            stop = entry * (1 - rng.uniform(0.001, 0.1))
            size = size_position(state, LIMITS, asset, entry, stop, ExposureSnapshot(open_risk=open_risk))
            if size.approved:
                approved += 1
                uv = asset.contract_size * asset.quote_to_account
                true_loss = size.units * uv * ((entry - stop) + (entry + stop) * c)
                assert size.risk_amount == pytest.approx(true_loss, rel=1e-9)
                worst = equity - open_risk - true_loss
                # A full stop-out still leaves the reserved cushion above both limits ...
                assert worst - state.daily_floor(LIMITS) >= floor_pct * equity - 1e-6
                assert worst - state.drawdown_floor(LIMITS) >= floor_pct * equity - 1e-6
                # ... and even a 100% gap through the new stop (twice the loss) cannot breach.
                gapped = equity - open_risk - 2 * true_loss
                assert gapped > state.daily_floor(LIMITS)
                assert gapped > state.drawdown_floor(LIMITS)
        assert approved > 100  # the property was actually exercised


class TestTradingCostsAndBudgetFloor:
    def test_round_trip_cost_formula(self) -> None:
        # (2 + 1) bps per side on both legs: 100 oz * (2000 + 1950) * 0.0003
        assert GOLD.round_trip_cost_per_unit(2000.0, 1950.0) == pytest.approx(118.5)
        assert GOLD.loss_per_unit_at_stop(2000.0, 1950.0) == pytest.approx(5_000 + 118.5)

    def test_costs_shrink_size_and_are_counted_as_risk(self) -> None:
        costly = dataclasses.replace(SPOT, commission_bps=5.0, slippage_bps=5.0)
        free = size_position(fresh(), LIMITS, SPOT, 100.0, 98.0)
        paid = size_position(fresh(), LIMITS, costly, 100.0, 98.0)
        per_unit = 2.0 + (100.0 + 98.0) * 0.001  # stop distance + round-trip costs
        assert paid.units == pytest.approx(math.floor(500.0 / per_unit / 0.001) * 0.001)
        assert paid.units < free.units
        assert paid.risk_amount == pytest.approx(paid.units * per_unit)
        assert paid.risk_amount <= 500.0

    def test_negative_costs_rejected(self) -> None:
        with pytest.raises(RiskConfigError):
            AssetSpec("X", commission_bps=-1.0)

    def test_min_lot_floor_blocks_when_one_lot_exceeds_budget(self) -> None:
        # Standard budget $500; one 0.01 lot with a $500 stop loses $500 + $1.05 costs.
        assert pre_trade_check(fresh(), LIMITS).max_risk == pytest.approx(500.0)
        size = size_position(fresh(), LIMITS, GOLD, 2000.0, 1500.0)
        assert not size.approved
        assert size.binding_constraint == "min_lot_floor"
        assert "min-lot" in size.reasons[0]

    def test_costs_alone_can_trip_the_min_lot_floor(self) -> None:
        """Price risk of one min lot ($499) fits the $500 budget; with costs ($500.05) it doesn't."""
        free_gold = dataclasses.replace(GOLD, commission_bps=0.0, slippage_bps=0.0)
        assert size_position(fresh(), LIMITS, free_gold, 2000.0, 1501.0).approved
        blocked = size_position(fresh(), LIMITS, GOLD, 2000.0, 1501.0)
        assert not blocked.approved and blocked.binding_constraint == "min_lot_floor"


def _daily_state(loss_pct: float) -> AccountState:
    return mark_to_market(fresh(), 100_000 * (1 - loss_pct), T0 + timedelta(hours=1), LIMITS).state


def _overall_state(dd_pct: float) -> AccountState:
    # Fresh day (no daily loss) with a drawdown of dd_pct from a 100k high-water mark.
    eq = 100_000 * (1 - dd_pct)
    return dataclasses.replace(fresh(), equity=eq, day_start_equity=eq)


class TestStrongBudgetFloor:
    def test_floor_default_is_one_standard_trade(self) -> None:
        assert LIMITS.min_budget_floor_pct == pytest.approx(LIMITS.risk_per_trade_pct)

    @pytest.mark.parametrize(("make", "level"), [(_daily_state, 0.021), (_overall_state, 0.056)])
    def test_halts_while_cushion_remains(self, make, level: float) -> None:
        """Remaining budget below 0.5% of equity -> halt, for daily and overall limits alike."""
        state = make(level)
        approval = pre_trade_check(state, LIMITS)
        assert not approval.approved and approval.code == "budget_floor"
        size = size_position(state, LIMITS, SPOT, 100.0, 98.0)  # even a tiny, cheap trade
        assert not size.approved and size.binding_constraint == "budget_floor"

    @pytest.mark.parametrize(("make", "level"), [(_daily_state, 0.02), (_overall_state, 0.055)])
    def test_no_micro_lot_edging_just_above_floor(self, make, level: float) -> None:
        """Just above the floor the spendable budget is a few dollars: refused, not micro-sized."""
        approval = pre_trade_check(make(level), LIMITS)
        assert not approval.approved and approval.code == "min_trade_size"
        assert not size_position(make(level), LIMITS, SPOT, 100.0, 98.0).approved

    @pytest.mark.parametrize("mode", ["daily", "overall"])
    @pytest.mark.parametrize("asset", [SPOT, GOLD])
    def test_losing_streak_halts_cleanly_with_safe_buffer(self, mode: str, asset: AssetSpec) -> None:
        """Take the largest allowed trade and lose it (costs included) until the engine refuses.

        ``daily``: every loss lands on the same day (tests the DLL side).
        ``overall``: every loss lands on a new day (daily resets; tests the drawdown side).
        The engine must stop by itself: no freeze, no breach, no token-sized trades, and a
        cushion of at least the floor left above the relevant limit.
        """
        state, trades = fresh(), 0
        entry, stop = (100.0, 98.0) if asset is SPOT else (2000.0, 1990.0)
        for i in range(1, 500):
            size = size_position(state, LIMITS, asset, entry, stop)
            if not size.approved:
                break
            standard = LIMITS.risk_per_trade_pct * state.equity
            assert size.risk_amount >= LIMITS.min_trade_risk_fraction * standard * 0.9  # no micro trades
            ts = T0 + (timedelta(minutes=i) if mode == "daily" else timedelta(days=i))
            ev = mark_to_market(state, state.equity - size.risk_amount, ts, LIMITS)
            assert ev.action is RiskAction.NONE, ev.message  # the engine never reaches a limit
            state, trades = ev.state, trades + 1
        else:
            pytest.fail("engine never halted")

        assert trades > 0
        assert size.binding_constraint in {"budget_floor", "min_trade_size", "min_lot_floor"}
        limit_floor = state.daily_floor(LIMITS) if mode == "daily" else state.drawdown_floor(LIMITS)
        cushion = state.equity - limit_floor
        assert cushion >= LIMITS.min_budget_floor_pct * state.equity - 1e-6
        if mode == "daily":
            assert 0.015 < -state.daily_pnl_pct() <= 0.0201  # used the day, stopped ~0.5% short of 2.5%
        else:
            assert 0.05 < state.drawdown_pct() <= 0.0553  # used the budget, stopped ~0.5% short of 6%

    def test_halt_band_worked_example(self) -> None:
        """The cushion is hundreds of dollars of real protection, not cents."""
        state = _overall_state(0.054)  # $600 left above the 6% line; floor = $473
        assert state.equity - state.drawdown_floor(LIMITS) == pytest.approx(600.0)
        approval = pre_trade_check(state, LIMITS)
        # spendable (600 - 473) * 0.8 = $101 < minimum trade 0.25 * $473 = $118
        assert not approval.approved and approval.code == "min_trade_size"


# ------------------------------------------------------------------ ratchet stop
class TestRatchetStop:
    def test_stop_must_start_on_losing_side(self) -> None:
        with pytest.raises(ValueError):
            RatchetStop.open(Side.LONG, 100.0, 101.0)
        with pytest.raises(ValueError):
            RatchetStop.open(Side.SHORT, 100.0, 99.0)

    def test_breakeven_lock_at_one_r(self) -> None:
        r = RatchetStop.open(Side.LONG, 100.0, 98.0)
        before = ratchet_stop(r, bar_high=101.9, bar_low=100.5, atr=10.0)
        assert not before.breakeven_locked and before.stop == 98.0
        after = ratchet_stop(r, bar_high=102.0, bar_low=100.5, atr=10.0)
        assert after.breakeven_locked
        assert after.stop == pytest.approx(100.0 + 0.05 * 2.0)

    def test_transition_is_non_mutating(self) -> None:
        r = RatchetStop.open(Side.LONG, 100.0, 98.0)
        nxt = ratchet_stop(r, 110.0, 105.0, atr=1.0)
        assert r.stop == 98.0 and nxt.stop > r.stop and nxt is not r

    def test_atr_multiple_decays_with_open_profit(self) -> None:
        cfg = RatchetConfig()
        assert cfg.atr_multiple(1.0) == pytest.approx(3.0)
        assert cfg.atr_multiple(3.0) == pytest.approx(2.0)
        assert cfg.atr_multiple(50.0) == pytest.approx(cfg.min_atr_multiple)

    def test_short_side_mirrors(self) -> None:
        r = RatchetStop.open(Side.SHORT, 100.0, 102.0)
        nxt = ratchet_stop(r, bar_high=99.0, bar_low=95.0, atr=0.5)
        assert nxt.breakeven_locked and nxt.stop < 100.0
        assert nxt.is_hit(bar_high=nxt.stop, bar_low=90.0)

    @pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
    def test_property_stop_never_loosens(self, side: Side) -> None:
        rng = random.Random(99)
        for _ in range(200):
            entry = 100.0
            r = RatchetStop.open(side, entry, entry - side.sign * rng.uniform(0.5, 5))
            price = entry
            for _ in range(100):
                price = max(1.0, price + rng.gauss(0, 1.5))
                hi, lo = price + abs(rng.gauss(0, 1)), price - abs(rng.gauss(0, 1))
                atr = rng.choice([rng.uniform(0.1, 3), math.nan, 0.0])
                nxt = ratchet_stop(r, hi, lo, atr)
                assert (nxt.stop - r.stop) * side.sign >= 0
                r = nxt


# ------------------------------------------------------------------ coordinator integration
def _coordinator(tmp_path) -> tuple[TradingCoordinator, PaperBroker, KillSwitch]:
    kill = KillSwitch(sentinel_path=tmp_path / "KILL")
    broker = PaperBroker(100_000.0, slippage_bps=0, commission_bps=0, guard=kill)
    coord = TradingCoordinator("TEST", SPOT, broker, kill, [TrendBreakoutStrategy()])
    return coord, broker, kill


class TestCoordinatorEnforcement:
    def test_dll_breach_flattens_and_blocks_entries(self, tmp_path) -> None:
        coord, broker, kill = _coordinator(tmp_path)
        bars = bars_from_closes([100.0] * 30)
        broker.mark("TEST", 100.0)
        coord.on_bar(bars)  # initialise day baseline at 100k
        broker.place_bracket(BracketOrder("TEST", Side.LONG, 1_000, 100.0, 90.0, 130.0))
        crash = bars[-1].__class__(bars[-1].ts + timedelta(hours=2), 100.0, 100.0, 97.0, 97.0)
        broker.mark("TEST", 97.0)  # -$3,000 = -3% intraday, stop not yet hit
        rep = coord.on_bar([*bars, crash])
        assert rep.risk_action == RiskAction.FREEZE_AND_FLATTEN.value
        assert broker.positions() == []
        assert coord.state is not None and coord.state.status is AccountStatus.DAILY_FROZEN
        assert not kill.tripped  # a daily freeze is not terminal

    def test_trailing_breach_trips_kill_switch(self, tmp_path) -> None:
        coord, broker, kill = _coordinator(tmp_path)
        bars = bars_from_closes([100.0] * 30)
        broker.mark("TEST", 100.0)
        coord.on_bar(bars)
        broker.place_bracket(BracketOrder("TEST", Side.LONG, 1_000, 100.0, 50.0, 200.0))
        broker.mark("TEST", 93.0)
        crash = bars[-1].__class__(bars[-1].ts + timedelta(hours=1), 100.0, 100.0, 93.0, 93.0)
        rep = coord.on_bar([*bars, crash])
        assert kill.tripped
        assert broker.positions() == []
        assert rep.decision.startswith("halted")
        assert (tmp_path / "KILL").exists()
