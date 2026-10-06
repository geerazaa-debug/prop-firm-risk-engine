"""aegis-verifier evidence for the prop-firm orchestrator's hard risk engine.

Audit:  python -m aegis_verifier.cli.main audit --target <path-to>/prop_firm_orchestrator

CONTRACTS encode the prop-firm rules as invariants of the real engine (core/risk_engine.py).
aegis injects its non-overridable extrema (0, negatives, +/-inf, NaN, None, float extremes,
10**30) into every argument; each call must either refuse explicitly (ValueError / TypeError)
or return a result that still satisfies the invariants:

* sized_trade   - an approved trade risks <= risk_per_trade_pct of equity, and a full stop-out
                  still leaves the reserved cushion above the daily-loss floor;
* equity_mark   - a mark at/below the daily-loss floor always freezes or terminates, and a
                  non-finite mark always terminates;
* ratchet_*     - the trailing stop never loosens;
* bracket_long  - SL < entry < TP, and tick rounding never tightens the stop or extends the target.

No EVIDENCE yet: Pillar 2 needs the return series of the strategy chosen for deployment.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from aegis_verifier.pillar_1_deterministic.boundary_checker import FunctionContract, Invariant
from aegis_verifier.pillar_4_multi_agent.critic_red_team import Claim
from core.risk_engine import (
    AccountState, AccountStatus, AssetSpec, RatchetStop, RiskAction, RiskLimits, Side, hard_bracket,
    mark_to_market, ratchet_stop, size_position,
)

_LIMITS = RiskLimits()
_T0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)   # a Monday, mid-session: no day roll
_EPS = 1e-6


# ---------------------------------------------------------------- probes over the real engine

def sized_trade(equity, entry, stop, leverage):
    """size_position for a gold-like contract on a fresh account. Returns (units, risk_amount, equity)."""
    state = AccountState.open(equity, _T0, _LIMITS)
    asset = AssetSpec(symbol="XAUUSD", leverage=leverage, contract_size=100.0, lot_step=0.01,
                      min_lot=0.01, price_tick=0.01)
    size = size_position(state, _LIMITS, asset, entry, stop)
    return size.units, size.risk_amount, state.equity


def equity_mark(day_start_equity, mark):
    """mark_to_market one tick after opening. Returns (status, action, daily_floor)."""
    state = AccountState.open(day_start_equity, _T0, _LIMITS)
    evaluation = mark_to_market(state, mark, _T0 + timedelta(seconds=1), _LIMITS)
    return evaluation.state.status.value, evaluation.action.value, state.daily_floor(_LIMITS)


def ratchet_long(entry, stop, bar_high, bar_low, atr):
    state = RatchetStop.open(Side.LONG, entry, stop)
    return state.stop, ratchet_stop(state, bar_high, bar_low, atr).stop


def ratchet_short(entry, stop, bar_high, bar_low, atr):
    state = RatchetStop.open(Side.SHORT, entry, stop)
    return state.stop, ratchet_stop(state, bar_high, bar_low, atr).stop


def bracket_long(entry, stop, take_profit, price_tick):
    asset = AssetSpec(symbol="XAUUSD", leverage=100.0, contract_size=100.0, price_tick=price_tick)
    levels = hard_bracket(Side.LONG, entry, stop, take_profit, asset)
    return levels.stop, levels.take_profit


# ---------------------------------------------------------------- invariants

def _risk_within_budget(args, result):
    units, risk, equity = result
    if units == 0:
        return risk == 0
    return units > 0 and risk <= _LIMITS.risk_per_trade_pct * equity * (1 + _EPS)


def _stop_out_keeps_cushion(args, result):
    units, risk, equity = result
    if units == 0:
        return True
    daily_floor = equity * (1 - _LIMITS.daily_loss_limit_pct)
    cushion = _LIMITS.min_budget_floor_pct * equity
    return equity - risk >= daily_floor + cushion - _EPS * equity


def _breach_never_missed(args, result):
    day_start, mark = args
    status, action, floor = result
    if not math.isfinite(mark):
        return action == RiskAction.TERMINATE_AND_FLATTEN.value
    if mark <= floor:
        return status != AccountStatus.ACTIVE.value and action != RiskAction.NONE.value
    return True


def _long_stop_never_loosens(args, result):
    before, after = result
    return after >= before


def _short_stop_never_loosens(args, result):
    before, after = result
    return after <= before


def _bracket_sane(args, result):
    entry, stop, take_profit, _tick = args
    new_stop, new_tp = result
    return new_stop < entry < new_tp and new_stop <= stop + _EPS and new_tp <= take_profit + _EPS


CONTRACTS = (
    FunctionContract("sized_trade", sized_trade, (100_000.0, 2_000.0, 1_990.0, 100.0),
                     invariants=(Invariant("risk_within_budget", _risk_within_budget),
                                 Invariant("stop_out_keeps_cushion", _stop_out_keeps_cushion))),
    FunctionContract("equity_mark", equity_mark, (100_000.0, 97_400.0),
                     invariants=(Invariant("breach_never_missed", _breach_never_missed),)),
    FunctionContract("ratchet_long", ratchet_long, (2_000.0, 1_990.0, 2_030.0, 2_005.0, 5.0),
                     invariants=(Invariant("stop_never_loosens", _long_stop_never_loosens),)),
    FunctionContract("ratchet_short", ratchet_short, (2_000.0, 2_010.0, 1_995.0, 1_970.0, 5.0),
                     invariants=(Invariant("stop_never_loosens", _short_stop_never_loosens),)),
    FunctionContract("bracket_long", bracket_long, (2_000.0, 1_990.0, 2_030.0, 0.01),
                     invariants=(Invariant("bracket_sane", _bracket_sane),)),
)

CLAIMS = (
    Claim("Hard risk engine survives every boundary extremum without violating a prop-firm rule",
          "pillar_1_deterministic.violations", "==", 0),
    Claim("No function exceeds the blocking complexity threshold (CC 20)",
          "pillar_3_complexity.static.max_cyclomatic", "<=", 20),
)
