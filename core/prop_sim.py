"""Prop-firm challenge Monte Carlo: probability of passing, and whether it is worth paying for.

The input is a strategy's *closed-trade R multiples* grouped by trading day (from
:func:`core.backtest.run_backtest`). Days are resampled with a stationary block bootstrap
(blocks keep regime clustering and losing streaks intact), each trade is re-sized at the
chosen risk per trade, and the resulting paths are run through the firm's rules:

* Phase 1 / Phase 2 profit targets, minimum trading days, optional time limit.
* Max daily loss vs. the day-start balance, and max (static) loss vs. the initial balance,
  both expressed as fractions of the *initial* balance (FTMO-style).
* A funded stage with periodic payouts at the firm's profit split; the fee is refunded
  with the first payout.

Trader policy mirrors ``core.risk_engine``: no new trade once daily headroom is below one
full stop (so open-trade drawdown cannot breach the daily limit), and a trade that would
put the max-loss line within one stop is downsized to 80% of the remaining headroom.
Breaches therefore come from gapped stops and from the max-loss line when sizing
floors out, not from the simulator ignoring floating P&L.

**Read the zero-edge null alongside every result.** Two symmetric barriers (+10% / -10%)
give a pure coin-flip roughly 50% odds of hitting the target first, so a pass rate on its
own says nothing about edge. Gate on ``edge = pass_rate - null_pass_rate`` and on the
expected value of the fee, never on pass rate alone.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from core.backtest import Trade

DayTrades = tuple[float, ...]  # R multiples closed on one trading day, in order


@dataclass(frozen=True, slots=True)
class PhaseRules:
    """One evaluation phase."""

    profit_target_pct: float
    min_trading_days: int = 4
    max_days: int | None = None  # None = unlimited (FTMO since 2024); ``horizon_days`` still caps it


@dataclass(frozen=True, slots=True)
class FirmRules:
    """Challenge rules. Loss limits are fractions of the *initial* balance.

    Defaults approximate the FTMO 2-step challenge; verify against the firm's current
    terms before relying on a number — fees and rules change.
    """

    name: str = "FTMO 2-step"
    account_size: float = 10_000.0
    fee: float = 170.0
    phases: tuple[PhaseRules, ...] = (PhaseRules(0.10), PhaseRules(0.05))
    max_daily_loss_pct: float = 0.05
    max_loss_pct: float = 0.10
    profit_split: float = 0.80
    payout_every_days: int = 21  # trading days between payouts
    funded_days: int = 252  # how long the funded account is simulated
    refund_fee_on_first_payout: bool = True
    horizon_days: int = 500  # hard cap on any phase without a time limit


@dataclass(frozen=True, slots=True)
class TraderPolicy:
    """How the trader sizes. ``risk_pct`` is the loss of one full stop-out (1R)."""

    risk_pct: float = 0.005
    headroom_use: float = 0.8  # max fraction of remaining max-loss headroom one trade may risk
    min_risk_fraction: float = 0.25  # refuse trades smaller than this fraction of risk_pct


@dataclass(frozen=True, slots=True)
class PathOutcome:
    passed: bool
    fail_reason: str  # "", "daily_loss", "max_loss", "timeout", "starved"
    phase_reached: int
    eval_days: int
    funded_payout: float  # gross cash to trader from the funded stage, incl. fee refund


@dataclass(frozen=True, slots=True)
class SimSummary:
    n_paths: int
    risk_pct: float
    pass_rate: float
    phase1_pass_rate: float
    fail_daily_loss: float
    fail_max_loss: float
    fail_starved: float
    fail_timeout: float
    median_days_to_pass: float | None
    mean_funded_payout: float
    ev_per_attempt: float  # mean payout - fee, in account currency
    fee: float
    outcomes: tuple[PathOutcome, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, float | int | None]:
        return {"risk_pct": self.risk_pct, "n_paths": self.n_paths,
                "pass_rate": round(self.pass_rate, 4), "phase1_pass_rate": round(self.phase1_pass_rate, 4),
                "fail_daily_loss": round(self.fail_daily_loss, 4), "fail_max_loss": round(self.fail_max_loss, 4),
                "fail_starved": round(self.fail_starved, 4), "fail_timeout": round(self.fail_timeout, 4), "median_days_to_pass": self.median_days_to_pass,
                "mean_funded_payout": round(self.mean_funded_payout, 2),
                "ev_per_attempt": round(self.ev_per_attempt, 2)}


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def daily_r_series(trades: Sequence[Trade], trading_days: Sequence[date]) -> list[DayTrades]:
    """Group closed-trade R multiples by exit date over every trading day in ``trading_days``.

    Days with no closed trade are kept as empty tuples: they are part of the time it
    takes to pass, and dropping them would make the strategy look faster than it is.
    """
    by_day: dict[date, list[float]] = {d: [] for d in trading_days}
    for t in sorted(trades, key=lambda t: t.exit_ts):
        d = t.exit_ts.date()
        if d in by_day and math.isfinite(t.r_multiple):
            by_day[d].append(t.r_multiple)
    return [tuple(by_day[d]) for d in sorted(by_day)]


def demeaned(days: Sequence[DayTrades]) -> list[DayTrades]:
    """Zero-edge null: the same trades, frequency and clustering, with mean R removed."""
    rs = [r for day in days for r in day]
    if not rs:
        return [tuple(day) for day in days]
    mu = statistics.fmean(rs)
    return [tuple(r - mu for r in day) for day in days]


def block_bootstrap(days: Sequence[DayTrades], n: int, rng: random.Random, mean_block: int = 10) -> list[DayTrades]:
    """Stationary (Politis-Romano) bootstrap of ``n`` days with geometric block lengths."""
    if not days:
        raise ValueError("no trading days to resample")
    p_new = 1.0 / max(1, mean_block)
    m = len(days)
    out: list[DayTrades] = []
    i = rng.randrange(m)
    for _ in range(n):
        out.append(days[i])
        i = rng.randrange(m) if rng.random() < p_new else (i + 1) % m
    return out


# --------------------------------------------------------------------------------------
# Account simulation
# --------------------------------------------------------------------------------------
class _DayStream:
    """Endless bootstrap stream so every phase draws fresh, continuing days."""

    def __init__(self, days: Sequence[DayTrades], rng: random.Random, mean_block: int) -> None:
        self._days, self._rng, self._block = days, rng, mean_block
        self._buf: list[DayTrades] = []

    def next(self) -> DayTrades:
        if not self._buf:
            self._buf = block_bootstrap(self._days, 256, self._rng, self._block)
            self._buf.reverse()
        return self._buf.pop()


def _trade_risk(equity: float, day_start: float, rules: FirmRules, policy: TraderPolicy) -> float:
    """Fraction of the initial balance risked on the next trade (0 = sit out)."""
    daily_floor = day_start - rules.max_daily_loss_pct
    max_floor = 1.0 - rules.max_loss_pct
    risk = policy.risk_pct * equity
    if equity - risk <= daily_floor:  # one full stop could breach the daily limit: stop for the day
        return 0.0
    room = equity - max_floor
    if equity - risk <= max_floor + 1e-12:
        risk = policy.headroom_use * room
    return risk if risk >= policy.min_risk_fraction * policy.risk_pct else 0.0


def _run_days(stream: _DayStream, rules: FirmRules, policy: TraderPolicy, target: float | None,
              min_days: int, max_days: int, payout_every: int | None = None) -> tuple[str, int, float]:
    """Run one phase (``target`` set) or the funded stage (``payout_every`` set).

    Equity is a fraction of the initial balance. Returns (status, days_used, payouts) where
    status is "passed", "daily_loss", "max_loss", "starved" (cannot size a trade above the
    max-loss line; a dead account in practice), "timeout" or "survived" (funded).
    """
    equity, traded_days, payouts = 1.0, 0, 0.0
    max_floor = 1.0 - rules.max_loss_pct
    for day in range(1, max_days + 1):
        if _trade_risk(equity, equity, rules, policy) <= 0:  # too close to max loss to size a trade
            return "starved", day, payouts
        day_start = equity
        daily_floor = day_start - rules.max_daily_loss_pct
        traded = False
        for r in stream.next():
            risk = _trade_risk(equity, day_start, rules, policy)
            if risk <= 0:
                break
            equity += r * risk
            traded = True
            if equity <= max_floor:
                return "max_loss", day, payouts
            if equity <= daily_floor:
                return "daily_loss", day, payouts
        traded_days += traded
        if target is not None and equity >= 1.0 + target and traded_days >= min_days:
            return "passed", day, payouts
        if payout_every and day % payout_every == 0 and equity > 1.0:
            payouts += (equity - 1.0) * rules.profit_split
            equity = 1.0
    return ("survived" if target is None else "timeout"), max_days, payouts


def simulate_path(days: Sequence[DayTrades], rules: FirmRules, policy: TraderPolicy, rng: random.Random,
                  mean_block: int = 10) -> PathOutcome:
    """One full attempt: every evaluation phase, then the funded stage if passed."""
    stream = _DayStream(days, rng, mean_block)
    used = 0
    for k, phase in enumerate(rules.phases, start=1):
        cap = min(phase.max_days or rules.horizon_days, rules.horizon_days)
        status, n, _ = _run_days(stream, rules, policy, phase.profit_target_pct, phase.min_trading_days, cap)
        used += n
        if status != "passed":
            return PathOutcome(False, status, k, used, 0.0)
    _, _, payout_frac = _run_days(stream, rules, policy, None, 0, rules.funded_days, rules.payout_every_days)
    cash = payout_frac * rules.account_size
    if cash > 0 and rules.refund_fee_on_first_payout:
        cash += rules.fee
    return PathOutcome(True, "", len(rules.phases), used, cash)


def run_monte_carlo(days: Sequence[DayTrades], rules: FirmRules | None = None,
                    policy: TraderPolicy | None = None, n_paths: int = 5_000, seed: int = 7,
                    mean_block: int = 10) -> SimSummary:
    """Simulate ``n_paths`` independent challenge attempts."""
    rules = rules or FirmRules()
    policy = policy or TraderPolicy()
    rng = random.Random(seed)
    outs = tuple(simulate_path(days, rules, policy, rng, mean_block) for _ in range(n_paths))
    n = len(outs)
    passed = [o for o in outs if o.passed]
    p1 = sum(1 for o in outs if o.passed or o.phase_reached > 1) / n
    mean_payout = statistics.fmean(o.funded_payout for o in outs)
    return SimSummary(
        n_paths=n, risk_pct=policy.risk_pct, pass_rate=len(passed) / n, phase1_pass_rate=p1,
        fail_daily_loss=sum(o.fail_reason == "daily_loss" for o in outs) / n,
        fail_max_loss=sum(o.fail_reason == "max_loss" for o in outs) / n,
        fail_starved=sum(o.fail_reason == "starved" for o in outs) / n,
        fail_timeout=sum(o.fail_reason == "timeout" for o in outs) / n,
        median_days_to_pass=statistics.median(o.eval_days for o in passed) if passed else None,
        mean_funded_payout=mean_payout, ev_per_attempt=mean_payout - rules.fee, fee=rules.fee, outcomes=outs,
    )
