"""Q-Agent Critic: tries to kill every candidate before it reaches execution.

A candidate is promoted only if it survives *all* of the following on a chronological
in-sample / out-of-sample split:

1. **Trade frequency** — at least ``min_oos_trades`` (default 15) OOS trades.
2. **Lo-corrected significance** — the OOS per-trade Sharpe ratio's t-statistic,
   using Lo (2002)'s asymptotic standard error ``sqrt((1 + SR²/2) / n)`` inflated by a
   Bartlett-weighted autocorrelation factor, must clear a hurdle.
3. **Snooping hurdle** — that hurdle is ``max(min_t_stat, z_{1 - α / (2·trials)})``:
   a Bonferroni bound over every hypothesis the generator has emitted so far.
4. **In-sample decay** — OOS Sharpe must retain at least ``1 - max_is_decay`` of the
   IS Sharpe, and OOS expectancy must be positive.
5. **Plausibility** — win rates or profit factors that are too good are overfitting
   red flags.
6. **Risk compliance** — no risk-engine breach in either sample and OOS max drawdown
   below a fraction of the trailing-drawdown cap.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from agent.generator import StrategyCandidate
from core.backtest import BacktestConfig, BacktestResult, run_backtest
from core.market_data import Bar


@dataclass(frozen=True, slots=True)
class LoStatistic:
    """Autocorrelation-corrected Sharpe significance (Lo, 2002)."""

    n: int
    sharpe: float
    standard_error: float
    t_stat: float
    lags: int
    inflation: float


def _autocorr(xs: Sequence[float], lag: int, mean: float, var: float) -> float:
    n = len(xs)
    if var <= 0 or lag >= n:
        return 0.0
    cov = sum((xs[i] - mean) * (xs[i - lag] - mean) for i in range(lag, n)) / n
    return cov / var


def lo_corrected_t_stat(returns: Sequence[float], max_lag: int | None = None) -> LoStatistic:
    """t-statistic of the Sharpe ratio with Lo (2002) serial-correlation correction.

    ``SE(SR) = sqrt(η · (1 + SR² / 2) / n)`` with
    ``η = 1 + 2 · Σ_{k=1..q} (1 - k/(q+1)) · ρ_k`` (Bartlett weights).

    η is floored at 1.0: negative autocorrelation is never allowed to *flatter* the
    statistic — the critic is intentionally one-sided conservative.

    Args:
        returns: Per-period (here: per-trade) returns.
        max_lag: Lags ``q``; defaults to the Newey-West rule ``floor(4 (n/100)^{2/9})``.
    """
    n = len(returns)
    if n < 3:
        return LoStatistic(n, 0.0, math.inf, 0.0, 0, 1.0)
    mean = statistics.fmean(returns)
    sd = statistics.stdev(returns)
    if sd <= 0:
        return LoStatistic(n, 0.0, math.inf, 0.0, 0, 1.0)
    sr = mean / sd
    q = max_lag if max_lag is not None else max(1, int(4 * (n / 100) ** (2 / 9)))
    q = max(0, min(q, n - 2))
    var_pop = statistics.pvariance(returns, mu=mean)
    eta = 1.0 + 2.0 * sum((1 - k / (q + 1)) * _autocorr(returns, k, mean, var_pop) for k in range(1, q + 1))
    eta = max(1.0, eta)
    se = math.sqrt(eta * (1 + 0.5 * sr * sr) / n)
    return LoStatistic(n, sr, se, sr / se, q, eta)


def snooping_hurdle(trials: int, alpha: float = 0.05, floor: float = 2.0) -> float:
    """Two-sided Bonferroni critical value over ``trials`` tests, floored at ``floor``."""
    trials = max(1, trials)
    return max(floor, NormalDist().inv_cdf(1 - alpha / (2 * trials)))


@dataclass(frozen=True, slots=True)
class CriticConfig:
    """Thresholds used by :class:`QCritic`."""

    oos_fraction: float = 0.30
    min_oos_trades: int = 15
    min_t_stat: float = 2.0
    family_alpha: float = 0.05
    apply_snooping_correction: bool = True
    max_is_decay: float = 0.50
    max_plausible_win_rate: float = 0.85
    max_plausible_profit_factor: float = 6.0
    max_oos_dd_fraction_of_cap: float = 0.75

    def __post_init__(self) -> None:
        if not 0.1 <= self.oos_fraction <= 0.6:
            raise ValueError("oos_fraction must be in [0.1, 0.6]")
        if self.min_oos_trades < 15:
            raise ValueError("min_oos_trades may not be set below the 15-trade floor")


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    """The critic's decision about one candidate."""

    candidate: StrategyCandidate
    promoted: bool
    reasons: tuple[str, ...]
    is_result: BacktestResult
    oos_result: BacktestResult
    lo: LoStatistic
    hurdle: float
    trials: int

    @property
    def score(self) -> float:
        """Ranking key among promoted candidates: excess t over the hurdle."""
        return self.lo.t_stat - self.hurdle

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "candidate": self.candidate.to_dict(),
            "promoted": self.promoted,
            "reasons": list(self.reasons),
            "in_sample": self.is_result.metrics.to_dict(),
            "out_of_sample": self.oos_result.metrics.to_dict(),
            "lo_t_stat": round(self.lo.t_stat, 3),
            "lo_inflation": round(self.lo.inflation, 3),
            "hurdle": round(self.hurdle, 3),
            "trials": self.trials,
        }


@dataclass
class QCritic:
    """Walk-forward critic. Stateless apart from its configuration."""

    config: CriticConfig = field(default_factory=CriticConfig)

    def split_index(self, n_bars: int) -> int:
        """Index of the first out-of-sample bar."""
        return int(n_bars * (1 - self.config.oos_fraction))

    def review(self, candidate: StrategyCandidate, bars: Sequence[Bar], backtest: BacktestConfig,
               trials: int, symbol: str = "SYM") -> CriticVerdict:
        """Backtest ``candidate`` IS and OOS and return a verdict.

        The OOS run receives the IS history for indicator warm-up but may only open
        trades from the split onward (``start_index``), so no OOS trade is informed by
        a parameter choice made on OOS data.
        """
        cfg = self.config
        split = self.split_index(len(bars))
        strat = candidate.build()
        is_res = run_backtest(bars[:split], strat, backtest, symbol)
        oos_res = run_backtest(bars, strat, backtest, symbol, start_index=split)
        oos_trades = list(oos_res.trades)  # already restricted to the OOS window
        oos_rets = [t.return_on_equity for t in oos_trades]
        lo = lo_corrected_t_stat(oos_rets)
        hurdle = (snooping_hurdle(trials, cfg.family_alpha, cfg.min_t_stat)
                  if cfg.apply_snooping_correction else cfg.min_t_stat)

        reasons: list[str] = []
        m_is, m_oos = is_res.metrics, oos_res.metrics
        n_oos = len(oos_trades)
        if n_oos < cfg.min_oos_trades:
            reasons.append(f"trade frequency: {n_oos} OOS trades < {cfg.min_oos_trades}")
        if lo.t_stat < hurdle:
            reasons.append(f"significance: Lo t={lo.t_stat:.2f} < hurdle {hurdle:.2f} ({trials} trials)")
        oos_expectancy = statistics.fmean(t.r_multiple for t in oos_trades) if oos_trades else 0.0
        if oos_expectancy <= 0:
            reasons.append(f"OOS expectancy {oos_expectancy:.2f}R <= 0")
        if m_is.sharpe_annualised > 0:
            retained = m_oos.sharpe_annualised / m_is.sharpe_annualised
            if retained < 1 - cfg.max_is_decay:
                reasons.append(f"in-sample decay: OOS Sharpe retains {retained:.0%} of IS "
                               f"({m_is.sharpe_annualised:.2f} -> {m_oos.sharpe_annualised:.2f})")
        else:
            reasons.append(f"in-sample Sharpe {m_is.sharpe_annualised:.2f} <= 0: no edge to validate")
        oos_wins = sum(1 for t in oos_trades if t.pnl > 0)
        if n_oos and oos_wins / n_oos > cfg.max_plausible_win_rate:
            reasons.append(f"implausible OOS win rate {oos_wins / n_oos:.0%}")
        if m_is.profit_factor > cfg.max_plausible_profit_factor:
            reasons.append(f"implausible IS profit factor {m_is.profit_factor:.2f}")
        if is_res.breached or oos_res.breached:
            reasons.append("risk engine breach during backtest")
        dd_cap = backtest.limits.max_trailing_drawdown_pct * 100 * cfg.max_oos_dd_fraction_of_cap
        if m_oos.max_drawdown_pct > dd_cap:
            reasons.append(f"OOS max drawdown {m_oos.max_drawdown_pct:.2f}% > {dd_cap:.2f}%")

        return CriticVerdict(candidate, not reasons, tuple(reasons), is_res, oos_res, lo, hurdle, trials)
