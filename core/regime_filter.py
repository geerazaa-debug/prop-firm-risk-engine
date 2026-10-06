"""Regime detection: Kaufman Efficiency Ratio (KER) and volatility regime.

Kaufman Efficiency Ratio over ``N`` bars::

    ER = |Close_t - Close_{t-N}| / sum_{i=t-N+1..t} |Close_i - Close_{i-1}|

ER is 1.0 for a perfectly straight move and approaches 0 in a random, back-and-forth
market. Directional entries are blocked while ``ER < 0.30`` — chop is where breakout
systems bleed through repeated false breaks.

The filter fails closed: with insufficient history it blocks entries.

Choosing ``N``: the ER look-back must be at least as long as the breakout channel it
gates. With a short window (Kaufman's classic N=10) the breakout bar *itself* is a
large share of the window's path, so ER is biased upward exactly at signal time and
the gate lets chop breakouts through. On the synthetic OU-chop regime, N=10 cut
breakout trades by ~0% while N=20 cut them by ~85% and left trend-regime returns
intact — hence the default ``er_period=20``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from core.indicators import atr_series, percentile_rank
from core.market_data import Bar, closes

DEFAULT_ER_THRESHOLD = 0.30


def efficiency_ratio(values: Sequence[float], period: int = 10) -> float:
    """Kaufman Efficiency Ratio of the last ``period`` changes of ``values``.

    Args:
        values: Price series (oldest first); needs at least ``period + 1`` points.
        period: Look-back ``N``.

    Returns:
        ER in ``[0, 1]``. A perfectly flat window (zero path length) returns 0.0 — no
        directional efficiency to exploit.

    Raises:
        ValueError: If ``period`` < 1 or there is not enough data.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if len(values) < period + 1:
        raise ValueError(f"need {period + 1} values, got {len(values)}")
    window = values[-(period + 1):]
    direction = abs(window[-1] - window[0])
    path = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    if path <= 0.0:
        return 0.0
    return min(1.0, direction / path)


def rolling_efficiency_ratio(values: Sequence[float], period: int = 10) -> list[float]:
    """ER at every index (``nan`` during warm-up), computed incrementally in O(n)."""
    n = len(values)
    out = [math.nan] * n
    if n < period + 1:
        return out
    diffs = [0.0] + [abs(values[i] - values[i - 1]) for i in range(1, n)]
    path = sum(diffs[1:period + 1])
    for t in range(period, n):
        if t > period:
            path += diffs[t] - diffs[t - period]
        direction = abs(values[t] - values[t - period])
        out[t] = 0.0 if path <= 1e-15 else min(1.0, direction / path)
    return out


class VolRegime(str, Enum):
    """Volatility regime by percentile rank of ATR% within a look-back window."""

    COMPRESSED = "compressed"
    NORMAL = "normal"
    EXPANDED = "expanded"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RegimeFilterConfig:
    """Configuration of :class:`RegimeFilter`.

    Attributes:
        er_period: KER look-back ``N`` (>= the gated breakout channel; see module doc).
        er_threshold: Minimum ER for directional entries (spec: 0.30).
        atr_period: ATR period used for the volatility regime.
        vol_lookback: Number of ATR% observations the current value is ranked against.
        compressed_pct / expanded_pct / extreme_pct: Percentile-rank cut points.
        block_extreme_vol: Also block entries in the EXTREME volatility regime.
    """

    er_period: int = 20
    er_threshold: float = DEFAULT_ER_THRESHOLD
    atr_period: int = 14
    vol_lookback: int = 100
    compressed_pct: float = 0.20
    expanded_pct: float = 0.80
    extreme_pct: float = 0.97
    block_extreme_vol: bool = True

    @property
    def warmup(self) -> int:
        """Bars needed before the filter can allow an entry."""
        return max(self.er_period + 1, self.atr_period + self.vol_lookback)


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Regime assessment at the latest bar."""

    efficiency_ratio: float
    vol_regime: VolRegime
    atr_pct_rank: float
    allow_directional: bool
    allow_carry: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


class RegimeFilter:
    """Stateless regime gatekeeper evaluated on a window of bars."""

    def __init__(self, config: RegimeFilterConfig | None = None) -> None:
        self.config = config or RegimeFilterConfig()

    @property
    def warmup(self) -> int:
        """Bars required for a full evaluation."""
        return self.config.warmup

    def volatility_regime(self, bars: Sequence[Bar]) -> tuple[VolRegime, float]:
        """Classify the current ATR% against its own recent history."""
        cfg = self.config
        atr = atr_series(bars, cfg.atr_period)
        atr_pct = [a / b.close if math.isfinite(a) else math.nan for a, b in zip(atr, bars, strict=True)]
        current = atr_pct[-1] if atr_pct else math.nan
        history = atr_pct[-(cfg.vol_lookback + 1):-1]
        finite_hist = [h for h in history if math.isfinite(h)]
        if not math.isfinite(current) or len(finite_hist) < max(20, cfg.vol_lookback // 2):
            return VolRegime.UNKNOWN, math.nan
        rank = percentile_rank(finite_hist, current)
        if rank >= cfg.extreme_pct:
            return VolRegime.EXTREME, rank
        if rank >= cfg.expanded_pct:
            return VolRegime.EXPANDED, rank
        if rank < cfg.compressed_pct:
            return VolRegime.COMPRESSED, rank
        return VolRegime.NORMAL, rank

    def evaluate(self, bars: Sequence[Bar]) -> RegimeSnapshot:
        """Assess whether directional and carry entries are allowed at the last bar."""
        cfg = self.config
        reasons: list[str] = []
        px = closes(bars)
        if len(px) < cfg.er_period + 1:
            return RegimeSnapshot(math.nan, VolRegime.UNKNOWN, math.nan, False, False,
                                  (f"insufficient history ({len(px)} bars) - fail closed",))
        er = efficiency_ratio(px, cfg.er_period)
        vol, rank = self.volatility_regime(bars)

        allow_directional = True
        if er < cfg.er_threshold:
            allow_directional = False
            reasons.append(f"chop: ER {er:.2f} < {cfg.er_threshold:.2f}")
        if vol is VolRegime.UNKNOWN:
            allow_directional = False
            reasons.append("volatility regime unknown (warm-up) - fail closed")
        allow_carry = vol is not VolRegime.UNKNOWN
        if cfg.block_extreme_vol and vol is VolRegime.EXTREME:
            allow_directional = False
            allow_carry = False
            reasons.append(f"extreme volatility (ATR% rank {rank:.2f})")
        return RegimeSnapshot(er, vol, rank, allow_directional, allow_carry, tuple(reasons))
