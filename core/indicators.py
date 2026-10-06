"""Dependency-free technical indicators operating on plain sequences.

Every series function returns a list aligned with its input, using ``math.nan`` for
the warm-up region, and only ever uses data at or before each index (no look-ahead).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from core.market_data import Bar


def true_range(bars: Sequence[Bar]) -> list[float]:
    """True range per bar; the first bar uses high - low."""
    out: list[float] = []
    for i, b in enumerate(bars):
        if i == 0:
            out.append(b.high - b.low)
        else:
            pc = bars[i - 1].close
            out.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    return out


def atr_series(bars: Sequence[Bar], period: int = 14) -> list[float]:
    """Wilder's Average True Range."""
    if period <= 0:
        raise ValueError("period must be positive")
    tr = true_range(bars)
    out = [math.nan] * len(bars)
    if len(bars) < period:
        return out
    value = sum(tr[:period]) / period
    out[period - 1] = value
    for i in range(period, len(bars)):
        value = (value * (period - 1) + tr[i]) / period
        out[i] = value
    return out


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """Exponential moving average seeded with an SMA of the first ``period`` values."""
    if period <= 0:
        raise ValueError("period must be positive")
    out = [math.nan] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    value = sum(values[:period]) / period
    out[period - 1] = value
    for i in range(period, len(values)):
        value = alpha * values[i] + (1 - alpha) * value
        out[i] = value
    return out


def donchian(bars: Sequence[Bar], period: int, end: int | None = None) -> tuple[float, float]:
    """Highest high / lowest low of the ``period`` bars *before* index ``end``.

    ``end`` defaults to the last bar, so the channel excludes the bar being tested.
    """
    end = len(bars) - 1 if end is None else end
    start = end - period
    if start < 0:
        return math.nan, math.nan
    window = bars[start:end]
    return max(b.high for b in window), min(b.low for b in window)


def percentile_rank(history: Sequence[float], value: float) -> float:
    """Fraction of finite ``history`` values strictly below ``value`` (0..1)."""
    finite = [h for h in history if math.isfinite(h)]
    if not finite:
        return math.nan
    return sum(1 for h in finite if h < value) / len(finite)
