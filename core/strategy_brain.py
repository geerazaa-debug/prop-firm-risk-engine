"""Alpha strategies and the multi-signal aggregator.

Strategies are pure: ``evaluate(symbol, bars)`` looks only at the supplied window and
returns at most one :class:`Signal` for the *last* bar. Entries are meant to be filled
on the next bar's open by the caller, so no strategy ever sees the bar it trades on.

Included strategies:

* :class:`TrendBreakoutStrategy` — Donchian channel breakout with an ATR stop.
* :class:`GoldTrendPullbackStrategy` — EMA-trend pullback/resumption tuned for XAUUSD's
  tendency to trend in sessions and retest the fast mean.
* :class:`FundingCarryStrategy` — market-neutral perp funding carry (short perp / long
  spot) driven by persistent positive funding.

:class:`SignalAggregator` combines them behind the regime gate: directional signals
require the KER filter to pass; carry signals are market-neutral and bypass the KER
gate but are still blocked in extreme volatility.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from core.indicators import atr_series, donchian, ema_series
from core.market_data import Bar, closes
from core.regime_filter import RegimeFilter, RegimeSnapshot, efficiency_ratio
from core.risk_engine import Side


class SignalKind(str, Enum):
    """Directional signals carry price risk; carry signals carry basis risk."""

    DIRECTIONAL = "directional"
    CARRY = "carry"


@dataclass(frozen=True, slots=True)
class Signal:
    """A trade proposal for the next bar.

    For CARRY signals, ``side`` refers to the perpetual leg (hedged by an opposite
    spot leg) and ``|entry - stop|`` is the tolerated adverse basis move per unit, so
    the same risk engine can size it.
    """

    strategy: str
    symbol: str
    side: Side
    kind: SignalKind
    entry: float
    stop: float
    take_profit: float
    confidence: float
    reason: str
    ts: datetime

    def __post_init__(self) -> None:
        if (self.entry - self.stop) * self.side.sign <= 0:
            raise ValueError(f"{self.strategy}: stop on wrong side of entry")
        if (self.take_profit - self.entry) * self.side.sign <= 0:
            raise ValueError(f"{self.strategy}: take-profit on wrong side of entry")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    @property
    def reward_to_risk(self) -> float:
        """Target distance divided by stop distance."""
        return abs(self.take_profit - self.entry) / abs(self.entry - self.stop)


class Strategy(Protocol):
    """Interface every directional strategy implements."""

    name: str

    @property
    def warmup(self) -> int:
        """Bars required before the strategy can emit a signal."""
        ...

    def params(self) -> dict[str, Any]:
        """Parameters, for audit trails and the critic."""
        ...

    def evaluate(self, symbol: str, bars: Sequence[Bar]) -> Signal | None:
        """Return a signal for the last bar of ``bars`` or None."""
        ...


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True, slots=True)
class TrendBreakoutStrategy:
    """Donchian breakout: close beyond the prior ``channel``-bar extreme.

    Stop is ``stop_atr`` ATRs from entry; take-profit is ``target_r`` times the stop
    distance. Confidence grows with how far the close cleared the channel (in ATRs).
    """

    channel: int = 20
    atr_period: int = 14
    stop_atr: float = 2.0
    target_r: float = 2.0
    allow_short: bool = True
    name: str = "trend_breakout"

    @property
    def warmup(self) -> int:
        return max(self.channel, self.atr_period) + 1

    def params(self) -> dict[str, Any]:
        return {"channel": self.channel, "atr_period": self.atr_period, "stop_atr": self.stop_atr,
                "target_r": self.target_r, "allow_short": self.allow_short}

    def evaluate(self, symbol: str, bars: Sequence[Bar]) -> Signal | None:
        if len(bars) < self.warmup:
            return None
        atr = atr_series(bars, self.atr_period)[-1]
        upper, lower = donchian(bars, self.channel)
        last = bars[-1]
        if not (math.isfinite(atr) and atr > 0 and math.isfinite(upper)):
            return None
        entry = last.close
        dist = self.stop_atr * atr
        if last.close > upper:
            conf = _clamp01(0.5 + 0.5 * (last.close - upper) / atr)
            return Signal(self.name, symbol, Side.LONG, SignalKind.DIRECTIONAL, entry, entry - dist,
                          entry + self.target_r * dist, conf,
                          f"close {entry:.2f} > {self.channel}-bar high {upper:.2f}", last.ts)
        if self.allow_short and last.close < lower and entry - dist > 0:
            conf = _clamp01(0.5 + 0.5 * (lower - last.close) / atr)
            tp = entry - self.target_r * dist
            if tp <= 0:
                return None
            return Signal(self.name, symbol, Side.SHORT, SignalKind.DIRECTIONAL, entry, entry + dist,
                          tp, conf, f"close {entry:.2f} < {self.channel}-bar low {lower:.2f}", last.ts)
        return None


@dataclass(frozen=True, slots=True)
class GoldTrendPullbackStrategy:
    """Trend-pullback for gold: trade resumption after a retest of the fast EMA.

    Long setup: fast EMA above slow EMA, slow EMA rising over ``slope_lookback`` bars,
    the previous bar's low tagged the fast EMA, and the current bar closes above the
    previous bar's high (resumption). Short setup mirrors it.
    """

    fast: int = 20
    slow: int = 50
    slope_lookback: int = 5
    atr_period: int = 14
    stop_atr: float = 1.5
    target_r: float = 2.5
    allow_short: bool = True
    name: str = "gold_pullback"

    @property
    def warmup(self) -> int:
        return max(self.slow + self.slope_lookback, self.atr_period) + 2

    def params(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "slope_lookback": self.slope_lookback,
                "atr_period": self.atr_period, "stop_atr": self.stop_atr, "target_r": self.target_r,
                "allow_short": self.allow_short}

    def evaluate(self, symbol: str, bars: Sequence[Bar]) -> Signal | None:
        if len(bars) < self.warmup:
            return None
        px = closes(bars)
        ef = ema_series(px, self.fast)
        es = ema_series(px, self.slow)
        atr = atr_series(bars, self.atr_period)[-1]
        if not all(math.isfinite(v) for v in (ef[-1], ef[-2], es[-1], es[-1 - self.slope_lookback], atr)):
            return None
        prev, last = bars[-2], bars[-1]
        entry = last.close
        dist = self.stop_atr * atr
        slope = es[-1] - es[-1 - self.slope_lookback]
        spread_atr = abs(ef[-1] - es[-1]) / atr
        if ef[-1] > es[-1] and slope > 0 and prev.low <= ef[-2] and last.close > prev.high:
            return Signal(self.name, symbol, Side.LONG, SignalKind.DIRECTIONAL, entry, entry - dist,
                          entry + self.target_r * dist, _clamp01(0.4 + 0.2 * spread_atr),
                          "uptrend pullback to fast EMA, resumption close", last.ts)
        if (self.allow_short and ef[-1] < es[-1] and slope < 0 and prev.high >= ef[-2]
                and last.close < prev.low and entry - self.target_r * dist > 0):
            return Signal(self.name, symbol, Side.SHORT, SignalKind.DIRECTIONAL, entry, entry + dist,
                          entry - self.target_r * dist, _clamp01(0.4 + 0.2 * spread_atr),
                          "downtrend pullback to fast EMA, resumption close", last.ts)
        return None


@dataclass(frozen=True, slots=True)
class FundingCarryStrategy:
    """Delta-neutral funding carry: short perp + long spot when funding stays positive.

    Fires when the mean of the last ``lookback`` funding prints, annualised, exceeds
    ``min_annualised`` *and* every print in the window is positive (persistence filter:
    a single spike is not a carry regime). The stop encodes tolerated basis blow-out.
    """

    min_annualised: float = 0.12
    lookback: int = 9
    periods_per_year: int = 3 * 365
    basis_stop_pct: float = 0.01
    target_multiple: float = 1.5
    name: str = "funding_carry"

    def params(self) -> dict[str, Any]:
        return {"min_annualised": self.min_annualised, "lookback": self.lookback,
                "periods_per_year": self.periods_per_year, "basis_stop_pct": self.basis_stop_pct}

    def annualised(self, funding_rates: Sequence[float]) -> float:
        """Annualised mean of the last ``lookback`` funding prints."""
        window = funding_rates[-self.lookback:]
        return sum(window) / len(window) * self.periods_per_year if window else 0.0

    def evaluate_funding(self, symbol: str, funding_rates: Sequence[float], mark_price: float,
                         ts: datetime) -> Signal | None:
        """Return a CARRY signal on the perp leg, or None."""
        if len(funding_rates) < self.lookback or mark_price <= 0:
            return None
        window = funding_rates[-self.lookback:]
        ann = self.annualised(funding_rates)
        if ann < self.min_annualised or any(r <= 0 for r in window):
            return None
        stop = mark_price * (1 + self.basis_stop_pct)
        tp = mark_price * (1 - self.basis_stop_pct * self.target_multiple)
        conf = _clamp01(ann / (2 * self.min_annualised))
        return Signal(self.name, symbol, Side.SHORT, SignalKind.CARRY, mark_price, stop, tp, conf,
                      f"funding {ann:.1%} annualised, {self.lookback} consecutive positive prints", ts)


@dataclass(frozen=True, slots=True)
class KerTrendVolStrategy:
    """Multi-filter breakout: efficiency + trend + volatility expansion must all agree.

    Long when, on the last closed bar:

    1. **Breakout** — close above the prior ``channel``-bar high;
    2. **Efficiency** — Kaufman ER over ``er_period`` >= ``er_min`` (stricter than the
       0.30 regime gate: the move itself must be directional, not just the regime);
    3. **Trend** — close above EMA(``trend_ema``) and that EMA rising over 5 bars;
    4. **Volatility expansion** — ATR(``atr_fast``) / ATR(``atr_slow``) >= ``vol_expansion``
       (breakouts out of compression-then-expansion, not out of dying ranges).

    Short mirrors every condition. Stop ``stop_atr`` ATRs, target ``target_r`` R.
    """

    channel: int = 20
    er_period: int = 20
    er_min: float = 0.35
    trend_ema: int = 100
    atr_fast: int = 14
    atr_slow: int = 50
    vol_expansion: float = 1.0
    stop_atr: float = 2.0
    target_r: float = 2.5
    allow_short: bool = True
    name: str = "ker_trend_vol"

    @property
    def warmup(self) -> int:
        return max(self.channel, self.er_period, self.trend_ema + 5, self.atr_slow) + 2

    def params(self) -> dict[str, Any]:
        return {"channel": self.channel, "er_period": self.er_period, "er_min": self.er_min,
                "trend_ema": self.trend_ema, "atr_fast": self.atr_fast, "atr_slow": self.atr_slow,
                "vol_expansion": self.vol_expansion, "stop_atr": self.stop_atr, "target_r": self.target_r,
                "allow_short": self.allow_short}

    def evaluate(self, symbol: str, bars: Sequence[Bar]) -> Signal | None:
        if len(bars) < self.warmup:
            return None
        px = closes(bars)
        atr_f = atr_series(bars, self.atr_fast)[-1]
        atr_s = atr_series(bars, self.atr_slow)[-1]
        ema = ema_series(px, self.trend_ema)
        upper, lower = donchian(bars, self.channel)
        if not all(math.isfinite(v) for v in (atr_f, atr_s, ema[-1], ema[-6], upper, lower)) or atr_s <= 0:
            return None
        er = efficiency_ratio(px, self.er_period)
        expansion = atr_f / atr_s
        if er < self.er_min or expansion < self.vol_expansion:
            return None
        last = bars[-1]
        entry, dist = last.close, self.stop_atr * atr_f
        slope = ema[-1] - ema[-6]
        conf = _clamp01(0.4 + 0.3 * (er - self.er_min) / max(1e-9, 1 - self.er_min)
                        + 0.3 * min(1.0, expansion - 1.0))
        why = f"breakout + ER {er:.2f} + EMA{self.trend_ema} trend + ATR ratio {expansion:.2f}"
        if last.close > upper and last.close > ema[-1] and slope > 0:
            return Signal(self.name, symbol, Side.LONG, SignalKind.DIRECTIONAL, entry, entry - dist,
                          entry + self.target_r * dist, conf, why, last.ts)
        if (self.allow_short and last.close < lower and last.close < ema[-1] and slope < 0
                and entry - self.target_r * dist > 0):
            return Signal(self.name, symbol, Side.SHORT, SignalKind.DIRECTIONAL, entry, entry + dist,
                          entry - self.target_r * dist, conf, why, last.ts)
        return None


# --------------------------------------------------------------------------------------
# Registry & aggregator
# --------------------------------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, Callable[..., Strategy]] = {
    "trend_breakout": TrendBreakoutStrategy,
    "gold_pullback": GoldTrendPullbackStrategy,
    "ker_trend_vol": KerTrendVolStrategy,
}


def build_strategy(name: str, params: Mapping[str, Any] | None = None) -> Strategy:
    """Instantiate a directional strategy by registry name."""
    try:
        factory = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; known: {sorted(STRATEGY_REGISTRY)}") from exc
    return factory(**dict(params or {}))


@dataclass(frozen=True, slots=True)
class AggregatedDecision:
    """Output of :meth:`SignalAggregator.aggregate`."""

    signal: Signal | None
    regime: RegimeSnapshot
    candidates: tuple[Signal, ...] = ()
    rejected: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (strategy, reason)

    @property
    def summary(self) -> str:
        """One-line explanation for audit logs."""
        if self.signal:
            return f"{self.signal.strategy} {self.signal.side.value} conf={self.signal.confidence:.2f}"
        why = "; ".join(f"{s}: {r}" for s, r in self.rejected) or "no setup"
        return f"no trade ({why})"


class SignalAggregator:
    """Combine strategy votes behind the regime gate.

    Rules:
      * Directional candidates are discarded when the regime blocks directional entries
        (ER below threshold, extreme or unknown volatility).
      * If surviving directional candidates disagree on side, nobody trades.
      * Otherwise the highest-confidence candidate wins; agreement from other strategies
        adds +0.1 confidence per extra vote (capped at 1.0).
      * A carry candidate is considered only if no directional trade is chosen, and only
        when the regime allows carry.
    """

    def __init__(self, strategies: Sequence[Strategy], regime_filter: RegimeFilter | None = None,
                 carry: FundingCarryStrategy | None = None) -> None:
        self.strategies = list(strategies)
        self.regime_filter = regime_filter or RegimeFilter()
        self.carry = carry

    @property
    def warmup(self) -> int:
        """Bars needed for every component to be live."""
        return max([self.regime_filter.warmup, *(s.warmup for s in self.strategies)], default=1)

    def aggregate(self, symbol: str, bars: Sequence[Bar],
                  funding_rates: Sequence[float] | None = None) -> AggregatedDecision:
        """Evaluate all strategies on ``bars`` and return the gated decision."""
        regime = self.regime_filter.evaluate(bars)
        rejected: list[tuple[str, str]] = []
        candidates: list[Signal] = []
        for strat in self.strategies:
            sig = strat.evaluate(symbol, bars)
            if sig is None:
                continue
            if not regime.allow_directional:
                rejected.append((strat.name, "regime gate: " + "; ".join(regime.reasons)))
                continue
            candidates.append(sig)

        chosen: Signal | None = None
        if candidates:
            sides = {c.side for c in candidates}
            if len(sides) > 1:
                rejected.extend((c.strategy, "conflicting directional votes") for c in candidates)
            else:
                best = max(candidates, key=lambda c: c.confidence)
                boost = 0.1 * (len(candidates) - 1)
                chosen = Signal(best.strategy, best.symbol, best.side, best.kind, best.entry, best.stop,
                                best.take_profit, min(1.0, best.confidence + boost), best.reason, best.ts)

        if chosen is None and self.carry is not None and funding_rates:
            carry_sig = self.carry.evaluate_funding(symbol, funding_rates, bars[-1].close, bars[-1].ts)
            if carry_sig is not None:
                if regime.allow_carry:
                    chosen = carry_sig
                    candidates.append(carry_sig)
                else:
                    rejected.append((self.carry.name, "regime gate: " + "; ".join(regime.reasons)))
        return AggregatedDecision(chosen, regime, tuple(candidates), tuple(rejected))
