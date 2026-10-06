"""Event-driven, bar-by-bar backtester wired through the real risk engine.

The backtester deliberately reuses the production components — ``SignalAggregator``
(regime gate included), ``size_position`` and ``mark_to_market`` from the risk engine,
and ``ratchet_stop`` — so a backtest exercises the same guardrails as live trading.

Execution assumptions (conservative by design):
  * Signals are computed on bar ``t``'s close and filled at bar ``t+1``'s open.
  * Slippage is adverse on every fill; commission is charged per side on notional.
  * Stops/targets that gap are filled at the open, not at the level.
  * If a bar touches both stop and target, the stop is assumed to fill first.
  * The ratchet stop is updated *after* the exit check, so a new level only applies
    from the following bar.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from core.indicators import atr_series
from core.market_data import Bar
from core.regime_filter import RegimeFilter, RegimeFilterConfig
from core.risk_engine import (
    AccountState,
    AccountStatus,
    AssetSpec,
    ExposureSnapshot,
    RatchetConfig,
    RatchetStop,
    RiskAction,
    RiskLimits,
    Side,
    mark_to_market,
    ratchet_stop,
    size_position,
)
from core.strategy_brain import Signal, SignalAggregator, Strategy

PF_CAP = 999.0


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest settings. ``asset`` defaults to an unlevered spot-like spec."""

    initial_equity: float = 100_000.0
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    limits: RiskLimits = field(default_factory=RiskLimits)
    asset: AssetSpec | None = None
    ratchet: RatchetConfig | None = field(default_factory=RatchetConfig)
    use_regime_filter: bool = True
    regime: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)
    periods_per_year: int = 252


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed round-trip."""

    strategy: str
    side: Side
    entry_ts: datetime
    exit_ts: datetime
    entry: float
    exit: float
    units: float
    pnl: float
    r_multiple: float
    return_on_equity: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary statistics of a backtest."""

    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_annualised: float
    trade_returns: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, float | int]:
        """JSON-friendly dict (omits the per-trade return vector)."""
        return {"n_trades": self.n_trades, "win_rate": round(self.win_rate, 4),
                "profit_factor": round(self.profit_factor, 3), "expectancy_r": round(self.expectancy_r, 3),
                "total_return_pct": round(self.total_return_pct, 3),
                "max_drawdown_pct": round(self.max_drawdown_pct, 3),
                "sharpe_annualised": round(self.sharpe_annualised, 3)}


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Full backtest output."""

    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]
    metrics: PerformanceMetrics
    risk_events: tuple[str, ...]
    breached: bool
    signals_blocked_by_regime: int


@dataclass(slots=True)
class _OpenPosition:
    signal: Signal
    side: Side
    entry: float
    units: float
    take_profit: float
    ratchet: RatchetStop
    entry_ts: datetime
    equity_at_entry: float
    risk_amount: float


def _max_drawdown_pct(curve: Sequence[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd * 100.0


def compute_metrics(trades: Sequence[Trade], curve: Sequence[float], periods_per_year: int) -> PerformanceMetrics:
    """Derive :class:`PerformanceMetrics` from trades and a bar-level equity curve."""
    n = len(trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl <= 0]
    gross_loss = sum(losses)
    pf = (sum(wins) / gross_loss) if gross_loss > 0 else (PF_CAP if wins else 0.0)
    bar_rets = [(curve[i] / curve[i - 1] - 1.0) for i in range(1, len(curve)) if curve[i - 1] > 0]
    sharpe = 0.0
    if len(bar_rets) > 2:
        sd = statistics.pstdev(bar_rets)
        if sd > 0:
            sharpe = statistics.fmean(bar_rets) / sd * math.sqrt(periods_per_year)
    return PerformanceMetrics(
        n_trades=n,
        win_rate=(len(wins) / n) if n else 0.0,
        profit_factor=min(pf, PF_CAP),
        expectancy_r=statistics.fmean(t.r_multiple for t in trades) if n else 0.0,
        total_return_pct=((curve[-1] / curve[0] - 1.0) * 100.0) if len(curve) > 1 else 0.0,
        max_drawdown_pct=_max_drawdown_pct(curve),
        sharpe_annualised=sharpe,
        trade_returns=tuple(t.return_on_equity for t in trades),
    )


def run_backtest(
    bars: Sequence[Bar],
    strategies: Strategy | Sequence[Strategy],
    config: BacktestConfig | None = None,
    symbol: str = "SYM",
    start_index: int = 0,
) -> BacktestResult:
    """Simulate ``strategies`` over ``bars`` through the full risk stack.

    Args:
        bars: Price history (oldest first).
        strategies: One strategy or several (aggregated by :class:`SignalAggregator`).
        config: Backtest settings.
        symbol: Symbol label.
        start_index: First bar on which new entries may be *signalled*. Earlier bars
            are used only for indicator warm-up (walk-forward OOS runs set this to the
            split point).
    """
    config = config or BacktestConfig()
    strat_list: list[Strategy] = [strategies] if hasattr(strategies, "evaluate") else list(strategies)  # type: ignore[arg-type]
    regime_cfg = config.regime if config.use_regime_filter else RegimeFilterConfig(
        er_period=config.regime.er_period, er_threshold=0.0, block_extreme_vol=False)
    aggregator = SignalAggregator(strat_list, RegimeFilter(regime_cfg))
    # Sizing must price in exactly the costs this simulation charges.
    asset = replace(config.asset or AssetSpec(symbol=symbol),
                    commission_bps=config.commission_bps, slippage_bps=config.slippage_bps)
    limits = config.limits
    window = aggregator.warmup + 20
    atr = atr_series(bars, regime_cfg.atr_period)
    slip = config.slippage_bps / 10_000
    comm = config.commission_bps / 10_000
    unit_value = asset.unit_value

    if not bars:
        empty = compute_metrics([], [config.initial_equity], config.periods_per_year)
        return BacktestResult((), (config.initial_equity,), empty, (), False, 0)

    cash = config.initial_equity
    state = AccountState.open(cash, bars[0].ts, limits)
    pos: _OpenPosition | None = None
    pending: Signal | None = None
    trades: list[Trade] = []
    curve: list[float] = [cash]
    events: list[str] = []
    blocked = 0
    breached = False

    def close(p: _OpenPosition, raw_price: float, ts: datetime, reason: str) -> None:
        nonlocal cash
        fill = raw_price * (1 - p.side.sign * slip)
        pnl = (fill - p.entry) * p.side.sign * p.units * unit_value - fill * p.units * unit_value * comm
        cash += pnl
        r = pnl / p.risk_amount if p.risk_amount > 0 else 0.0
        trades.append(Trade(p.signal.strategy, p.side, p.entry_ts, ts, p.entry, fill, p.units, pnl, r,
                            pnl / p.equity_at_entry, reason))

    for i, bar in enumerate(bars):
        # 1) fill the pending entry at this bar's open
        if pending is not None and pos is None and state.trading_allowed:
            sig = pending
            fill = bar.open * (1 + sig.side.sign * slip)
            stop_dist = abs(sig.entry - sig.stop)
            stop = fill - sig.side.sign * stop_dist
            tp = fill + sig.side.sign * abs(sig.take_profit - sig.entry)
            if stop > 0 and tp > 0:
                size = size_position(state, limits, asset, fill, stop, ExposureSnapshot())
                if size.approved:
                    cash -= fill * size.units * unit_value * comm
                    pos = _OpenPosition(sig, sig.side, fill, size.units, tp, RatchetStop.open(sig.side, fill, stop),
                                        bar.ts, state.equity, size.risk_amount)
        pending = None

        # 2) exits on this bar's range
        if pos is not None:
            stop_px = pos.ratchet.stop
            if pos.side is Side.LONG:
                stop_hit, tp_hit = bar.low <= stop_px, bar.high >= pos.take_profit
                stop_fill = min(bar.open, stop_px)
                tp_fill = max(bar.open, pos.take_profit)
            else:
                stop_hit, tp_hit = bar.high >= stop_px, bar.low <= pos.take_profit
                stop_fill = max(bar.open, stop_px)
                tp_fill = min(bar.open, pos.take_profit)
            if stop_hit:
                close(pos, stop_fill, bar.ts, "trailing_stop" if pos.ratchet.breakeven_locked else "stop_loss")
                pos = None
            elif tp_hit:
                close(pos, tp_fill, bar.ts, "take_profit")
                pos = None

        # 3) mark to market and run the hard risk engine
        unreal = 0.0 if pos is None else (bar.close - pos.entry) * pos.side.sign * pos.units * unit_value
        evaluation = mark_to_market(state, cash + unreal, bar.ts, limits)
        state = evaluation.state
        if evaluation.action is not RiskAction.NONE:
            events.append(f"{bar.ts.date()} {evaluation.action.value}: {evaluation.message}")
            if pos is not None:
                close(pos, bar.close, bar.ts, evaluation.action.value)
                pos = None
            state = mark_to_market(state, cash, bar.ts, limits).state
            if state.status is AccountStatus.BREACHED:
                breached = True
                curve.append(cash)
                break
        curve.append(cash + (0.0 if pos is None else unreal))

        # 4) ratchet the stop for the next bar
        if pos is not None and config.ratchet is not None:
            pos.ratchet = ratchet_stop(pos.ratchet, bar.high, bar.low, atr[i], config.ratchet)

        # 5) look for a new entry (signal on close, fill next open)
        if pos is None and i >= start_index and i + 1 < len(bars) and i + 1 >= aggregator.warmup \
                and state.trading_allowed:
            decision = aggregator.aggregate(symbol, bars[max(0, i + 1 - window):i + 1])
            blocked += sum(1 for _, r in decision.rejected if r.startswith("regime gate"))
            pending = decision.signal

    if pos is not None:
        last = bars[-1]
        close(pos, last.close, last.ts, "end_of_data")
        curve[-1] = cash

    # Walk-forward runs report statistics for the evaluation window only.
    if start_index > 0:
        first_ts = bars[min(start_index, len(bars) - 1)].ts
        trades = [t for t in trades if t.entry_ts >= first_ts]
        curve = curve[min(start_index, len(curve) - 1):]
    metrics = compute_metrics(trades, curve, config.periods_per_year)
    return BacktestResult(tuple(trades), tuple(curve), metrics, tuple(events), breached, blocked)
