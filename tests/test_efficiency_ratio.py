"""Kaufman Efficiency Ratio correctness and whipsaw / chop suppression."""
from __future__ import annotations

import math
import random

import pytest

from core.backtest import BacktestConfig, run_backtest
from core.market_data import synthetic_ohlcv
from core.regime_filter import (
    DEFAULT_ER_THRESHOLD,
    RegimeFilter,
    RegimeFilterConfig,
    efficiency_ratio,
    rolling_efficiency_ratio,
)
from core.strategy_brain import SignalAggregator, TrendBreakoutStrategy
from tests.conftest import bars_from_closes


# ------------------------------------------------------------------ formula
class TestEfficiencyRatioFormula:
    def test_straight_line_is_one(self) -> None:
        assert efficiency_ratio([float(i) for i in range(50)], 10) == pytest.approx(1.0)

    def test_perfect_zigzag_is_zero(self) -> None:
        zigzag = [100.0 + (i % 2) for i in range(41)]
        assert efficiency_ratio(zigzag, 10) == pytest.approx(0.0)

    def test_hand_computed_value(self) -> None:
        # |12 - 10| / (1 + 0.5 + 1.5) = 2 / 3
        assert efficiency_ratio([10.0, 11.0, 10.5, 12.0], 3) == pytest.approx(2 / 3)

    def test_uses_only_the_last_n_changes(self) -> None:
        series = [1.0, 50.0, 10.0, 11.0, 10.5, 12.0]
        assert efficiency_ratio(series, 3) == pytest.approx(2 / 3)

    def test_flat_series_is_zero_not_nan(self) -> None:
        assert efficiency_ratio([5.0] * 30, 10) == 0.0

    def test_direction_symmetric(self) -> None:
        up = [10.0, 11.0, 10.5, 12.0]
        down = [20.0 - x for x in up]
        assert efficiency_ratio(up, 3) == pytest.approx(efficiency_ratio(down, 3))

    def test_scale_and_translation_invariant(self) -> None:
        rng = random.Random(3)
        xs = [100 + rng.gauss(0, 1) for _ in range(40)]
        base = efficiency_ratio(xs, 20)
        assert efficiency_ratio([x * 7.5 for x in xs], 20) == pytest.approx(base)
        assert efficiency_ratio([x + 1_000 for x in xs], 20) == pytest.approx(base)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError):
            efficiency_ratio([1.0, 2.0], 10)
        with pytest.raises(ValueError):
            efficiency_ratio([1.0, 2.0, 3.0], 0)

    def test_bounded_between_zero_and_one(self) -> None:
        rng = random.Random(11)
        for _ in range(500):
            xs = [rng.uniform(1, 100) for _ in range(rng.randint(3, 60))]
            n = rng.randint(1, len(xs) - 1)
            assert 0.0 <= efficiency_ratio(xs, n) <= 1.0

    def test_rolling_matches_pointwise(self) -> None:
        rng = random.Random(5)
        xs = [100.0]
        for _ in range(300):
            xs.append(xs[-1] * math.exp(rng.gauss(0, 0.01)))
        rolled = rolling_efficiency_ratio(xs, 20)
        assert all(math.isnan(v) for v in rolled[:20])
        for t in range(20, len(xs)):
            assert rolled[t] == pytest.approx(efficiency_ratio(xs[:t + 1], 20), abs=1e-9)


# ------------------------------------------------------------------ regime gate
def _chop_then_breakout(n: int = 200) -> list:
    """Wide zigzag between 100 and 110, then a final close above the channel high."""
    closes = [100.0 if i % 2 == 0 else 110.0 for i in range(n)]
    closes.append(111.0)
    return bars_from_closes(closes, wick=0.1)


class TestRegimeGate:
    def test_default_threshold_is_spec_value(self) -> None:
        assert DEFAULT_ER_THRESHOLD == pytest.approx(0.30)
        assert RegimeFilterConfig().er_threshold == pytest.approx(0.30)

    def test_trend_allows_entries(self) -> None:
        rng = random.Random(1)
        closes = [100.0]
        for _ in range(250):
            closes.append(closes[-1] * (1.004 + rng.gauss(0, 0.002)))
        snap = RegimeFilter().evaluate(bars_from_closes(closes, wick=0.2))
        assert snap.efficiency_ratio >= 0.30
        assert snap.allow_directional

    def test_chop_blocks_entries(self) -> None:
        snap = RegimeFilter().evaluate(_chop_then_breakout())
        assert snap.efficiency_ratio < 0.30
        assert not snap.allow_directional
        assert any("chop" in r for r in snap.reasons)

    def test_threshold_boundary(self) -> None:
        bars = _chop_then_breakout()
        er = RegimeFilter().evaluate(bars).efficiency_ratio
        at = RegimeFilter(RegimeFilterConfig(er_threshold=er)).evaluate(bars)
        above = RegimeFilter(RegimeFilterConfig(er_threshold=er + 1e-9)).evaluate(bars)
        assert at.allow_directional  # ER == threshold passes; only ER < threshold blocks
        assert not above.allow_directional

    def test_fails_closed_without_history(self) -> None:
        snap = RegimeFilter().evaluate(bars_from_closes([float(i) for i in range(1, 15)]))
        assert not snap.allow_directional and not snap.allow_carry

    def test_breakout_in_chop_is_vetoed_by_aggregator(self) -> None:
        bars = _chop_then_breakout()
        strat = TrendBreakoutStrategy(channel=20)
        raw = strat.evaluate("X", bars)
        assert raw is not None, "the strategy alone does see a breakout"
        gated = SignalAggregator([strat]).aggregate("X", bars)
        assert gated.signal is None
        assert any(r.startswith("regime gate") for _, r in gated.rejected)
        ungated = SignalAggregator([strat], RegimeFilter(RegimeFilterConfig(er_threshold=0.0))).aggregate("X", bars)
        assert ungated.signal is not None


# ------------------------------------------------------------------ whipsaw suppression in backtests
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_ker_gate_suppresses_whipsaw_losses_in_chop(seed: int) -> None:
    chop = synthetic_ohlcv(800, seed=seed, schedule=["chop"])
    gated = run_backtest(chop, TrendBreakoutStrategy(), BacktestConfig(use_regime_filter=True))
    ungated = run_backtest(chop, TrendBreakoutStrategy(), BacktestConfig(use_regime_filter=False))
    assert gated.metrics.n_trades < ungated.metrics.n_trades / 2
    assert gated.metrics.total_return_pct > ungated.metrics.total_return_pct
    assert gated.signals_blocked_by_regime > 0
    assert not gated.breached


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ker_gate_keeps_trading_in_trends(seed: int) -> None:
    trend = synthetic_ohlcv(800, seed=seed, schedule=["trend"])
    res = run_backtest(trend, TrendBreakoutStrategy(), BacktestConfig(use_regime_filter=True))
    assert res.metrics.n_trades >= 15
    assert res.metrics.total_return_pct > 0


def test_backtest_never_breaches_on_mixed_regimes() -> None:
    for seed in range(1, 6):
        res = run_backtest(synthetic_ohlcv(1_500, seed=seed), TrendBreakoutStrategy(), BacktestConfig())
        assert not res.breached
        assert res.metrics.max_drawdown_pct < 6.0
