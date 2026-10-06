"""Generator-Critic statistics and the MCP service's safety rules."""
from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from agent.critic import CriticConfig, QCritic, lo_corrected_t_stat, snooping_hurdle
from agent.generator import HypothesisGenerator, StrategyCandidate
from core.backtest import BacktestConfig
from core.market_data import synthetic_ohlcv
from mcp_server.mcp_bridge import LIVE_ENV_FLAG, OrchestratorService, handle_jsonrpc


# ------------------------------------------------------------------ statistics
class TestLoStatistic:
    def test_iid_matches_lo_standard_error(self) -> None:
        rng = random.Random(0)
        xs = [rng.gauss(0.01, 0.02) for _ in range(400)]
        lo = lo_corrected_t_stat(xs, max_lag=0)
        assert lo.inflation == pytest.approx(1.0)
        assert lo.t_stat == pytest.approx(lo.sharpe / ((1 + 0.5 * lo.sharpe ** 2) / 400) ** 0.5)

    def test_positive_autocorrelation_deflates_t(self) -> None:
        rng = random.Random(1)
        e, xs = 0.0, []
        for _ in range(400):
            e = 0.7 * e + rng.gauss(0, 0.02)
            xs.append(0.004 + e)
        naive = lo_corrected_t_stat(xs, max_lag=0)
        corrected = lo_corrected_t_stat(xs)
        assert corrected.inflation > 1.5
        assert corrected.t_stat < naive.t_stat

    def test_negative_autocorrelation_never_flatters(self) -> None:
        xs = [0.02 if i % 2 else -0.01 for i in range(200)]
        assert lo_corrected_t_stat(xs).t_stat <= lo_corrected_t_stat(xs, max_lag=0).t_stat + 1e-12

    def test_degenerate_inputs(self) -> None:
        assert lo_corrected_t_stat([0.01, 0.02]).t_stat == 0.0
        assert lo_corrected_t_stat([0.01] * 50).t_stat == 0.0

    def test_snooping_hurdle_grows_with_trials(self) -> None:
        assert snooping_hurdle(1) == pytest.approx(2.0)
        assert snooping_hurdle(10) > snooping_hurdle(2) > 2.0
        assert snooping_hurdle(28) == pytest.approx(3.12, abs=0.01)


# ------------------------------------------------------------------ generator / critic
class TestGeneratorCritic:
    def test_generator_counts_every_trial_and_is_reproducible(self) -> None:
        a, b = HypothesisGenerator(seed=3), HypothesisGenerator(seed=3)
        ca, cb = a.generate(5), b.generate(5)
        assert [c.candidate_id for c in ca] == [c.candidate_id for c in cb]
        more = a.generate()
        assert a.trials == a.grid_size() == len(ca) + len(more)
        assert len({c.candidate_id for c in ca + more}) == a.trials
        assert a.generate() == []

    def test_trade_frequency_floor_cannot_be_lowered(self) -> None:
        with pytest.raises(ValueError):
            CriticConfig(min_oos_trades=10)

    def test_sparse_strategy_is_rejected_for_frequency(self) -> None:
        bars = synthetic_ohlcv(600, seed=2)
        cand = StrategyCandidate("x", "trend_breakout", {"channel": 55, "stop_atr": 3.0}, "h", 1)
        verdict = QCritic().review(cand, bars, BacktestConfig(), trials=1)
        assert not verdict.promoted
        assert any(r.startswith("trade frequency") for r in verdict.reasons)

    def test_more_trials_raise_the_bar(self) -> None:
        bars = synthetic_ohlcv(1_500, seed=7)
        cand = StrategyCandidate("x", "trend_breakout", {"channel": 20, "stop_atr": 2.0, "target_r": 3.0}, "h", 1)
        few = QCritic().review(cand, bars, BacktestConfig(), trials=1)
        many = QCritic().review(cand, bars, BacktestConfig(), trials=10_000)
        assert many.hurdle > few.hurdle
        assert few.lo.t_stat == pytest.approx(many.lo.t_stat)
        assert not many.promoted


# ------------------------------------------------------------------ MCP service
@pytest.fixture()
def service(tmp_path) -> OrchestratorService:
    # Fixed mid-week clock: weekend/news compliance must not make tests depend on today's date.
    svc = OrchestratorService(tmp_path, clock=lambda: datetime(2026, 9, 23, 12, 0, tzinfo=UTC))
    svc.fetch_ohlcv("XAUUSD", limit=300)
    return svc


class TestMcpService:
    def test_paper_order_is_sized_by_risk_engine(self, service: OrchestratorService) -> None:
        px = service._bars["XAUUSD"][-1].close
        res = service.execute_order("long", stop_loss=px * 0.98, take_profit=px * 1.04)
        assert res["accepted"]
        assert res["risk_amount"] <= 0.005 * 100_000 + 1e-6
        metrics = service.get_risk_metrics()
        assert len(metrics["positions"]) == 1 and metrics["open_risk"] > 0

    def test_live_refused_without_operator_flag(self, service: OrchestratorService, monkeypatch) -> None:
        monkeypatch.delenv(LIVE_ENV_FLAG, raising=False)
        px = service._bars["XAUUSD"][-1].close
        res = service.execute_order("long", px * 0.98, px * 1.04, mode="live", confirm=True)
        assert not res["accepted"] and LIVE_ENV_FLAG in res["reason"]

    def test_live_refused_without_confirm_or_broker(self, service: OrchestratorService, monkeypatch) -> None:
        monkeypatch.setenv(LIVE_ENV_FLAG, "ENABLED")
        px = service._bars["XAUUSD"][-1].close
        assert "confirm" in service.execute_order("long", px * 0.98, px * 1.04, mode="live")["reason"]
        res = service.execute_order("long", px * 0.98, px * 1.04, mode="live", confirm=True)
        assert "no live broker" in res["reason"]

    def test_kill_switch_blocks_and_flattens(self, service: OrchestratorService) -> None:
        px = service._bars["XAUUSD"][-1].close
        service.execute_order("long", px * 0.98, px * 1.04)
        report = service.trip_kill_switch("test")
        assert report["flattened"] == 1
        res = service.execute_order("long", px * 0.98, px * 1.04)
        assert not res["accepted"] and "kill switch" in res["reason"]

    def test_bad_stop_side_rejected(self, service: OrchestratorService) -> None:
        px = service._bars["XAUUSD"][-1].close
        assert not service.execute_order("long", stop_loss=px * 1.01, take_profit=px * 1.05)["accepted"]

    def test_jsonrpc_dispatch(self, service: OrchestratorService) -> None:
        listed = handle_jsonrpc(service, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {t["name"] for t in listed["result"]}
        assert {"fetch_ohlcv", "backtest_strategy", "get_risk_metrics", "execute_order"} <= names
        assert handle_jsonrpc(service, {"id": 2, "method": "nope"})["error"]["code"] == -32601
        bt = handle_jsonrpc(service, {"id": 3, "method": "backtest_strategy",
                                      "params": {"bars": 600, "critic": False}})
        assert "metrics" in bt["result"]
