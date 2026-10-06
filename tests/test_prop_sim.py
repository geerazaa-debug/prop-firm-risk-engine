"""Prop-firm Monte Carlo: rule mechanics, sizing policy, and the zero-edge null."""
from __future__ import annotations

import random
from datetime import UTC, date, datetime

import pytest

from core.backtest import Trade
from core.prop_sim import (
    FirmRules,
    PhaseRules,
    TraderPolicy,
    _trade_risk,
    block_bootstrap,
    daily_r_series,
    demeaned,
    run_monte_carlo,
)
from core.risk_engine import Side

ONE_PHASE = FirmRules(phases=(PhaseRules(0.10, min_trading_days=1),), funded_days=0)


def _trade(day: int, r: float) -> Trade:
    ts = datetime(2024, 1, day, 12, tzinfo=UTC)
    return Trade("s", Side.LONG, ts, ts, 1.0, 1.0, 1.0, r, r, r / 100, "x")


class TestInputs:
    def test_empty_days_are_kept(self):
        days = [date(2024, 1, d) for d in (1, 2, 3)]
        out = daily_r_series([_trade(1, 1.0), _trade(1, -1.0), _trade(3, 2.0)], days)
        assert out == [(1.0, -1.0), (), (2.0,)]

    def test_demeaned_has_zero_mean_and_same_shape(self):
        days = [(1.0, 2.0), (), (3.0,)]
        null = demeaned(days)
        assert [len(d) for d in null] == [2, 0, 1]
        assert sum(r for d in null for r in d) == pytest.approx(0.0)

    def test_bootstrap_draws_only_existing_days(self):
        days = [(float(i),) for i in range(20)]
        sample = block_bootstrap(days, 500, random.Random(1), mean_block=5)
        assert len(sample) == 500 and set(sample) <= set(days)


class TestSizingPolicy:
    def test_sits_out_when_one_stop_could_breach_daily_limit(self):
        rules, pol = FirmRules(), TraderPolicy(risk_pct=0.01)
        assert _trade_risk(0.955, 1.0, rules, pol) == 0.0  # 0.955 - 0.00955 <= 0.95
        assert _trade_risk(0.97, 1.0, rules, pol) > 0.0

    def test_downsizes_near_max_loss_line(self):
        rules, pol = FirmRules(), TraderPolicy(risk_pct=0.01)
        risk = _trade_risk(0.905, 0.905, rules, pol)
        assert risk == pytest.approx(0.8 * 0.005)
        assert 0.905 - risk > 0.90


class TestRules:
    def test_certain_winner_always_passes(self):
        days = [(1.0,)] * 50
        s = run_monte_carlo(days, ONE_PHASE, TraderPolicy(0.01), n_paths=200)
        assert s.pass_rate == 1.0

    def test_certain_loser_starves_without_breaching(self):
        days = [(-1.0,)] * 50
        s = run_monte_carlo(days, ONE_PHASE, TraderPolicy(0.01), n_paths=200)
        assert s.pass_rate == 0.0
        # stops are bounded at -1R, so the policy shrinks size until it cannot trade at all
        assert s.fail_max_loss == 0.0 and s.fail_daily_loss == 0.0 and s.fail_starved == 1.0

    def test_gapped_stop_can_breach_daily_limit(self):
        days = [(-6.0,)]  # a 6R gap at 1% risk = -6%
        s = run_monte_carlo(days, ONE_PHASE, TraderPolicy(0.01), n_paths=50)
        assert s.fail_daily_loss == 1.0

    def test_min_trading_days_delays_pass(self):
        rules = FirmRules(phases=(PhaseRules(0.01, min_trading_days=4),), funded_days=0)
        s = run_monte_carlo([(5.0,)], rules, TraderPolicy(0.01), n_paths=20)
        assert s.median_days_to_pass == 4

    def test_ev_is_fee_negative_without_passes(self):
        s = run_monte_carlo([(-1.0,)] * 10, ONE_PHASE, TraderPolicy(0.01), n_paths=20)
        assert s.ev_per_attempt == pytest.approx(-ONE_PHASE.fee)

    def test_payouts_and_fee_refund(self):
        rules = FirmRules(phases=(PhaseRules(0.01, min_trading_days=1),), funded_days=21, payout_every_days=21)
        s = run_monte_carlo([(1.0,)], rules, TraderPolicy(0.01), n_paths=10)
        # 21 days of +1% compounding on the funded account, 80% split, plus fee refund
        expected = ((1.01 ** 21) - 1) * 0.8 * rules.account_size + rules.fee
        assert s.mean_funded_payout == pytest.approx(expected, rel=1e-6)

    def test_zero_edge_null_still_passes_often(self):
        """Symmetric barriers: a pure coin flip passes a lot. Pass rate alone is not evidence."""
        rng = random.Random(3)
        days = [tuple(rng.choice((2.0, -1.0, -1.0)) for _ in range(rng.randint(0, 2))) for _ in range(500)]
        null = demeaned(days)
        s = run_monte_carlo(null, FirmRules(phases=(PhaseRules(0.10),), funded_days=0),
                            TraderPolicy(0.01), n_paths=2_000)
        assert 0.15 < s.pass_rate < 0.6
