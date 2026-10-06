"""FTMO-style challenge Monte Carlo on real backtest trades, against a zero-edge null.

Usage:
    python scripts/prop_firm_mc.py --csv data/XAUUSD_H1.csv

For each strategy: backtest year-by-year through the real risk stack (a breach in one
year does not truncate the trade sample for the rest), group closed trades by day,
then simulate challenge attempts at several risk-per-trade levels on both the real
trades and their demeaned (zero-edge) twin.

Decision gate (pre-registered here, before the first run):
    BUY a challenge only if, at some risk level,
      pass_rate - null_pass_rate >= 0.10   (edge, not variance, is doing the passing)
      AND ev_per_attempt > 0                (the fee is worth paying)
      AND the strategy's Lo-corrected t of R >= snooping_hurdle(trials)
                                            (the trades themselves are not noise)
    Otherwise HOLD the money.

``trials`` counts every distinct (strategy, params) ever run on this data, read from
an append-only ledger (``--ledger``, git-tracked under research/), so re-running variants until one clears the bar
raises the bar: 1-2 trials -> 2.00 (floor), 10 -> 2.81, 50 -> 3.29.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.critic import lo_corrected_t_stat, snooping_hurdle
from core.backtest import BacktestConfig, Trade, run_backtest
from core.market_data import load_csv
from core.prop_sim import FirmRules, TraderPolicy, daily_r_series, demeaned, run_monte_carlo
from core.risk_engine import RiskLimits
from core.strategy_brain import build_strategy

STRATEGIES = {
    "trend_breakout": {},
    "trend_breakout_live": {"channel": 20, "stop_atr": 3.0, "target_r": 2.0},  # what the demo loop runs
    "gold_pullback": {},
    "ker_trend_vol": {},
}
RISK_LEVELS = (0.0025, 0.005, 0.01, 0.02)
WARMUP_BARS = 400


def yearly_trades(bars, strategy, cfg) -> tuple[list[Trade], int]:
    trades: list[Trade] = []
    breaches = 0
    years = sorted({b.ts.year for b in bars})
    for y in years:
        idx = [i for i, b in enumerate(bars) if b.ts.year == y]
        lo = max(0, idx[0] - WARMUP_BARS)
        res = run_backtest(bars[lo:idx[-1] + 1], strategy, cfg, "XAUUSD", start_index=idx[0] - lo)
        trades.extend(res.trades)
        breaches += res.breached
    return trades, breaches


def count_trials(ledger: Path, candidates: dict[str, dict]) -> int:
    """Record this run's candidates and return the number of distinct ones ever tested."""
    seen: set[str] = set()
    if ledger.exists():
        seen = {line.strip() for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()}
    new = [json.dumps({"strategy": k, "params": v}, sort_keys=True) for k, v in candidates.items()]
    with ledger.open("a", encoding="utf-8") as fh:
        for key in new:
            if key not in seen:
                fh.write(key + "\n")
    return len(seen | set(new))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--paths", type=int, default=5_000)
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--out", default="state/prop_mc_report.json")
    ap.add_argument("--ledger", default="research/prop_mc_trials.jsonl",
                    help="append-only record of every (strategy, params) tested; drives the hurdle")
    args = ap.parse_args()

    bars = [b for b in load_csv(args.csv) if b.ts.year >= args.from_year - 1]
    trading_days = sorted({b.ts.date() for b in bars if b.ts.year >= args.from_year})
    bars_eval_start = next(i for i, b in enumerate(bars) if b.ts.year >= args.from_year)
    bars = bars[max(0, bars_eval_start - WARMUP_BARS):]
    print(f"{len(bars)} bars, {len(trading_days)} trading days, {trading_days[0]} .. {trading_days[-1]}")

    cfg = BacktestConfig(initial_equity=10_000.0,
                         limits=RiskLimits(day_rollover_tz="Europe/Prague"))
    rules = FirmRules()
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    trials = count_trials(ledger, STRATEGIES)
    hurdle = snooping_hurdle(trials)
    print(f"trials so far: {trials} -> t hurdle {hurdle:.2f}")
    report: dict = {"rules": {"name": rules.name, "fee": rules.fee, "account": rules.account_size},
              "trials": trials, "t_hurdle": round(hurdle, 3),
                    "strategies": {}}
    for name, params in STRATEGIES.items():
        strat = build_strategy(name.removesuffix("_live"), params)
        trades, breaches = yearly_trades(bars, strat, cfg)
        trades = [t for t in trades if t.exit_ts.year >= args.from_year]
        rs = [t.r_multiple for t in trades]
        days = daily_r_series(trades, trading_days)
        null = demeaned(days)
        row = {"params": params, "n_trades": len(rs), "mean_r": round(statistics.fmean(rs), 4) if rs else 0.0,
               "t_stat": round(lo_corrected_t_stat(rs).t_stat, 2), "engine_breach_years": breaches, "by_risk": []}
        print(f"\n== {name} {params}: {len(rs)} trades, mean R {row['mean_r']:+.3f}, Lo t={row['t_stat']:+.2f} (hurdle {hurdle:.2f})")
        print(f"{'risk':>6} {'pass':>6} {'null':>6} {'edge':>6} {'p1':>6} {'DLL':>6} {'maxL':>6} "
              f"{'starv':>6} {'t/o':>6} {'days':>5} {'EV$':>8} {'nullEV$':>8}")
        for risk in RISK_LEVELS:
            pol = TraderPolicy(risk_pct=risk)
            s = run_monte_carlo(days, rules, pol, args.paths, seed=11)
            n = run_monte_carlo(null, rules, pol, args.paths, seed=11)
            edge = s.pass_rate - n.pass_rate
            print(f"{risk:6.2%} {s.pass_rate:6.1%} {n.pass_rate:6.1%} {edge:+6.1%} {s.phase1_pass_rate:6.1%} "
                  f"{s.fail_daily_loss:6.1%} {s.fail_max_loss:6.1%} {s.fail_starved:6.1%} {s.fail_timeout:6.1%} "
                  f"{(s.median_days_to_pass or 0):5.0f} {s.ev_per_attempt:8.1f} {n.ev_per_attempt:8.1f}")
            row["by_risk"].append({"real": s.to_dict(), "null": n.to_dict(), "edge": round(edge, 4)})
        gate = any(r["edge"] >= 0.10 and r["real"]["ev_per_attempt"] > 0 for r in row["by_risk"]) \
            and row["t_stat"] >= hurdle
        row["verdict"] = "BUY" if gate else "HOLD"
        print(f"verdict: {row['verdict']}")
        report["strategies"][name] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
