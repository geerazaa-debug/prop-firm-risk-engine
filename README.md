# Prop-firm risk engine and challenge Monte Carlo

**Short version:** this repository is a prop-firm trading engine with hard risk limits.
It also contains a Monte Carlo test of whether paying for an FTMO challenge with these
strategies is worth the fee. **It is not.** Every strategy passed the challenge *less* often
than a zero-edge copy of itself, so no fee was paid.

## Result: is the $170 FTMO challenge worth buying?

[`core/prop_sim.py`](core/prop_sim.py) and [`scripts/prop_firm_mc.py`](scripts/prop_firm_mc.py)
take each strategy's real closed trades (XAUUSD H1, 2015-2026, backtested year by year
through the full risk engine). They resample trading days with a stationary block
bootstrap, so losing streaks stay intact, and run every path through FTMO 2-step rules:
targets, minimum days, 5% daily loss, 10% max loss, then a funded stage with payouts.
The same paths are also run on a **demeaned (zero-edge) twin** of the trades.

The decision gate was written into the script before the first run. Buy only if:
pass rate minus null pass rate ≥ 10 pp, **and** EV per attempt > 0, **and** Lo-corrected t
of trade R ≥ the snooping hurdle for the trials in the ledger (4 trials → 2.50).

| Strategy | Trades | Mean R | Lo t | Best risk level: pass / **null pass** | EV / **null EV** | Verdict |
|---|---:|---:|---:|---|---|---|
| trend_breakout | 1,421 | −0.044 | −1.54 | 23% / **33%** (2% risk) | −$54 / **−$7** | HOLD |
| trend_breakout_live | 1,029 | −0.011 | −0.33 | 35% / **41%** (2% risk) | **+$32** / **+$80** | HOLD |
| gold_pullback | 585 | −0.044 | −0.89 | 15% / **20%** (2% risk) | −$94 / **−$47** | HOLD |
| ker_trend_vol | 780 | −0.040 | −1.00 | 17% / **26%** (2% risk) | −$27 / **+$78** | HOLD |

*Monte Carlo used 100 paths per cell, so a pass rate has a standard error of about ±5 pp.
The conclusion does not depend on that precision: every edge is ≤ 0.*

**The trap this gate catches.** `trend_breakout_live` at 2% risk shows a 35% pass rate
and a *positive* EV of +$32 per attempt, which looks like "buy the challenge". But its
zero-edge twin passes 41% of the time with EV +$80. The challenge's +10% / −10% barriers
reward **variance**, not edge. A coin flip with enough risk per trade passes about a third
of the time. So a pass rate, or even a positive EV, says nothing on its own. The quantity
that matters is `pass_rate − null_pass_rate`.

---

## Engine overview

A terminal-first prop-firm orchestrator and multi-agent trading engine. Alpha signals
(trend breakout, gold trend-pullback, funding carry) can only reach a broker through a
fixed chain of vetoes:

```
kill switch ─► hard risk engine ─► ratchet stops ─► KER regime gate ─► leverage-aware sizing ─► atomic bracket
```

Research candidates reach that chain only after a **Generator → Critic** loop promotes them
using out-of-sample tests, a trade-frequency floor, Lo-corrected t-statistics and a
multiple-testing (snooping) hurdle.

Everything runs offline in **paper mode** by default. The core is pure standard-library
Python.

## Project structure

```
core/ ... (repository root)
├── core/
│   ├── risk_engine.py      DLL 2.5%, trailing DD 6%, leverage-aware sizing, immutable ratchet stop
│   ├── regime_filter.py    Kaufman Efficiency Ratio + ATR% volatility regime (fails closed)
│   ├── strategy_brain.py   TrendBreakout, GoldTrendPullback, FundingCarry, SignalAggregator
│   ├── backtest.py         bar-by-bar backtester wired through the real risk engine   (added)
│   ├── indicators.py       dependency-free ATR / EMA / Donchian                        (added)
│   ├── market_data.py      Bar model, regime-switching synthetic data, CSV, public ccxt (added)
│   ├── audit.py            hash-chained, append-only JSONL audit trail                 (added)
│   ├── compliance.py       Friday 19:30 UTC auto-flatten / weekend block, ComplianceGate
│   ├── news_filter.py      high-impact news blackout (±5 min), Forex-Factory/CSV calendars
│   └── instruments.py      default contract specs (XAUUSD: 100 oz, 0.01 price tick)
├── agent/
│   ├── generator.py        seeded hypothesis sampler; counts every trial
│   ├── critic.py           Q-Agent critic: OOS, N>=15, Lo t-stat, Bonferroni hurdle, decay
│   └── coordinator.py      ResearchCoordinator + per-bar TradingCoordinator (regime gatekeeper)
├── execution/
│   ├── base.py             BracketOrder, BrokerAdapter protocol, thread-safe PaperBroker (added)
│   ├── cctv_executor.py    CEX executor (ccxt): atomic-or-rollback brackets, trade-only keys
│   ├── mt5_bridge.py       MT5 wire protocol, ZeroMQ/native transports, heartbeat + reconnect
│   ├── mt5_adapter.py      MT5 BrokerAdapter: link gating, timeout reconciliation, 1-shot FLATTEN_ALL
│   ├── live_runner.py      live MT5 loop: closed-bar feed, per-step DEMO-only guard, supervision
│   └── kill_switch.py      thread-safe, persistent, cancel-then-flatten circuit breaker
├── mcp_server/mcp_bridge.py  fetch_ohlcv, backtest_strategy, get_risk_metrics, execute_order, trip_kill_switch
├── mql5/
│   ├── ZeroMQ_Listener.mq5 MT5 EA: atomic brackets, FLATTEN_ALL, weekend guard, ms audit log
│   ├── README_MT5.md       step-by-step MT5 / ZeroMQ setup guide
│   └── tools/patch_mql_zmq.py  makes mql-zmq compile on current MT5 builds
├── scripts/mt5_bridge_selftest.py  real-socket self-test against a mock EA (needs pyzmq)
├── app/main.py             dashboard, interactive shell, research/backtest/kill/rearm/audit CLI
├── tests/                  367 tests (risk, KER, execution, MT5 bridge, compliance, agents, MCP)
├── requirements.txt
└── pyproject.toml
```

Files marked *(added)* are not in the original spec. They hold shared logic that the
specified modules depend on.

## Quick start

```bash
pip install -r requirements.txt        # only pytest + tzdata are required
python -m pytest                       # 367 passed
python -m app.main research            # Generator -> Critic round on synthetic data
python -m app.main backtest --strategy gold_pullback
python -m app.main dashboard           # live paper-replay dashboard
python -m app.main shell               # step / run / risk / kill / rearm / audit
python -m app.main mt5-ping            # check the MT5 ZeroMQ EA (set QM_MT5_TOKEN first)
python -m app.main live --symbol XAUUSD --timeframe 1h   # DEMO-ONLY live loop (see below)
```

Use your own data with `--csv path.csv` (columns `ts,open,high,low,close[,volume]`).

## Hard risk guardrails (`core/risk_engine.py`)

| Rule | Engine | Typical firm | Effect |
|---|---|---|---|
| Daily loss limit | **2.5%** of day-start equity | 5% | `FREEZE_AND_FLATTEN`; unfreezes next trading day |
| Trailing drawdown | **6.0%** below high-water mark | 10% | `TERMINATE_AND_FLATTEN`; terminal; trips kill switch |
| Risk per trade | ≤ 0.5% (hard max 1%) | — | fixed-fractional |
| Open risk | ≤ 1.5% (must be < DLL) | — | all stops hit together still can't breach the DLL |

* The ceilings are module constants. `RiskLimits` can tighten them, and construction
  fails if it tries to loosen them.
* **Headroom-aware sizing.** A new trade's worst-case loss is capped at 80% of the
  remaining DLL and drawdown headroom after existing open risk. A randomised property
  test (3,000 cases) checks that no stop-out can cross either floor.
* **Costs count as risk.** Risk per unit is the stop distance plus estimated round-trip
  commission and slippage (`AssetSpec.commission_bps` / `slippage_bps`, default 2 + 1
  bps per side). `risk_amount` is therefore the full loss if the stop is hit. The
  backtester overrides the asset's costs with its own, so sizing and simulation agree.
* **Reserved budget floor.** A cushion of `min_budget_floor_pct` of equity (default
  0.5%, one standard trade; hard minimum 0.25%) is kept between equity and *both* loss
  limits:
  * New trades halt once the remaining daily or overall loss budget, after open risk,
    drops below it (`binding_constraint="budget_floor"`).
  * A new trade may only spend 80% of the budget *above* the floor, so a full stop-out
    still leaves the whole cushion. Even a gap to twice the planned loss can't breach.
  * A trade whose allowed risk is under 25% of a standard trade is refused
    (`"min_trade_size"`) rather than micro-sized, so the engine can't creep toward the
    floor with tiny lots.
  * If the budget can't fund one minimum lot including costs, the trade is blocked
    (`"min_lot_floor"`).
  * In practice, trading stops at about 2.0% daily loss (limit 2.5%) and about 5.4–5.5%
    drawdown (limit 6%). On the real hourly gold data, the losing runs stopped $520–595
    above the drawdown line.
* **Leverage scaling.** The risk budget is multiplied by `min(1, 30 / asset_leverage)`,
  then capped by gross notional (3× equity) and margin use (50%). Lots are floored to
  the step size.
* **Ratchet stop.** This is a frozen dataclass: each `ratchet_stop()` call returns a new
  state and never mutates the old one. Breakeven (+0.05R) locks at +1R MFE. After that,
  an ATR chandelier trail tightens from 3.0× to 1.5× ATR as MFE grows. A property test
  checks the stop never loosens, including with NaN or zero ATR.
* **Day rollover.** Set `day_rollover_tz` to the firm's zone (FTMO uses
  `"Europe/Prague"`).
* **Fail-safe input handling.** A NaN or infinite equity mark is treated as a breach.

## Regime filter: why the ER look-back is 20

The formula is `ER = |C_t − C_{t−N}| / Σ|C_i − C_{i−1}|`. Directional entries are
blocked when `ER < 0.30`, and also during warm-up (the filter fails closed) and in
extreme volatility.

With Kaufman's classic N = 10, **the gate barely filtered anything in chop**. The
breakout bar itself makes up a large share of a 10-bar window, so ER is biased upward
exactly at signal time. Measured on the synthetic OU-chop regime over 5 seeds:

| ER period | chop trades | chop return | trend return |
|---|---|---|---|
| 10 | ~18 | −6.0% (hit DD cap) | +42% |
| **20** | **~3** | **−1.3%** | **+40%** |
| 30 | 0 | 0% | +35% |

N must be at least the breakout channel length. `tests/test_efficiency_ratio.py` checks
the suppression property directly: fewer than half the trades, a better return, and no
breach in chop.

## Generator-Critic (`agent/`)

The generator samples a seeded grid and counts **every** candidate it emits. The critic
splits the data chronologically, using the last 30% as out-of-sample (OOS). The OOS run
uses in-sample data only to warm up indicators. To be promoted, a candidate must pass
all of these:

1. At least 15 OOS trades. `CriticConfig` refuses to lower this floor.
2. A Lo (2002) t-statistic on per-trade returns: `SE = sqrt(η(1+SR²/2)/n)`, where η is a
   Bartlett-weighted autocorrelation inflation. η is floored at 1, so negative
   autocorrelation can never flatter the result.
3. That t-statistic must be ≥ `max(2.0, z_{1−α/(2·trials)})`. For 28 trials the hurdle
   is 3.12.
4. OOS Sharpe must keep at least 50% of in-sample Sharpe, and OOS expectancy must be
   positive.
5. Plausibility caps: OOS win rate ≤ 85% and in-sample PF ≤ 6.
6. No risk-engine breach, and OOS max drawdown ≤ 75% of the 6% cap.

On the bundled synthetic data the critic usually promotes **0–3 of 28** candidates. Most
fail the frequency floor or the snooping hurdle, or look implausibly good. This is
intended. When nothing is promoted, the paper dashboard runs a default breakout and
shows a banner saying so.

## Execution safety (`execution/`)

* **Brackets only.** No API opens a position without SL and TP. `BracketOrder` checks
  that both are on the correct side of the entry.
* **CEX atomicity.** Bybit, OKX and Bitget send SL and TP inside the entry request.
  Other venues use sequential placement. If any child order fails, the executor cancels
  the children, flattens the entry with a reduce-only order, and raises
  `BracketRollbackError`. If the rollback itself fails, the coordinator trips the kill
  switch.
* **Trade-only keys.** Keys come from environment variables only
  (`QM_CEX_API_KEY` / `_SECRET` / `_PASSWORD`). `connect()` queries the venue's
  permission endpoint (Binance `apiRestrictions`, Bybit `query-api`, OKX `account/config`):
  * A key with **withdrawal permission is refused in every mode.**
  * A key whose permissions can't be verified is refused on mainnet (fail closed).
  * `repr()` never shows secrets, and the audit log redacts credential-like keys.
* **MT5.** SL and TP travel inside the `TRADE_ACTION_DEAL` request, so brackets are
  atomic. `connect()` refuses investor (read-only) logins and a terminal with algo
  trading disabled. Stops closer than the broker's stops level are rejected before
  sending.
* **Kill switch.**
  * The flag is set before any network call, and the switch is safe to trip from many
    threads at once (tested with 16 threads).
  * It cancels orders before flattening and retries transient failures. Persistent
    failures are reported, not raised.
  * A sentinel file lets `python -m app.main kill` stop a running engine from another
    terminal, and a restarted engine starts tripped.
  * Re-arming needs the exact phrase `REARM` and flat books.

## Session compliance (`core/compliance.py`, `core/news_filter.py`)

* **Weekend guard:** from Friday 19:30 UTC to Sunday 22:00 UTC (configurable with
  `--friday-flatten HH:MM`; 19:30 leaves a 30-min buffer before the broker's 20:00 UTC gold close), open
  positions are flattened through `KillSwitch.soft_flatten` and new orders are blocked.
  This is a scheduled flatten, so the kill switch isn't latched. If the flatten fails,
  the coordinator escalates to a full kill-switch trip. The MT5 EA runs the same guard
  as a backup.
* **News blackout:** new entries are blocked from 5 minutes before to 5 minutes after
  high-impact events (NFP/CPI/FOMC are high impact by keyword, whatever the feed's
  label) in currencies that affect the symbol. Blackouts never flatten. With no
  calendar, or a stale one, the filter fails closed.
* **Where it applies:** the coordinator, the MCP `execute_order` tool (weekend guard
  always on; set `QM_NEWS_CALENDAR` for news), and the CLI:
  `python -m app.main dashboard --weekend-guard --news calendar.json`.

## MetaTrader 5 (ZeroMQ)

See [`mql5/README_MT5.md`](mql5/README_MT5.md). In short:
* The EA takes flat-JSON commands on `tcp://127.0.0.1:5555` and publishes heartbeats
  and deals on `:5556`.
* Every order's SL/TP is computed by `risk_engine.hard_bracket()`, which rounds them to
  the tick and respects the broker's stop distance. `build_bracket_payload()` refuses
  to encode an order without them, and the EA sends entry + SL + TP as one
  `TRADE_ACTION_DEAL` request.
* The heartbeat marks the link DOWN after 3 missed pings and reconnects with backoff.
  Orders are refused unless the link is UP.

## Live demo loop (`python -m app.main live`)

```powershell
$env:QM_MT5_TOKEN = "<EA InpAuthToken>"
python -m app.main live --symbol XAUUSD --timeframe 1h          # refuses unless a strategy is promoted
python -m app.main live --strategy ker_trend_vol --param er_min=0.35 --allow-unpromoted   # explicit demo override
```

* **Demo only.** `ACCOUNT_TRADE_MODE` is checked before any other request and on every
  loop step (the EA reports it in every PING). A REAL account, or an unknown or missing
  mode, raises `DemoAccountGuardError`. The loop exits with code 3 without sending or
  closing anything, since it never touches positions on a real account. The EA refuses
  too (`InpDemoOnly=true`).
* **Strategy gate.** At startup the Critic reviews the chosen strategy on the broker's
  own history (or runs a full research round). If nothing is promoted, it refuses to
  start unless you pass `--allow-unpromoted`, and that choice is written to the audit log.
* **Each poll** (default 5 s):
  * check the demo guard, the kill switch and heartbeat health;
  * mark equity to market, so the daily-loss and drawdown limits react within seconds;
  * apply the weekend flatten at wall-clock time;
  * on a new *closed* bar, make one decision. Bars missed during an outage become
    history; they're never traded late.
* **News:** `--news auto` (the default) downloads the weekly high-impact calendar at
  startup and fails if it can't. Other options are `--news FILE` or `--news off`.
* **Stopping:** Ctrl+C stops the loop and leaves positions protected by their SL/TP.
  `python -m app.main kill` flattens from another terminal.

## MCP server (Claude integration)

```bash
pip install mcp
claude mcp add quantmind -- python /abs/path/to/repo/mcp_server/mcp_bridge.py
# no-dependency alternative: line-delimited JSON-RPC 2.0 over stdio
python -m app.main mcp --jsonrpc
```

Tools: `fetch_ohlcv`, `backtest_strategy` (with a critic verdict), `get_risk_metrics`,
`execute_order`, `trip_kill_switch`.

`execute_order` never takes a size from the caller: the risk engine computes it. Live
orders need all four of the following, otherwise they are refused:

* `mode="live"`
* `confirm=true`
* the operator has set `QM_LIVE_TRADING=ENABLED`
* a live broker is attached in code

## Known limitations

* **Bar-level risk monitoring.** In live use, call `TradingCoordinator.on_bar` on a short
  cadence (for example 1-minute bars). A gap straight through a stop can still overshoot
  the DLL. The 80% headroom buffer and the 2.5%-vs-5% margin exist to absorb that.
* **Test coverage of live adapters.** The CEX and MT5 adapters are tested against fakes
  of the ccxt and MT5 interfaces, not against live venues. Validate on a testnet or demo
  account before use. The MQL5 Expert Advisor (`mql5/ZeroMQ_Listener.mq5`) is included
  but is not covered by the Python test suite.
* **MCP SDK compatibility** (1.x `FastMCP` and 2.x `MCPServer`) is tested mostly with
  fakes of the SDK. One test uses the real `mcp` package when it is installed. The JSON-RPC transport and all tool logic are tested directly.
* **Funding carry.** `FundingCarryStrategy` needs a funding-rate feed. The backtester
  covers directional strategies only.
