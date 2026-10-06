# MT5 ZeroMQ Bridge — Setup Guide

This connects the Python orchestrator to MetaTrader 5 through `ZeroMQ_Listener.mq5`, an
Expert Advisor that executes commands and reports back over ZeroMQ.

```
Python (execution/mt5_adapter.py)                       MetaTrader 5 (ZeroMQ_Listener.mq5)
  ZmqMt5Transport  ── REQ ─ tcp://127.0.0.1:5555 ─ REP ──►  command handler ─► CTrade (Buy/Sell with SL+TP)
  HeartbeatMonitor ── PING every 1 s ───────────────────►  PING reply (trade permission, weekend state)
  Mt5EventSubscriber ◄─ SUB ─ tcp://127.0.0.1:5556 ─ PUB ─  heartbeats, deals, audit lines
                                                          └► MQL5/Files/QuantMind/ea_audit.jsonl
```

> **Use a demo or challenge account first.** Every step below was checked with MetaEditor
> build tooling and a mock EA over real sockets. The EA has **not** been run against a
> live broker by the author of this guide.

---

## 1. Prerequisites

| Item | Version |
|---|---|
| MetaTrader 5 terminal (Windows) | any current build |
| Python | 3.11+ |
| `pyzmq` | `pip install pyzmq` (bundles libzmq 4.3.x) |
| `mql-zmq` binding | <https://github.com/dingmaotu/mql-zmq> (Apache-2.0) |

Find the terminal's **data folder**: in MT5, choose **File → Open Data Folder**. It looks
like `%APPDATA%\MetaQuotes\Terminal\<hash>\`. Below, it is called `<DATA>`.

## 2. Install the ZeroMQ binding into MT5

```powershell
git clone --depth 1 https://github.com/dingmaotu/mql-zmq.git
Copy-Item -Recurse mql-zmq\Include\Zmq  "<DATA>\MQL5\Include\"
Copy-Item -Recurse mql-zmq\Include\Mql  "<DATA>\MQL5\Include\"
Copy-Item mql-zmq\Library\MT5\libzmq.dll, mql-zmq\Library\MT5\libsodium.dll "<DATA>\MQL5\Libraries\"
```

### 2a. Patch the binding for current MT5 builds (required)

Upstream `mql-zmq` has not been updated since Dec 2023, and current MetaEditor compilers
reject it with 26 errors like
`error 246: parameter convertion type 'char[]' to 'uchar[] &' is not allowed`.
Run the bundled patcher. It rewrites `char` byte buffers to `uchar`, which is
ABI-identical, keeps each file's encoding, and saves `.orig` backups:

```powershell
python mql5\tools\patch_mql_zmq.py "<DATA>\MQL5\Include\Zmq"
# patched Socket.mqh: 9 char -> uchar ... 5 file(s) patched
```

Running it again is safe ("already patched").

## 3. Compile the EA

1. Copy `mql5\ZeroMQ_Listener.mq5` to `<DATA>\MQL5\Experts\QuantMind\`.
2. Open it in MetaEditor (F4 from the terminal) and press **F7 (Compile)**.
   Expected result: **`0 errors, 0 warnings`**.

To compile from the command line (useful for CI):

```powershell
& "C:\Program Files\MetaTrader 5\MetaEditor64.exe" `
    /compile:"<DATA>\MQL5\Experts\QuantMind\ZeroMQ_Listener.mq5" /log:"compile.log"
Get-Content compile.log -Encoding Unicode | Select-String "Result"
```

## 4. Terminal settings

**Tools → Options → Expert Advisors:**

- ☑ **Allow algorithmic trading**
- ☑ **Allow DLL imports**. The binding calls `libzmq.dll`; without this the EA fails on load.
- ☐ You don't need to add WebRequest URLs.

The toolbar **Algo Trading** button must be green. Traffic stays on `127.0.0.1`, so no
firewall rule is needed.

## 5. Attach the EA

Drag **QuantMind → ZeroMQ_Listener** onto **one** chart. Any symbol works, because the EA
trades whatever symbol each command names. On the **Common** tab, tick *Allow Algo
Trading* and *Allow DLL imports*. Then set the inputs:

| Input | Default | Notes |
|---|---|---|
| `InpAuthToken` | *(empty)* | **Required.** The EA refuses to start without it. Generate one with `python -c "import secrets; print(secrets.token_hex(16))"`. |
| `InpRepEndpoint` / `InpPubEndpoint` | `tcp://127.0.0.1:5555` / `:5556` | Keep loopback unless Python runs on another host. |
| `InpTimerMs` | 5 | Command poll interval. `FLATTEN_ALL` starts within one tick. |
| `InpWeekendGuard` | true | Backup to Python: blocks orders and flattens from Fri 19:30 to Sun 22:00 UTC. |
| `InpFlattenDow` / `InpFlattenHourUTC` / `InpFlattenMinuteUTC` | 5 / 19 / 30 | MQL numbers days **0=Sun … 5=Fri** (Python uses Mon=0 … Fri=4). |
| `InpReopenDow` / `InpReopenHourUTC` | 0 / 22 | Sunday 22:00 UTC. |
| `InpDeadManSec` | 0 (off) | If > 0, flatten when no Python command arrives for this many seconds. Heartbeats count, so 5–10 s is reasonable once running. |
| `InpAccountWideFlatten` | true | Guards flatten **every** position on the account, not only this engine's magic. |
| `InpMagic` | 260926 | Scope used when `InpAccountWideFlatten=false`. |
| `InpMaxDeviationPts` | 50 | Caps any requested slippage. |
| `InpAuditFile` | `QuantMind\ea_audit.jsonl` | Relative to `<DATA>\MQL5\Files\`. |
| `InpDemoOnly` | true | The EA refuses to start, and refuses `SEND_BRACKET`, on a REAL account. |
| `InpMaxRatesBars` | 5000 | Cap on the bars returned per `RATES` request. |

The **Experts** tab should show nothing alarming. The audit file's first line is an
`ea_start` record.

Only **one** EA instance may bind a port. A second instance on another chart fails with
"failed to bind … (port in use?)".

## 6. Connect Python

```powershell
pip install pyzmq
$env:QM_MT5_TOKEN = "<same token as InpAuthToken>"
```

```python
import os
from execution.mt5_bridge import ZmqMt5Transport, HeartbeatMonitor
from execution.mt5_adapter import Mt5Adapter
from execution.kill_switch import KillSwitch

transport = ZmqMt5Transport("tcp://127.0.0.1:5555", token=os.environ["QM_MT5_TOKEN"])
print(transport.ping())                 # {'trade_allowed': True, 'connected': True, ...}

monitor = HeartbeatMonitor(transport)   # PING every 1 s, DOWN after 3 misses, reconnect with backoff
kill = KillSwitch(sentinel_path="state/KILL_SWITCH")
mt5 = Mt5Adapter(transport, monitor=monitor, kill_guard=kill)
mt5.connect()                           # checks trade permissions, starts the heartbeat thread

# Use it as the coordinator's broker; TradingCoordinator registers it with the kill switch.
# from agent.coordinator import TradingCoordinator
# from core.compliance import ComplianceGate, WeekendGuard
# coord = TradingCoordinator("XAUUSD", asset, mt5, kill, strategies,
#                            compliance=ComplianceGate(WeekendGuard()))
```

Pass real broker values in `AssetSpec` (`price_tick`, `min_stop_distance`,
`contract_size`, lot sizes), so the risk engine computes SL/TP that the broker will
accept.

## 7. Verify on a demo account

1. Run the connectivity check:

   ```powershell
   $env:QM_MT5_TOKEN = "<same token as InpAuthToken>"
   python -m app.main mt5-ping                  # --symbol XAUUSD --endpoint tcp://127.0.0.1:5555
   ```

   It pings the EA, prints the account and symbol specification, and checks them
   against our `AssetSpec`. The key check is that our 0.01 gold tick is a multiple of
   the broker's `SYMBOL_TRADE_TICK_SIZE`. Lot and contract differences are reported as
   `WARN`. Exit code 0 means the link is ready. If the EA can't be reached, it prints a
   checklist of likely causes.
2. For the full details, `transport.symbol_info("XAUUSD")` should match the Market Watch
   specification.
3. Place a tiny bracket with `mt5.place_bracket(BracketOrder(...))`. Check that the
   position in the Toolbox shows **both SL and TP**.
4. Run `python -m app.main kill --reason test`, or call `kill.trip("test")` in-process.
   The position should close within one timer tick. `ea_audit.jsonl` shows `flatten_all`
   with `elapsed_ms`.
5. Check the audit log: every line has `ts` (UTC, milliseconds), `ts_ms`,
   `server_time` and the MT5 result (`retcode`, `order`, `deal`, `price`,
   `latency_ms`).

## 8. Protocol reference

Requests are **flat** JSON objects (strings, numbers, booleans and null only), so the EA
can parse them without a JSON library. Every request carries `v`, `id`, `token` and
`action`. Replies echo the `id`.

| Action | Fields | Reply `data` | Retried by Python on timeout |
|---|---|---|---|
| `PING` | — | `trade_allowed, connected, weekend_blackout, server_time, gmt_ms, equity, positions, trade_mode, login` | yes |
| `ACCOUNT_INFO` | — | `login, trade_mode (0 demo, 1 contest, 2 real), trade_mode_name, server, currency, equity, balance, margin_free, leverage, trade_allowed, trade_expert` | yes |
| `SYMBOL_INFO` | `symbol` | `digits, point, trade_tick_size, bid, ask, volume_min/max/step, trade_contract_size, trade_stops_level, filling_mode` | yes |
| `SEND_BRACKET` | `symbol, side (BUY/SELL), volume, sl, tp, deviation, magic, client_id, comment` | `retcode, order, deal, price, volume, repair, latency_ms` | **no** (see below) |
| `MODIFY_SLTP` | `ticket, symbol, sl, tp` (both required) | `retcode` | no |
| `POSITIONS` / `PENDING_ORDERS` | `magic` (-1 = all) | list | yes |
| `CANCEL_ALL` | `magic` | `cancelled, failed[]` | yes |
| `CLOSE_POSITION` | `ticket, deviation` | `retcode` | yes |
| `FLATTEN_ALL` | `magic, deviation` | `closed, cancelled, failed[], elapsed_ms` | yes |
| `RATES` | `symbol, timeframe (M1..D1), count` | `server_offset, bars[[t_server,o,h,l,c,v]...]`. Closed bars only; Python converts to UTC. | yes |

**Why `SEND_BRACKET` is never retried:** a timeout means the outcome is unknown. Python
never re-sends. Instead, it looks for the order's `client_id` among open positions. The
EA also caches results by `client_id`, so even a duplicate request can't open a second
position.

## 9. Safety guarantees and who enforces them

| Guarantee | Python | EA |
|---|---|---|
| SL and TP mandatory, correct side | `hard_bracket()` and `build_bracket_payload()` | rejects `sl<=0` / `tp<=0` / wrong side |
| SL/TP travel in the same request as the entry | — | `CTrade.Buy/Sell(volume, symbol, 0, sl, tp)` = one `TRADE_ACTION_DEAL` |
| Position left without SL/TP | — | repaired with `PositionModify`, or closed |
| Netting account with an existing position | — | rejected (it would merge and overwrite SL/TP) |
| Weekend flatten (Fri 19:30 UTC) | `ComplianceGate` + `KillSwitch.soft_flatten` | backup timer guard |
| High-impact news ±5 min | `NewsFilter` (fails closed on a stale calendar) | — |
| Link health | `HeartbeatMonitor`; orders refused unless UP | `PING` reports trade permission |
| Kill switch | `KillSwitch.trip` → `CANCEL_ALL` + `FLATTEN_ALL` | executes both in-terminal |
| Authentication | token in every request | rejects a mismatch; won't start without a token |

The EA's weekend guard uses `TimeGMT()`, which comes from the PC clock. **Keep Windows
time synced (NTP).** Some firms define the trading day on server time; the guard is
expressed in UTC on purpose.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error 246 ... 'char[]' to 'uchar[] &'` on compile | Run `mql5/tools/patch_mql_zmq.py` (step 2a). |
| `cannot load 'libzmq.dll'` / `DLL imports not allowed` | Enable *Allow DLL imports* globally **and** on the EA's Common tab; check the DLLs are in `MQL5\Libraries`. |
| EA removed immediately, `INIT_PARAMETERS_INCORRECT` | `InpAuthToken` is empty. |
| `failed to bind ... (port in use?)` | Another EA instance or process holds 5555/5556. Use one instance, or change both endpoints on both sides. |
| Python `Mt5Timeout` / heartbeat `DOWN` | EA not attached, Algo Trading off, or the terminal is busy. The monitor reconnects on its own (backoff up to 10 s). |
| `EA rejected PING: authentication failed` | The token differs between `InpAuthToken` and Python. |
| `netting account already has a position` | Netting accounts hold one position per symbol. Close it first, or use a hedging account. |
| `retcode 10030` (unsupported filling) | The broker's filling mode. `SetTypeFillingBySymbol` normally handles it; report the symbol's `filling_mode`. |
| `retcode 10016` (invalid stops) | SL/TP too close. Set `AssetSpec.min_stop_distance` = `trade_stops_level * point`. |

## 11. Prop-firm notes

- Most firms (FTMO, The5ers, FundedNext) allow EAs, but **rules on news trading and
  weekend holding vary by firm and account type**. Configure `NewsFilterConfig` (for
  example `before=after=2 min` for FTMO funded accounts) and `WeekendGuardConfig` to
  match your firm, and read its current rules.
- The news filter **only blocks entries**. It never closes positions during the window,
  because some firms also forbid *closing* trades near news.
- Account-wide flatten (the default) is deliberate: every open position on the account
  counts toward the firm's loss limits, not just this engine's positions.
