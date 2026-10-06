"""Live MT5 orchestrator loop with a hard demo-only guard.

::

    LiveRunner.run()
      preflight(): ACCOUNT_INFO ──► enforce_demo_account()   (REAL -> DemoAccountGuardError)
                   RATES        ──► prime bar history (closed bars, UTC)
      loop every poll_seconds:
        step():  trade mode from the latest PING ──► enforce_demo_account()   (every step)
                 kill switch tripped?            ──► HALTED (no orders)
                 heartbeat unhealthy?            ──► WAITING_LINK (no orders)
                 coordinator.supervise(now)      ──► mark-to-market, weekend flatten
                 new closed bar?                 ──► coordinator.on_bar() ──► Mt5Adapter

Safety properties:

* **Demo only, continuously.** The account type is checked before the first request
  that could trade and again on every step (the EA reports ``trade_mode`` in every
  PING), so switching the terminal to a real account mid-run stops the loop.
  An unknown or missing account type is treated as real (fail closed). The EA has its
  own ``InpDemoOnly`` guard as a second, independent layer.
* **A real account is never touched.** On a guard violation the runner raises
  :class:`DemoAccountGuardError` *without* flattening: it did not open positions on
  that account and has no business closing them.
* **Stale data never trades.** Only the latest closed bar can produce an entry; after a
  reconnect gap the missed bars are appended as history but not traded.
* **Normal shutdown leaves positions protected** by their broker-side SL/TP; use the
  kill switch to flatten.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from agent.coordinator import CycleReport, TradingCoordinator
from core.audit import AuditTrail
from core.market_data import Bar
from core.risk_engine import AssetSpec, RiskConfigError
from execution.kill_switch import KillSwitch
from execution.mt5_bridge import (
    TRADE_MODE_CONTEST,
    TRADE_MODE_DEMO,
    TRADE_MODE_REAL,
    HeartbeatMonitor,
    Mt5Error,
    Mt5Transport,
)

ALLOWED_TRADE_MODES: frozenset[int] = frozenset({TRADE_MODE_DEMO, TRADE_MODE_CONTEST})
_TRADE_MODE_NAMES = {TRADE_MODE_DEMO: "DEMO", TRADE_MODE_CONTEST: "CONTEST", TRADE_MODE_REAL: "REAL"}

_TIMEFRAMES: dict[str, tuple[str, timedelta]] = {
    "1m": ("M1", timedelta(minutes=1)), "5m": ("M5", timedelta(minutes=5)),
    "15m": ("M15", timedelta(minutes=15)), "30m": ("M30", timedelta(minutes=30)),
    "1h": ("H1", timedelta(hours=1)), "4h": ("H4", timedelta(hours=4)), "1d": ("D1", timedelta(days=1)),
}


class DemoAccountGuardError(RuntimeError):
    """CRITICAL: the connected MT5 account is not a demo/contest account."""


def enforce_demo_account(info: dict[str, Any]) -> int:
    """Raise :class:`DemoAccountGuardError` unless ``info`` describes a DEMO or CONTEST account.

    ``info`` is an ``ACCOUNT_INFO`` or ``PING`` reply. A missing, malformed or unknown
    ``trade_mode`` is rejected (fail closed). Returns the verified trade mode.
    """
    raw = info.get("trade_mode")
    try:
        mode = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise DemoAccountGuardError(
            f"CRITICAL: account trade mode unknown ({raw!r}) - refusing to trade (fail closed). "
            "Is the EA older than the RATES/trade_mode protocol?") from None
    if mode not in ALLOWED_TRADE_MODES:
        name = _TRADE_MODE_NAMES.get(mode, f"UNKNOWN({mode})")
        raise DemoAccountGuardError(
            f"CRITICAL: MT5 account {info.get('login', '?')} is {name} - the orchestrator runs ONLY on "
            "DEMO/CONTEST accounts. Aborting without sending any order.")
    return mode


def parse_timeframe(tf: str) -> tuple[str, timedelta]:
    """``"1h"`` -> ``("H1", 1 hour)``. Accepts the MT5 code too (``"H1"``)."""
    key = tf.strip().lower()
    for short, (code, delta) in _TIMEFRAMES.items():
        if key in (short, code.lower()):
            return code, delta
    raise ValueError(f"unsupported timeframe {tf!r}; use one of {sorted(_TIMEFRAMES)}")


def asset_from_symbol_info(symbol: str, sym: dict[str, Any], account: dict[str, Any],
                           commission_bps: float = 2.0, slippage_bps: float = 1.0) -> AssetSpec:
    """Build the :class:`AssetSpec` from the broker's live symbol and account data.

    Sizing assumes PnL in the quote currency equals account currency; a ``...USD``
    symbol on a non-USD account is refused rather than mis-sized.
    """
    currency = str(account.get("currency", "")).upper()
    if symbol.upper().endswith("USD") and currency and currency != "USD":
        raise RiskConfigError(f"{symbol} PnL is in USD but the account is in {currency}: "
                              "quote-to-account conversion is not implemented - refusing to size")
    point = float(sym.get("point") or 0.0)
    return AssetSpec(
        symbol,
        leverage=max(1.0, float(account.get("leverage") or 1)),
        contract_size=float(sym.get("trade_contract_size") or 1.0),
        lot_step=float(sym.get("volume_step") or 0.01),
        min_lot=float(sym.get("volume_min") or 0.01),
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        price_tick=float(sym.get("trade_tick_size") or point),
        min_stop_distance=float(sym.get("trade_stops_level") or 0) * point,
    )


class Mt5BarFeed:
    """Closed-bar feed from the EA's ``RATES`` action, converted from server time to UTC."""

    def __init__(self, transport: Mt5Transport, symbol: str, timeframe: str, history: int = 500) -> None:
        self.transport = transport
        self.symbol = symbol
        self.code, self.period = parse_timeframe(timeframe)
        self.history = history
        self.last_ts: datetime | None = None

    def fetch(self, count: int) -> list[Bar]:
        """Up to ``count`` most recent closed bars, oldest first, timestamps in UTC."""
        data = self.transport.rates(self.symbol, self.code, count)
        offset = int(data.get("server_offset") or 0)
        bars = [Bar(datetime.fromtimestamp(int(r[0]) - offset, tz=UTC), float(r[1]), float(r[2]),
                    float(r[3]), float(r[4]), float(r[5])) for r in data.get("bars", [])]
        bars.sort(key=lambda b: b.ts)
        return bars

    def prime(self) -> list[Bar]:
        """Initial history; sets the high-water mark for :meth:`poll`."""
        bars = self.fetch(self.history)
        if not bars:
            raise Mt5Error(f"no {self.code} history for {self.symbol} (open a chart so MT5 downloads it)")
        self.last_ts = bars[-1].ts
        return bars

    def poll(self, lookback: int = 20) -> list[Bar]:
        """Closed bars newer than the last one seen (empty if none)."""
        bars = self.fetch(lookback)
        new = [b for b in bars if self.last_ts is None or b.ts > self.last_ts]
        if new:
            self.last_ts = new[-1].ts
        return new


class RunnerState(str, Enum):
    """Lifecycle of the live loop."""

    INIT = "init"
    RUNNING = "running"
    WAITING_LINK = "waiting_link"
    HALTED = "halted"  # kill switch engaged: supervised, never trades
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class StepReport:
    """Outcome of one loop iteration."""

    state: RunnerState
    now: datetime
    new_bars: int
    cycle: CycleReport | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """Loop settings."""

    poll_seconds: float = 5.0
    max_history: int = 1_000


class LiveRunner:
    """Continuous MT5 loop around a :class:`TradingCoordinator` (see module docstring)."""

    def __init__(self, coordinator: TradingCoordinator, feed: Mt5BarFeed, transport: Mt5Transport,
                 kill_switch: KillSwitch, monitor: HeartbeatMonitor | None = None,
                 config: LiveConfig | None = None, audit: AuditTrail | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.coordinator = coordinator
        self.feed = feed
        self.transport = transport
        self.kill_switch = kill_switch
        self.monitor = monitor
        self.config = config or LiveConfig()
        self.audit = audit or coordinator.audit
        self.clock = clock
        self.sleep = sleep
        self.state = RunnerState.INIT
        self.bars: list[Bar] = []
        self.account: dict[str, Any] = {}
        self.steps = 0

    # ------------------------------------------------------------------------------
    def _guard(self, info: dict[str, Any], where: str) -> None:
        try:
            enforce_demo_account(info)
        except DemoAccountGuardError as exc:
            self.state = RunnerState.STOPPED
            self.audit.record("demo_guard_violation", {"where": where, "error": str(exc),
                                                       "login": info.get("login")})
            raise

    def preflight(self) -> None:
        """Verify the account is DEMO/CONTEST, then load bar history. No orders before this passes."""
        self.account = self.transport.account_info()
        self._guard(self.account, "preflight")
        self.bars = self.feed.prime()
        self.state = RunnerState.RUNNING
        self.audit.record("live_start", {
            "symbol": self.feed.symbol, "timeframe": self.feed.code, "login": self.account.get("login"),
            "server": self.account.get("server"), "trade_mode": self.account.get("trade_mode_name"),
            "history_bars": len(self.bars), "last_bar": self.bars[-1].ts.isoformat()})

    def _latest_link_info(self) -> dict[str, Any]:
        if self.monitor is not None:
            return self.monitor.last_info
        return self.transport.ping()

    def step(self) -> StepReport:
        """One iteration. Raises :class:`DemoAccountGuardError` on a non-demo account."""
        if self.state in (RunnerState.INIT, RunnerState.STOPPED):
            raise RuntimeError("call preflight() first")
        self.steps += 1
        now = self.clock()

        # 1) Demo guard on every step, from the freshest account snapshot we have.
        info = self._latest_link_info()
        if info:
            self._guard(info, "step")

        # 2) Kill switch (including an external sentinel written by `app.main kill`).
        ext = self.kill_switch.check_external_trigger()
        if self.kill_switch.tripped:
            self.state = RunnerState.HALTED
            note = f"kill switch engaged ({ext.reason})" if ext else "kill switch engaged"
            return StepReport(self.state, now, 0, notes=(note,))

        # 3) Link health: without a healthy link we neither trade nor pretend to.
        if self.monitor is not None and not self.monitor.healthy():
            if self.state is not RunnerState.WAITING_LINK:
                self.audit.record("live_link_down", {"state": self.monitor.state.value})
            self.state = RunnerState.WAITING_LINK
            return StepReport(self.state, now, 0, notes=(f"link {self.monitor.state.value}",))
        if self.state is RunnerState.WAITING_LINK:
            self.audit.record("live_link_restored", {})
        self.state = RunnerState.RUNNING

        # 4) Between-bar supervision: equity limits + weekend flatten at wall-clock time.
        notes = self.coordinator.supervise(now)
        if self.kill_switch.tripped:
            self.state = RunnerState.HALTED
            return StepReport(self.state, now, 0, notes=tuple(notes))

        # 5) New closed bar(s) -> one decision on the latest only.
        new = self.feed.poll()
        if not new:
            return StepReport(self.state, now, 0, notes=tuple(notes))
        self.bars.extend(new)
        self.bars = self.bars[-self.config.max_history:]
        cycle = self.coordinator.on_bar(self.bars, now=now)
        if len(new) > 1:
            notes.append(f"caught up {len(new) - 1} missed bar(s) as history only")
        return StepReport(self.state, now, len(new), cycle, tuple(notes))

    def run(self, max_steps: int | None = None, on_step: Callable[[StepReport], Any] | None = None) -> None:
        """Preflight, then loop until ``max_steps`` or Ctrl+C. Transport errors are retried."""
        self.preflight()
        try:
            while max_steps is None or self.steps < max_steps:
                try:
                    report = self.step()
                except Mt5Error as exc:  # step() already counted this attempt
                    report = StepReport(self.state, self.clock(), 0, notes=(f"transport error: {exc}",))
                if on_step is not None:
                    on_step(report)
                if max_steps is not None and self.steps >= max_steps:
                    break
                self.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            if self.state is not RunnerState.STOPPED:
                self.state = RunnerState.STOPPED
                self.audit.record("live_stop", {"steps": self.steps})
