"""Broker-neutral order types, the adapter protocol, and a paper broker.

A :class:`BracketOrder` is the only way to open a position anywhere in this codebase:
there is intentionally no "naked entry" API. Validation guarantees that the stop and
take-profit sit on the correct sides of the entry before any adapter sees the order.
"""
from __future__ import annotations

import itertools
import math
import threading
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol, runtime_checkable

from core.risk_engine import Side


class ExecutionMode(str, Enum):
    """Paper trades are simulated locally; live trades reach a real venue."""

    PAPER = "paper"
    LIVE = "live"


class OrderRejected(RuntimeError):
    """An order was refused before or by the venue (no position was left open)."""


class BracketRollbackError(RuntimeError):
    """A non-atomic bracket failed part-way and was rolled back (position flattened)."""


class KillSwitchEngaged(RuntimeError):
    """An order was attempted while the kill switch is tripped."""


@dataclass(frozen=True, slots=True)
class BracketOrder:
    """Entry + stop-loss + take-profit, submitted as one unit.

    ``entry_price`` is the reference price (limit price for LIMIT entries, expected
    fill for MARKET entries). ``client_id`` is an idempotency key.
    """

    symbol: str
    side: Side
    units: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_type: str = "market"
    client_id: str = field(default_factory=lambda: "qm-" + uuid.uuid4().hex[:16])
    strategy: str = "manual"

    def __post_init__(self) -> None:
        if not (self.units > 0 and math.isfinite(self.units)):
            raise OrderRejected(f"units must be positive, got {self.units}")
        if self.entry_type not in ("market", "limit"):
            raise OrderRejected(f"unsupported entry type {self.entry_type}")
        if (self.entry_price - self.stop_loss) * self.side.sign <= 0:
            raise OrderRejected("stop-loss is on the wrong side of entry")
        if (self.take_profit - self.entry_price) * self.side.sign <= 0:
            raise OrderRejected("take-profit is on the wrong side of entry")
        if min(self.entry_price, self.stop_loss, self.take_profit) <= 0:
            raise OrderRejected("prices must be positive")

    @property
    def risk_per_unit(self) -> float:
        """Price distance to the stop."""
        return abs(self.entry_price - self.stop_loss)


@dataclass(frozen=True, slots=True)
class BracketResult:
    """Outcome of a bracket submission."""

    accepted: bool
    client_id: str
    fill_price: float | None = None
    venue_ids: tuple[str, ...] = ()
    message: str = ""
    atomic: bool = True


@dataclass(frozen=True, slots=True)
class Position:
    """An open position as reported by an adapter."""

    symbol: str
    side: Side
    units: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    client_id: str = ""

    def unrealised(self, price: float, unit_value: float = 1.0) -> float:
        """Floating PnL at ``price``."""
        return (price - self.entry_price) * self.side.sign * self.units * unit_value

    def open_risk(self, unit_value: float = 1.0) -> float:
        """Loss if the current stop is hit (0 when the stop is at/through breakeven)."""
        if self.stop_loss is None:
            return math.inf
        return max(0.0, (self.entry_price - self.stop_loss) * self.side.sign) * self.units * unit_value


@runtime_checkable
class BrokerAdapter(Protocol):
    """What the coordinator and kill switch need from any venue."""

    name: str
    mode: ExecutionMode

    def place_bracket(self, order: BracketOrder) -> BracketResult:
        """Submit entry, SL and TP atomically (or roll back)."""
        ...

    def amend_stop(self, symbol: str, new_stop: float) -> bool:
        """Move the protective stop of the open position on ``symbol``."""
        ...

    def cancel_all_orders(self) -> int:
        """Cancel every pending order; return how many were cancelled."""
        ...

    def flatten_all(self) -> int:
        """Market-close every open position; return how many were closed."""
        ...

    def positions(self) -> list[Position]:
        """Currently open positions."""
        ...

    def equity(self) -> float:
        """Account equity in account currency."""
        ...


@dataclass(frozen=True, slots=True)
class Fill:
    """A paper-broker closing fill."""

    symbol: str
    side: Side
    units: float
    entry: float
    exit: float
    pnl: float
    reason: str


class PaperBroker:
    """Thread-safe in-memory broker used for paper trading and tests.

    Positions carry their SL/TP natively (like MT5), so brackets are atomic by
    construction. :meth:`on_bar` simulates intrabar stop/target fills with the same
    conservative rules as the backtester (gap fills at the open, stop before target).
    """

    def __init__(self, starting_cash: float = 100_000.0, slippage_bps: float = 1.0,
                 commission_bps: float = 2.0, unit_value: float = 1.0, guard: KillGuard | None = None) -> None:
        self.name = "paper"
        self.mode = ExecutionMode.PAPER
        self._lock = threading.RLock()
        self._cash = starting_cash
        self._slip = slippage_bps / 10_000
        self._comm = commission_bps / 10_000
        self._unit_value = unit_value
        self._positions: dict[str, Position] = {}
        self._last_price: dict[str, float] = {}
        self._ids = itertools.count(1)
        self.fills: list[Fill] = []
        self.guard = guard

    # -- market data ---------------------------------------------------------------
    def mark(self, symbol: str, price: float) -> None:
        """Update the last traded price used for market fills and equity."""
        with self._lock:
            self._last_price[symbol] = price

    def on_bar(self, symbol: str, open_: float, high: float, low: float, close: float) -> list[Fill]:
        """Process SL/TP for ``symbol`` against a completed bar; then mark at close."""
        out: list[Fill] = []
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is not None and pos.stop_loss is not None and pos.take_profit is not None:
                if pos.side is Side.LONG:
                    stop_hit, tp_hit = low <= pos.stop_loss, high >= pos.take_profit
                    stop_px, tp_px = min(open_, pos.stop_loss), max(open_, pos.take_profit)
                else:
                    stop_hit, tp_hit = high >= pos.stop_loss, low <= pos.take_profit
                    stop_px, tp_px = max(open_, pos.stop_loss), min(open_, pos.take_profit)
                if stop_hit:
                    out.append(self._close(pos, stop_px, "stop_loss"))
                elif tp_hit:
                    out.append(self._close(pos, tp_px, "take_profit"))
            self._last_price[symbol] = close
        return out

    # -- adapter protocol ----------------------------------------------------------
    def place_bracket(self, order: BracketOrder) -> BracketResult:
        if self.guard is not None and self.guard.tripped:
            raise KillSwitchEngaged("kill switch engaged - order refused")
        with self._lock:
            if order.symbol in self._positions:
                return BracketResult(False, order.client_id, message="position already open on symbol")
            ref = self._last_price.get(order.symbol, order.entry_price)
            fill = ref * (1 + order.side.sign * self._slip) if order.entry_type == "market" else order.entry_price
            shift = fill - order.entry_price
            sl, tp = order.stop_loss + shift, order.take_profit + shift
            self._cash -= fill * order.units * self._unit_value * self._comm
            self._positions[order.symbol] = Position(order.symbol, order.side, order.units, fill, sl, tp,
                                                     order.client_id)
            n = next(self._ids)
            return BracketResult(True, order.client_id, fill, (f"P{n}-entry", f"P{n}-sl", f"P{n}-tp"),
                                 "filled (native bracket)")

    def amend_stop(self, symbol: str, new_stop: float) -> bool:
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return False
            if pos.stop_loss is not None and (new_stop - pos.stop_loss) * pos.side.sign < 0:
                return False  # never loosen a stop
            self._positions[symbol] = replace(pos, stop_loss=new_stop)
            return True

    def cancel_all_orders(self) -> int:
        return 0  # paper entries are market-only; nothing rests

    def flatten_all(self) -> int:
        with self._lock:
            positions = list(self._positions.values())
            for pos in positions:
                self._close(pos, self._last_price.get(pos.symbol, pos.entry_price), "flatten")
            return len(positions)

    def positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def equity(self) -> float:
        with self._lock:
            floating = sum(p.unrealised(self._last_price.get(p.symbol, p.entry_price), self._unit_value)
                           for p in self._positions.values())
            return self._cash + floating

    # -- internals -------------------------------------------------------------------
    def _close(self, pos: Position, raw_price: float, reason: str) -> Fill:
        exit_px = raw_price * (1 - pos.side.sign * self._slip)
        pnl = pos.unrealised(exit_px, self._unit_value) - exit_px * pos.units * self._unit_value * self._comm
        self._cash += pnl
        del self._positions[pos.symbol]
        fill = Fill(pos.symbol, pos.side, pos.units, pos.entry_price, exit_px, pnl, reason)
        self.fills.append(fill)
        return fill


class KillGuard(Protocol):
    """Anything exposing a ``tripped`` flag (the kill switch)."""

    @property
    def tripped(self) -> bool:
        """Whether trading is halted."""
        ...
