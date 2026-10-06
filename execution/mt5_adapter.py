"""MT5 broker adapter: the :class:`~execution.base.BrokerAdapter` over an MT5 transport.

Responsibilities on top of the raw transport:

* **Mandatory brackets** — every order is encoded with
  :func:`~execution.mt5_bridge.build_bracket_payload`, which refuses to produce a
  payload without valid SL and TP. The EA re-validates on its side.
* **Link gating** — with a :class:`~execution.mt5_bridge.HeartbeatMonitor` attached,
  new orders are refused while the link is not healthy. Flatten/cancel are *always*
  attempted (they are what you want when things go wrong).
* **Unknown-outcome reconciliation** — if ``SEND_BRACKET`` times out, the order may or
  may not have filled. The adapter never re-sends; it queries open positions for the
  order's ``client_id`` and reports what actually happened.
* **Single-shot flatten** — the kill switch's ``flatten_all`` maps to one
  ``FLATTEN_ALL`` command executed inside the EA (milliseconds, no per-ticket round
  trips). ``cancel_all_orders`` maps to one ``CANCEL_ALL``.
"""
from __future__ import annotations

import math
import threading
from typing import Any

from core.risk_engine import Side
from execution.base import (
    BracketOrder,
    BracketResult,
    ExecutionMode,
    KillGuard,
    KillSwitchEngaged,
    OrderRejected,
    Position,
)
from execution.mt5_bridge import (
    RETCODE_OK,
    HeartbeatMonitor,
    Mt5Error,
    Mt5Timeout,
    Mt5Transport,
    PayloadError,
    build_bracket_payload,
    sanitize_comment,
)


def _round_step(volume: float, step: float) -> float:
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    return round(math.floor(volume / step + 1e-9) * step, decimals)


class Mt5Adapter:
    """MT5 execution adapter.

    Args:
        transport: ZeroMQ or native transport.
        monitor: Optional heartbeat monitor; when present, orders require a healthy link.
        magic: Magic number identifying this engine's positions.
        deviation: Max slippage in points for market orders.
        kill_guard: Kill switch; orders are refused while it is tripped.
        account_wide_kill: Cancel/flatten *every* position on the account (not only this
            magic) — the prop-firm default, since any open exposure counts against the rules.
    """

    mode = ExecutionMode.LIVE

    def __init__(self, transport: Mt5Transport, monitor: HeartbeatMonitor | None = None, magic: int = 260926,
                 deviation: int = 20, kill_guard: KillGuard | None = None, account_wide_kill: bool = True) -> None:
        self.name = "mt5"
        self.transport = transport
        self.monitor = monitor
        self.magic = magic
        self.deviation = deviation
        self._guard = kill_guard
        self._account_wide = account_wide_kill
        self._lock = threading.RLock()
        self._connected = False
        self.last_payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------------------
    def connect(self) -> dict[str, Any]:
        """Verify the account/terminal allow automated trading; start the heartbeat."""
        info = self.transport.account_info()
        if not info.get("trade_allowed"):
            raise PermissionError("MT5 account does not allow trading (investor password?)")
        if not info.get("trade_expert", True):
            raise PermissionError("MT5 terminal has Algo Trading disabled")
        self._connected = True
        if self.monitor is not None:
            self.monitor.beat()
            self.monitor.start()
        return info

    def close(self) -> None:
        """Stop the heartbeat thread."""
        if self.monitor is not None:
            self.monitor.stop()

    def _require(self) -> None:
        if not self._connected:
            raise OrderRejected("MT5 adapter not connected - call connect() first")

    @property
    def link_healthy(self) -> bool:
        """True without a monitor; otherwise the monitor's verdict."""
        return self.monitor is None or self.monitor.healthy()

    # ------------------------------------------------------------------------------
    def place_bracket(self, order: BracketOrder) -> BracketResult:
        """Send entry+SL+TP as one ``SEND_BRACKET`` (a single atomic MT5 deal request)."""
        if self._guard is not None and self._guard.tripped:
            raise KillSwitchEngaged("kill switch engaged - order refused")
        self._require()
        if order.entry_type != "market":
            raise OrderRejected("MT5 adapter sends market brackets only")
        if not self.link_healthy:
            state = self.monitor.state.value if self.monitor else "?"
            return BracketResult(False, order.client_id, message=f"MT5 link not healthy ({state}) - order refused")
        with self._lock:
            sym = self.transport.symbol_info(order.symbol)
            step, vmin = float(sym.get("volume_step", 0.01)), float(sym.get("volume_min", 0.01))
            vmax = float(sym.get("volume_max", math.inf) or math.inf)
            volume = min(_round_step(order.units, step), vmax)
            if volume < vmin:
                return BracketResult(False, order.client_id, message=f"volume {volume} < min {vmin}")
            digits = int(sym.get("digits", 5))
            point = float(sym.get("point", 10 ** -digits))
            quote = sym.get("ask") if order.side is Side.LONG else sym.get("bid")
            ref = float(quote or order.entry_price)
            min_dist = float(sym.get("trade_stops_level", 0)) * point
            if abs(ref - order.stop_loss) < min_dist or abs(order.take_profit - ref) < min_dist:
                return BracketResult(False, order.client_id, message=f"SL/TP inside stops level ({min_dist})")
            try:
                fields = build_bracket_payload(order, volume, digits, self.magic, self.deviation)
            except PayloadError as exc:
                raise OrderRejected(str(exc)) from exc
            self.last_payload = fields
            try:
                res = self.transport.send_bracket(fields)
            except Mt5Timeout:
                return self._reconcile(order, fields["client_id"])
            except Mt5Error as exc:
                return BracketResult(False, order.client_id, message=f"EA error: {exc}")
            ok = int(res.get("retcode", 0)) in RETCODE_OK
            return BracketResult(ok, order.client_id, res.get("price"),
                                 tuple(str(res[k]) for k in ("order", "deal") if res.get(k)),
                                 f"retcode {res.get('retcode')}: {res.get('comment', '')}", atomic=True)

    def _reconcile(self, order: BracketOrder, client_id: str) -> BracketResult:
        """Resolve a timed-out SEND_BRACKET by looking for its client_id among positions."""
        try:
            for p in self.transport.positions(self.magic):
                if sanitize_comment(str(p.get("comment", ""))) == client_id:
                    return BracketResult(True, order.client_id, float(p.get("price_open") or 0) or None,
                                         (str(p.get("ticket")),), "timeout; reconciled: position found", atomic=True)
        except Mt5Error as exc:
            return BracketResult(False, order.client_id,
                                 message=f"timeout; reconciliation failed ({exc}) - OUTCOME UNKNOWN")
        return BracketResult(False, order.client_id, message="timeout; reconciled: no position opened")

    def amend_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten the SL of this engine's position on ``symbol`` (never loosens)."""
        self._require()
        with self._lock:
            for p in self.transport.positions(self.magic):
                if p.get("symbol") != symbol:
                    continue
                side = Side.LONG if int(p.get("type", 0)) == 0 else Side.SHORT
                current = float(p.get("sl") or 0.0)
                if current and (new_stop - current) * side.sign <= 0:
                    return False
                tp = float(p.get("tp") or 0.0)
                if not tp:
                    return False  # never strip a missing TP further; bracket invariant broken
                res = self.transport.modify_sltp(int(p["ticket"]), symbol, new_stop, tp)
                return int(res.get("retcode", 0)) in RETCODE_OK
            return False

    def cancel_all_orders(self) -> int:
        """One ``CANCEL_ALL`` command. Raises if the EA reports failures."""
        res = self.transport.cancel_all(None if self._account_wide else self.magic)
        if res.get("failed"):
            raise Mt5Error(f"cancel failed for {res['failed']}")
        return int(res.get("cancelled", 0))

    def flatten_all(self) -> int:
        """One ``FLATTEN_ALL`` command executed inside the EA. Raises if anything stayed open."""
        res = self.transport.flatten_all(None if self._account_wide else self.magic, self.deviation)
        if res.get("failed"):
            raise Mt5Error(f"flatten failed for tickets {res['failed']}")
        return int(res.get("closed", 0))

    def positions(self) -> list[Position]:
        """This engine's open positions."""
        out: list[Position] = []
        for p in self.transport.positions(self.magic):
            side = Side.LONG if int(p.get("type", 0)) == 0 else Side.SHORT
            out.append(Position(p["symbol"], side, float(p["volume"]), float(p["price_open"]),
                                float(p.get("sl") or 0.0) or None, float(p.get("tp") or 0.0) or None,
                                str(p.get("comment", p.get("ticket", "")))))
        return out

    def equity(self) -> float:
        """Account equity."""
        return float(self.transport.account_info().get("equity", 0.0))
