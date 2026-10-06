"""MetaTrader 5 bridge: wire protocol, transports, heartbeat and reconnect.

Architecture::

    agent/coordinator.py ──BracketOrder──► execution/mt5_adapter.py (BrokerAdapter)
                                                   │ build_bracket_payload()  (SL/TP mandatory)
                                                   ▼
                          ZmqMt5Transport  ──REQ tcp://127.0.0.1:5555──►  mql5/ZeroMQ_Listener.mq5 (REP)
                          HeartbeatMonitor ──PING every 1 s──────────────►        │
                          Mt5EventSubscriber ◄──SUB tcp://127.0.0.1:5556── (PUB: heartbeats, trade events)

Wire protocol (version :data:`PROTOCOL_VERSION`)
-----------------------------------------------
Requests are **flat** JSON objects — only string/number/bool/null values, no nesting —
so the EA can parse them with a small strict extractor instead of a JSON library::

    {"v": 1, "id": "<uuid>", "token": "<shared secret>", "action": "SEND_BRACKET",
     "symbol": "XAUUSD", "side": "BUY", "volume": 0.10, "sl": 1985.12, "tp": 2044.5,
     "deviation": 20, "magic": 260926, "client_id": "qm-...", "comment": "qm-..."}

Replies: ``{"ok": true, "id": "<same id>", "data": {...}}`` or
``{"ok": false, "id": "<same id>", "error": "..."}``. A reply whose ``id`` does not
match the request is rejected (a stale reply after a timeout can never be mistaken
for the current one).

Actions: see :data:`ACTIONS`. Only :data:`IDEMPOTENT_ACTIONS` are retried after a
timeout. ``SEND_BRACKET`` is never re-sent automatically; the EA also de-duplicates
by ``client_id``, and the adapter reconciles an unknown outcome by looking for the
``client_id`` among open positions.
"""
from __future__ import annotations

import contextlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from core.risk_engine import Side
from execution.base import BracketOrder

PROTOCOL_VERSION = 1
DEFAULT_REQ_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_PUB_ENDPOINT = "tcp://127.0.0.1:5556"

ACTIONS: frozenset[str] = frozenset({
    "PING", "ACCOUNT_INFO", "SYMBOL_INFO", "SEND_BRACKET", "MODIFY_SLTP",
    "POSITIONS", "PENDING_ORDERS", "CANCEL_ALL", "CLOSE_POSITION", "FLATTEN_ALL", "RATES",
})
IDEMPOTENT_ACTIONS: frozenset[str] = frozenset({
    "PING", "ACCOUNT_INFO", "SYMBOL_INFO", "POSITIONS", "PENDING_ORDERS",
    "CANCEL_ALL", "CLOSE_POSITION", "FLATTEN_ALL", "RATES",
})

#: MT5 ``ACCOUNT_TRADE_MODE`` values.
TRADE_MODE_DEMO = 0
TRADE_MODE_CONTEST = 1
TRADE_MODE_REAL = 2

#: Timeframe codes understood by the EA's ``RATES`` action.
TIMEFRAME_CODES: frozenset[str] = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1"})

RETCODE_PLACED = 10008
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
RETCODE_OK = frozenset({RETCODE_PLACED, RETCODE_DONE, RETCODE_DONE_PARTIAL})

_COMMENT_MAX = 31  # MT5 order comment limit
_SAFE_COMMENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")


class Mt5Error(RuntimeError):
    """Transport, protocol or terminal failure."""


class Mt5Timeout(Mt5Error):
    """The EA did not answer in time; the outcome of the request is unknown."""


class PayloadError(ValueError):
    """A request payload violates the protocol (e.g. a bracket without SL/TP)."""


# --------------------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------------------
def sanitize_comment(text: str) -> str:
    """Restrict to characters the EA's flat parser and MT5 comments handle safely."""
    return "".join(ch for ch in text if ch in _SAFE_COMMENT)[:_COMMENT_MAX]


def _check_flat(payload: dict[str, Any]) -> None:
    for k, v in payload.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise PayloadError(f"invalid key {k!r}")
        if v is not None and not isinstance(v, (str, int, float, bool)):
            raise PayloadError(f"field {k!r} must be a primitive, got {type(v).__name__}")
        if isinstance(v, float) and not math.isfinite(v):
            raise PayloadError(f"field {k!r} is not finite")
        if isinstance(v, str) and ('"' in v or "\\" in v):
            raise PayloadError(f"field {k!r} contains characters the EA parser rejects")


def make_request(action: str, token: str, **fields: Any) -> dict[str, Any]:
    """Build a flat, versioned request envelope."""
    if action not in ACTIONS:
        raise PayloadError(f"unknown action {action!r}")
    payload: dict[str, Any] = {"v": PROTOCOL_VERSION, "id": uuid.uuid4().hex, "token": token,
                               "action": action, **fields}
    _check_flat(payload)
    return payload


def encode(payload: dict[str, Any]) -> str:
    """Compact JSON; refuses NaN/inf."""
    _check_flat(payload)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def build_bracket_payload(order: BracketOrder, volume: float, digits: int, magic: int,
                          deviation: int) -> dict[str, Any]:
    """Fields of a ``SEND_BRACKET`` request, with SL/TP guaranteed present and valid.

    Raises :class:`PayloadError` if SL or TP is missing, zero, non-finite, or on the wrong
    side of the entry reference — a bracket without both levels can never be encoded.
    """
    sl, tp, ref = order.stop_loss, order.take_profit, order.entry_price
    for name, value in (("sl", sl), ("tp", tp), ("entry", ref)):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PayloadError(f"bracket {name} missing or invalid: {value!r}")
    sign = order.side.sign
    if (ref - sl) * sign <= 0 or (tp - ref) * sign <= 0:
        raise PayloadError(f"SL/TP on the wrong side for {order.side.value}: sl={sl} entry={ref} tp={tp}")
    if not (volume > 0 and math.isfinite(volume)):
        raise PayloadError(f"volume must be positive, got {volume}")
    return {
        "symbol": order.symbol,
        "side": "BUY" if order.side is Side.LONG else "SELL",
        "volume": round(volume, 8),
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": int(deviation),
        "magic": int(magic),
        "client_id": sanitize_comment(order.client_id),
        "comment": sanitize_comment(order.client_id),
    }


# --------------------------------------------------------------------------------------
# Transport protocol
# --------------------------------------------------------------------------------------
class Mt5Transport(Protocol):
    """Operations the adapter needs from a terminal connection."""

    def ping(self) -> dict[str, Any]: ...
    def account_info(self) -> dict[str, Any]: ...
    def symbol_info(self, symbol: str) -> dict[str, Any]: ...
    def send_bracket(self, fields: dict[str, Any]) -> dict[str, Any]: ...
    def modify_sltp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict[str, Any]: ...
    def positions(self, magic: int | None) -> list[dict[str, Any]]: ...
    def pending_orders(self, magic: int | None) -> list[dict[str, Any]]: ...
    def cancel_all(self, magic: int | None) -> dict[str, Any]: ...
    def close_position(self, ticket: int, deviation: int) -> dict[str, Any]: ...
    def flatten_all(self, magic: int | None, deviation: int) -> dict[str, Any]: ...
    def rates(self, symbol: str, timeframe: str, count: int) -> dict[str, Any]: ...
    def reconnect(self) -> None: ...


class SocketTimeout(Exception):
    """Raised by a socket wrapper when send/recv exceeds its timeout."""


class ReqSocket(Protocol):
    """Minimal REQ-socket surface (a pyzmq socket wrapper or a test fake)."""

    def send_string(self, data: str) -> None: ...
    def recv_string(self) -> str: ...
    def close(self) -> None: ...


class _PyZmqReq:
    """Wraps a pyzmq REQ socket, mapping ``zmq.Again`` to :class:`SocketTimeout`."""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        import zmq  # type: ignore[import-not-found]  # optional dependency

        self._zmq = zmq
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        for opt, val in ((zmq.RCVTIMEO, timeout_ms), (zmq.SNDTIMEO, timeout_ms), (zmq.LINGER, 0)):
            self._sock.setsockopt(opt, val)
        self._sock.connect(endpoint)

    def send_string(self, data: str) -> None:
        try:
            self._sock.send_string(data)
        except self._zmq.error.Again as exc:
            raise SocketTimeout("send timeout") from exc

    def recv_string(self) -> str:
        try:
            return str(self._sock.recv_string())
        except self._zmq.error.Again as exc:
            raise SocketTimeout("recv timeout") from exc

    def close(self) -> None:
        self._sock.close(linger=0)


SocketFactory = Callable[[], ReqSocket]


class ZmqMt5Transport:
    """REQ client for ``ZeroMQ_Listener.mq5`` with lazy-pirate reconnection.

    A REQ socket is unusable after a timeout (its send/recv state machine is stuck), so
    on every timeout the socket is closed and recreated. Idempotent actions are retried
    up to ``retries`` times; everything else raises :class:`Mt5Timeout` immediately so
    the caller can reconcile instead of risking a duplicate order.

    Args:
        endpoint: EA REP endpoint.
        token: Shared secret matching the EA's ``InpAuthToken``.
        timeout_ms: Per-request send/receive timeout.
        retries: Extra attempts for idempotent actions.
        socket_factory: Override for tests; defaults to a pyzmq REQ socket.
    """

    def __init__(self, endpoint: str = DEFAULT_REQ_ENDPOINT, token: str = "", timeout_ms: int = 2_000,
                 retries: int = 1, socket_factory: SocketFactory | None = None) -> None:
        self.endpoint = endpoint
        self._token = token
        self._retries = max(0, retries)
        self._factory: SocketFactory = socket_factory or (lambda: _PyZmqReq(endpoint, timeout_ms))
        self._lock = threading.Lock()
        self._sock: ReqSocket | None = None
        self.reconnects = 0

    # -- connection management ------------------------------------------------------
    def _socket(self) -> ReqSocket:
        if self._sock is None:
            self._sock = self._factory()
        return self._sock

    def _reset(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(Exception):  # a broken socket may fail to close; drop it anyway
                self._sock.close()
        self._sock = None

    def reconnect(self) -> None:
        """Drop and recreate the socket (used by the heartbeat monitor)."""
        with self._lock:
            self._reset()
            self._socket()
            self.reconnects += 1

    def close(self) -> None:
        """Release the socket."""
        with self._lock:
            self._reset()

    # -- request/reply --------------------------------------------------------------
    def request(self, action: str, **fields: Any) -> Any:
        """Send one request and return the reply's ``data`` (raises on error/timeout)."""
        attempts = 1 + (self._retries if action in IDEMPOTENT_ACTIONS else 0)
        last: Exception | None = None
        for _ in range(attempts):
            payload = make_request(action, self._token, **fields)
            with self._lock:
                try:
                    sock = self._socket()
                    sock.send_string(encode(payload))
                    raw = sock.recv_string()
                except SocketTimeout as exc:
                    self._reset()
                    last = exc
                    continue
            try:
                reply = json.loads(raw)
            except ValueError as exc:
                with self._lock:
                    self._reset()
                raise Mt5Error(f"malformed reply to {action}: {raw[:120]!r}") from exc
            if reply.get("id") != payload["id"]:
                with self._lock:
                    self._reset()
                raise Mt5Error(f"reply id mismatch for {action} (stale reply discarded)")
            if not reply.get("ok"):
                raise Mt5Error(f"EA rejected {action}: {reply.get('error')}")
            return reply.get("data")
        raise Mt5Timeout(f"EA did not answer {action} after {attempts} attempt(s): {last}")

    # -- typed actions ----------------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        return dict(self.request("PING"))

    def account_info(self) -> dict[str, Any]:
        return dict(self.request("ACCOUNT_INFO"))

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        return dict(self.request("SYMBOL_INFO", symbol=symbol))

    def send_bracket(self, fields: dict[str, Any]) -> dict[str, Any]:
        return dict(self.request("SEND_BRACKET", **fields))

    def modify_sltp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict[str, Any]:
        return dict(self.request("MODIFY_SLTP", ticket=ticket, symbol=symbol, sl=sl, tp=tp))

    def positions(self, magic: int | None) -> list[dict[str, Any]]:
        return list(self.request("POSITIONS", magic=magic if magic is not None else -1))

    def pending_orders(self, magic: int | None) -> list[dict[str, Any]]:
        return list(self.request("PENDING_ORDERS", magic=magic if magic is not None else -1))

    def cancel_all(self, magic: int | None) -> dict[str, Any]:
        return dict(self.request("CANCEL_ALL", magic=magic if magic is not None else -1))

    def close_position(self, ticket: int, deviation: int) -> dict[str, Any]:
        return dict(self.request("CLOSE_POSITION", ticket=ticket, deviation=deviation))

    def flatten_all(self, magic: int | None, deviation: int) -> dict[str, Any]:
        return dict(self.request("FLATTEN_ALL", magic=magic if magic is not None else -1, deviation=deviation))

    def rates(self, symbol: str, timeframe: str, count: int) -> dict[str, Any]:
        """Closed bars: ``{"server_offset": s, "bars": [[t_server, o, h, l, c, v], ...]}``."""
        if timeframe not in TIMEFRAME_CODES:
            raise PayloadError(f"timeframe must be one of {sorted(TIMEFRAME_CODES)}")
        return dict(self.request("RATES", symbol=symbol, timeframe=timeframe, count=int(count)))


class NativeMt5Transport:
    """Same interface backed by the ``MetaTrader5`` package (same machine, Windows)."""

    def __init__(self, login: int | None = None, password: str | None = None, server: str | None = None,
                 path: str | None = None) -> None:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]

        self.mt5 = mt5
        self._init_args = ({k: v for k, v in {"login": login, "password": password, "server": server}.items()
                            if v}, path)
        self.reconnect()

    def reconnect(self) -> None:
        kwargs, path = self._init_args
        self.mt5.shutdown()
        ok = self.mt5.initialize(path, **kwargs) if path else self.mt5.initialize(**kwargs)
        if not ok:
            raise Mt5Error(f"MT5 initialize failed: {self.mt5.last_error()}")

    @staticmethod
    def _d(obj: Any) -> dict[str, Any]:
        return obj._asdict() if obj is not None else {}

    def _fill_mode(self, symbol: str) -> int:
        mode = int(self._d(self.mt5.symbol_info(symbol)).get("filling_mode", 0))
        return self.mt5.ORDER_FILLING_FOK if mode & 1 else self.mt5.ORDER_FILLING_IOC

    def ping(self) -> dict[str, Any]:
        term = self._d(self.mt5.terminal_info())
        if not term.get("connected"):
            raise Mt5Error("terminal not connected to trade server")
        acc = self._d(self.mt5.account_info())
        return {"trade_allowed": bool(term.get("trade_allowed")) and bool(acc.get("trade_allowed")),
                "equity": acc.get("equity"), "server_time": int(time.time())}

    def account_info(self) -> dict[str, Any]:
        info = self._d(self.mt5.account_info())
        info["trade_expert"] = bool(self._d(self.mt5.terminal_info()).get("trade_allowed"))
        return info

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        if not self.mt5.symbol_select(symbol, True):
            raise Mt5Error(f"symbol {symbol} unavailable")
        info = self._d(self.mt5.symbol_info(symbol))
        tick = self._d(self.mt5.symbol_info_tick(symbol))
        info.update(bid=tick.get("bid"), ask=tick.get("ask"))
        return info

    def send_bracket(self, fields: dict[str, Any]) -> dict[str, Any]:
        mt5, sym = self.mt5, fields["symbol"]
        tick = mt5.symbol_info_tick(sym)
        buy = fields["side"] == "BUY"
        return self._d(mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": fields["volume"],
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, "price": tick.ask if buy else tick.bid,
            "sl": fields["sl"], "tp": fields["tp"], "deviation": fields["deviation"], "magic": fields["magic"],
            "comment": fields["comment"], "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._fill_mode(sym)}))

    def modify_sltp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict[str, Any]:
        return self._d(self.mt5.order_send({"action": self.mt5.TRADE_ACTION_SLTP, "position": ticket,
                                            "symbol": symbol, "sl": sl, "tp": tp}))

    def positions(self, magic: int | None) -> list[dict[str, Any]]:
        rows = [self._d(p) for p in (self.mt5.positions_get() or ())]
        return [r for r in rows if magic is None or r.get("magic") == magic]

    def pending_orders(self, magic: int | None) -> list[dict[str, Any]]:
        rows = [self._d(o) for o in (self.mt5.orders_get() or ())]
        return [r for r in rows if magic is None or r.get("magic") == magic]

    def cancel_all(self, magic: int | None) -> dict[str, Any]:
        done, failed = 0, []
        for o in self.pending_orders(magic):
            res = self._d(self.mt5.order_send({"action": self.mt5.TRADE_ACTION_REMOVE, "order": o["ticket"]}))
            done, failed = (done + 1, failed) if res.get("retcode") in RETCODE_OK else (done, [*failed, o["ticket"]])
        return {"cancelled": done, "failed": failed}

    def close_position(self, ticket: int, deviation: int) -> dict[str, Any]:
        mt5 = self.mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {"retcode": RETCODE_DONE, "comment": "already closed"}
        p = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        is_buy = p.type == mt5.POSITION_TYPE_BUY
        return self._d(mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "position": ticket, "symbol": p.symbol, "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY, "price": tick.bid if is_buy else tick.ask,
            "deviation": deviation, "magic": p.magic, "comment": "qm-flatten",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._fill_mode(p.symbol)}))

    def rates(self, symbol: str, timeframe: str, count: int) -> dict[str, Any]:
        tf = getattr(self.mt5, f"TIMEFRAME_{timeframe}")
        rows = self.mt5.copy_rates_from_pos(symbol, tf, 1, count)  # shift 1: closed bars only
        if rows is None:
            raise Mt5Error(f"copy_rates_from_pos failed: {self.mt5.last_error()}")
        tick = self.mt5.symbol_info_tick(symbol)
        offset = round((int(tick.time) - time.time()) / 900) * 900 if tick is not None else 0
        return {"server_offset": offset,
                "bars": [[int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                          float(r["close"]), int(r["tick_volume"])] for r in rows]}

    def flatten_all(self, magic: int | None, deviation: int) -> dict[str, Any]:
        cancelled = self.cancel_all(magic)
        closed, failed = 0, list(cancelled["failed"])
        for p in self.positions(magic):
            res = self.close_position(int(p["ticket"]), deviation)
            closed, failed = (closed + 1, failed) if res.get("retcode") in RETCODE_OK else (closed, [*failed, p["ticket"]])
        return {"closed": closed, "cancelled": cancelled["cancelled"], "failed": failed}


# --------------------------------------------------------------------------------------
# Heartbeat monitoring & reconnect
# --------------------------------------------------------------------------------------
class LinkState(str, Enum):
    """Health of the Python <-> EA link."""

    UNKNOWN = "unknown"
    UP = "up"
    DEGRADED = "degraded"  # missed beats below the threshold, or trading disabled terminal-side
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class HeartbeatConfig:
    """Heartbeat timing. The link is DOWN after ``max_missed`` consecutive failed pings."""

    interval_s: float = 1.0
    max_missed: int = 3
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 10.0


class HeartbeatMonitor:
    """PINGs the EA, tracks link health, and reconnects with exponential backoff.

    :meth:`beat` performs one step and is what tests drive with a fake clock;
    :meth:`start` runs it on a daemon thread every ``interval_s``.
    """

    def __init__(self, transport: Mt5Transport, config: HeartbeatConfig | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 on_state_change: Callable[[LinkState, LinkState], Any] | None = None) -> None:
        self.transport = transport
        self.config = config or HeartbeatConfig()
        self.clock = clock
        self.on_state_change = on_state_change
        self.state = LinkState.UNKNOWN
        self.missed = 0
        self.last_ok: float | None = None
        self.last_info: dict[str, Any] = {}
        self._backoff = self.config.backoff_initial_s
        self._next_reconnect = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _set(self, new: LinkState) -> None:
        old, self.state = self.state, new
        if old is not new and self.on_state_change is not None:
            self.on_state_change(old, new)

    def beat(self) -> LinkState:
        """Ping once and update state; reconnect when DOWN and the backoff has elapsed."""
        with self._lock:
            now = self.clock()
            try:
                info = self.transport.ping()
            except Mt5Error:
                self.missed += 1
                if self.missed >= self.config.max_missed:
                    self._set(LinkState.DOWN)
                    if now >= self._next_reconnect:
                        try:
                            self.transport.reconnect()
                        except Mt5Error:
                            pass
                        self._next_reconnect = now + self._backoff
                        self._backoff = min(self._backoff * 2, self.config.backoff_max_s)
                else:
                    self._set(LinkState.DEGRADED)
                return self.state
            self.missed = 0
            self.last_ok = now
            self.last_info = info
            self._backoff = self.config.backoff_initial_s
            self._next_reconnect = 0.0
            self._set(LinkState.UP if info.get("trade_allowed", True) else LinkState.DEGRADED)
            return self.state

    def healthy(self) -> bool:
        """UP and the last successful ping is recent enough."""
        if self.state is not LinkState.UP or self.last_ok is None:
            return False
        return self.clock() - self.last_ok <= self.config.interval_s * self.config.max_missed

    def start(self) -> None:
        """Run :meth:`beat` on a daemon thread."""
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self.beat()
                self._stop.wait(self.config.interval_s)

        self._thread = threading.Thread(target=loop, name="mt5-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class Mt5EventSubscriber:
    """Optional SUB listener for the EA's PUB stream (heartbeats, trade transactions).

    Each message is one JSON object with a ``type`` field; ``handler`` is called for each.
    """

    def __init__(self, handler: Callable[[dict[str, Any]], Any], endpoint: str = DEFAULT_PUB_ENDPOINT) -> None:
        self.endpoint = endpoint
        self.handler = handler
        self.last_event_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Connect and consume on a daemon thread (requires ``pyzmq``)."""
        import zmq  # type: ignore[import-not-found]

        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.endpoint)

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    msg = json.loads(sock.recv_string())
                except zmq.error.Again:
                    continue
                except ValueError:
                    continue
                self.last_event_at = time.monotonic()
                self.handler(msg)
            sock.close(linger=0)

        self._thread = threading.Thread(target=loop, name="mt5-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop consuming."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
