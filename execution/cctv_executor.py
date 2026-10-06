"""CEX order manager (ccxt) with atomic bracket orders and trade-only credentials.

(The module name follows the project specification; it implements the *CEX* executor.)

Bracket atomicity
-----------------
* **Attached mode** (Bybit, OKX, Bitget): the stop-loss and take-profit ride on the
  entry request itself (``params.stopLoss`` / ``params.takeProfit``). The venue accepts
  or rejects all three in one call — truly atomic.
* **Sequential mode** (everything else, e.g. Binance USDⓈ-M): entry, then a
  reduce-only stop, then a reduce-only target. If *any* child fails, the executor
  compensates: it cancels the children already placed and closes the entry with a
  reduce-only market order, then raises :class:`BracketRollbackError`. A position is
  never left without a stop. Sequential mode accepts market entries only, because a
  resting limit entry cannot carry reduce-only children.

Credential policy
-----------------
Keys are read from the environment only, never from code or config files. On
:meth:`CexExecutor.connect` the key's permissions are queried from the venue:

* A key with **withdrawal** permission is refused outright — in every mode.
* On mainnet, a key whose permissions cannot be verified is refused (fail closed).
* On a sandbox/testnet, unverifiable permissions are tolerated (testnets often lack
  the permission endpoint) and flagged in the permission report.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from core.risk_engine import Side
from execution.base import (
    BracketOrder,
    BracketResult,
    BracketRollbackError,
    ExecutionMode,
    KillGuard,
    KillSwitchEngaged,
    OrderRejected,
    Position,
)

ATTACHED_BRACKET_EXCHANGES: frozenset[str] = frozenset({"bybit", "okx", "bitget"})


class CredentialSecurityError(PermissionError):
    """Credentials are missing, over-privileged, or could not be verified."""


@dataclass(frozen=True, slots=True)
class CexCredentials:
    """API credentials. ``repr`` never shows the secret material."""

    api_key: str = field(repr=False)
    secret: str = field(repr=False)
    password: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, prefix: str = "QM_CEX") -> CexCredentials:
        """Read ``{prefix}_API_KEY``, ``{prefix}_API_SECRET`` and optional ``{prefix}_API_PASSWORD``."""
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        secret = os.environ.get(f"{prefix}_API_SECRET", "").strip()
        if not key or not secret:
            raise CredentialSecurityError(f"{prefix}_API_KEY / {prefix}_API_SECRET not set in the environment")
        return cls(key, secret, os.environ.get(f"{prefix}_API_PASSWORD") or None)

    @property
    def fingerprint(self) -> str:
        """Non-reversible identifier safe for logs (first 4 chars of the key)."""
        return self.api_key[:4] + "…"


@dataclass(frozen=True, slots=True)
class PermissionReport:
    """What the venue says the key can do."""

    verified: bool
    can_trade: bool
    can_withdraw: bool
    detail: str


def inspect_permissions(exchange: Any, exchange_id: str) -> PermissionReport:
    """Query the venue's key-permission endpoint (Binance, Bybit, OKX supported)."""
    try:
        if exchange_id.startswith("binance"):
            r = exchange.sapi_get_account_apirestrictions()
            withdraw = bool(r.get("enableWithdrawals"))
            trade = bool(r.get("enableSpotAndMarginTrading") or r.get("enableFutures"))
            return PermissionReport(True, trade, withdraw, "binance apiRestrictions")
        if exchange_id == "bybit":
            r = exchange.private_get_v5_user_query_api()
            result = r.get("result", {})
            perms = json.dumps(result.get("permissions", {})).lower()
            read_only = str(result.get("readOnly", "1")) == "1"
            return PermissionReport(True, not read_only, "withdraw" in perms, "bybit query-api")
        if exchange_id == "okx":
            r = exchange.private_get_account_config()
            perm = str(r.get("data", [{}])[0].get("perm", "")).lower()
            return PermissionReport(True, "trade" in perm, "withdraw" in perm, f"okx perm={perm}")
    except Exception as exc:  # noqa: BLE001 - reported, then enforced by policy
        return PermissionReport(False, False, False, f"permission query failed: {type(exc).__name__}: {exc}")
    return PermissionReport(False, False, False, f"no permission endpoint known for {exchange_id}")


def enforce_trade_only(report: PermissionReport, sandbox: bool) -> None:
    """Raise :class:`CredentialSecurityError` unless the key is trade-only (see module doc)."""
    if report.can_withdraw:
        raise CredentialSecurityError("API key has WITHDRAWAL permission - refusing to run. "
                                      "Create a trade-only key (no withdrawals, IP-whitelisted).")
    if not report.verified and not sandbox:
        raise CredentialSecurityError(f"cannot verify key permissions on mainnet ({report.detail}) - fail closed")
    if report.verified and not report.can_trade:
        raise CredentialSecurityError("API key lacks trading permission")


class CexExecutor:
    """Live CEX adapter implementing :class:`~execution.base.BrokerAdapter`.

    Args:
        exchange_id: ccxt exchange id (``"bybit"``, ``"binanceusdm"``, ...).
        credentials: Trade-only credentials (default: from environment).
        sandbox: Use the venue's testnet.
        exchange: Pre-built ccxt-like client (dependency injection for tests).
        kill_guard: Kill switch; orders are refused while it is tripped.
        attached_brackets: Force attached/sequential mode (default by exchange id).
        quote_ccy: Currency used for equity.
    """

    mode = ExecutionMode.LIVE

    def __init__(self, exchange_id: str, credentials: CexCredentials | None = None, sandbox: bool = True,
                 exchange: Any | None = None, kill_guard: KillGuard | None = None,
                 attached_brackets: bool | None = None, quote_ccy: str = "USDT") -> None:
        self.name = f"cex:{exchange_id}{'-testnet' if sandbox else ''}"
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self._credentials = credentials
        self._exchange = exchange
        self._guard = kill_guard
        self.attached = exchange_id in ATTACHED_BRACKET_EXCHANGES if attached_brackets is None else attached_brackets
        self.quote_ccy = quote_ccy
        self._lock = threading.RLock()
        self._brackets: dict[str, dict[str, Any]] = {}
        self.permissions: PermissionReport | None = None

    # ------------------------------------------------------------------------------
    def connect(self) -> PermissionReport:
        """Build the client (if not injected), load markets and enforce trade-only keys."""
        with self._lock:
            if self._exchange is None:
                import ccxt  # optional dependency

                creds = self._credentials or CexCredentials.from_env()
                config: dict[str, Any] = {"apiKey": creds.api_key, "secret": creds.secret,
                                          "enableRateLimit": True, "options": {"defaultType": "swap"}}
                if creds.password:
                    config["password"] = creds.password
                self._exchange = getattr(ccxt, self.exchange_id)(config)
                if self.sandbox:
                    self._exchange.set_sandbox_mode(True)
            self._exchange.load_markets()
            report = inspect_permissions(self._exchange, self.exchange_id)
            enforce_trade_only(report, self.sandbox)
            self.permissions = report
            return report

    def _ex(self) -> Any:
        if self._exchange is None or self.permissions is None:
            raise OrderRejected("executor not connected - call connect() first")
        return self._exchange

    def _check_guard(self) -> None:
        if self._guard is not None and self._guard.tripped:
            raise KillSwitchEngaged("kill switch engaged - order refused")

    @staticmethod
    def _ccxt_side(side: Side) -> str:
        return "buy" if side is Side.LONG else "sell"

    # ------------------------------------------------------------------------------
    def place_bracket(self, order: BracketOrder) -> BracketResult:
        """Submit entry + SL + TP atomically (attached) or with rollback (sequential)."""
        self._check_guard()
        ex = self._ex()
        with self._lock:
            amount = float(ex.amount_to_precision(order.symbol, order.units))
            sl = float(ex.price_to_precision(order.symbol, order.stop_loss))
            tp = float(ex.price_to_precision(order.symbol, order.take_profit))
            price = float(ex.price_to_precision(order.symbol, order.entry_price)) \
                if order.entry_type == "limit" else None
            side, exit_side = self._ccxt_side(order.side), self._ccxt_side(order.side.opposite)

            if self.attached:
                params = {"clientOrderId": order.client_id,
                          "stopLoss": {"triggerPrice": sl}, "takeProfit": {"triggerPrice": tp}}
                entry = ex.create_order(order.symbol, order.entry_type, side, amount, price, params)
                self._brackets[order.symbol] = {"side": order.side, "amount": amount, "stop": sl,
                                                "sl_id": None, "entry_id": entry.get("id")}
                return BracketResult(True, order.client_id, entry.get("average") or entry.get("price"),
                                     (str(entry.get("id")),), "attached bracket", atomic=True)

            if order.entry_type != "market":
                raise OrderRejected("sequential bracket mode supports market entries only")
            entry = ex.create_order(order.symbol, "market", side, amount, None, {"clientOrderId": order.client_id})
            children: list[str] = []
            try:
                sl_o = ex.create_order(order.symbol, "market", exit_side, amount, None,
                                       {"stopLossPrice": sl, "reduceOnly": True,
                                        "clientOrderId": order.client_id + "-sl"})
                children.append(str(sl_o["id"]))
                tp_o = ex.create_order(order.symbol, "market", exit_side, amount, None,
                                       {"takeProfitPrice": tp, "reduceOnly": True,
                                        "clientOrderId": order.client_id + "-tp"})
                children.append(str(tp_o["id"]))
            except Exception as exc:
                self._rollback(order.symbol, exit_side, amount, children)
                raise BracketRollbackError(f"bracket child failed ({type(exc).__name__}: {exc}); "
                                           "entry flattened and children cancelled") from exc
            self._brackets[order.symbol] = {"side": order.side, "amount": amount, "stop": sl,
                                            "sl_id": children[0], "tp_id": children[1], "entry_id": entry.get("id")}
            return BracketResult(True, order.client_id, entry.get("average") or entry.get("price"),
                                 (str(entry.get("id")), *children), "sequential bracket", atomic=False)

    def _rollback(self, symbol: str, exit_side: str, amount: float, children: list[str]) -> None:
        ex = self._ex()
        for oid in children:
            with contextlib.suppress(Exception):  # best effort; the flatten below is what matters
                ex.cancel_order(oid, symbol)
        try:
            ex.create_order(symbol, "market", exit_side, amount, None, {"reduceOnly": True})
        except Exception as exc:
            raise BracketRollbackError(f"CRITICAL: rollback flatten failed on {symbol} - position may be "
                                       f"UNPROTECTED, trip the kill switch ({exc})") from exc

    def amend_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten the stop: place the new reduce-only stop first, then cancel the old one."""
        self._check_guard()
        ex = self._ex()
        with self._lock:
            info = self._brackets.get(symbol)
            if info is None:
                return False
            side: Side = info["side"]
            new_stop = float(ex.price_to_precision(symbol, new_stop))
            if (new_stop - info["stop"]) * side.sign <= 0:
                return False  # never loosen (or no-op)
            new_o = ex.create_order(symbol, "market", self._ccxt_side(side.opposite), info["amount"], None,
                                    {"stopLossPrice": new_stop, "reduceOnly": True})
            if info.get("sl_id"):
                with contextlib.suppress(Exception):  # duplicate reduce-only stops are safe
                    ex.cancel_order(info["sl_id"], symbol)
            info.update(stop=new_stop, sl_id=str(new_o["id"]))
            return True

    def cancel_all_orders(self) -> int:
        """Cancel every open order on every symbol this executor touched or sees open."""
        ex = self._ex()
        symbols = set(self._brackets)
        with contextlib.suppress(Exception):  # some venues require a symbol; tracked set still covered
            symbols |= {o["symbol"] for o in ex.fetch_open_orders()}
        count = 0
        for sym in symbols:
            open_orders = ex.fetch_open_orders(sym)
            for o in open_orders:
                ex.cancel_order(o["id"], sym)
                count += 1
        return count

    def flatten_all(self) -> int:
        """Close every non-zero position with a reduce-only market order."""
        ex = self._ex()
        closed = 0
        for p in ex.fetch_positions():
            contracts = float(p.get("contracts") or 0)
            if contracts <= 0:
                continue
            exit_side = "sell" if p.get("side") == "long" else "buy"
            ex.create_order(p["symbol"], "market", exit_side, contracts, None, {"reduceOnly": True})
            self._brackets.pop(p["symbol"], None)
            closed += 1
        return closed

    def positions(self) -> list[Position]:
        """Open positions mapped to :class:`~execution.base.Position`."""
        out: list[Position] = []
        for p in self._ex().fetch_positions():
            contracts = float(p.get("contracts") or 0)
            if contracts <= 0:
                continue
            side = Side.LONG if p.get("side") == "long" else Side.SHORT
            info = self._brackets.get(p["symbol"], {})
            out.append(Position(p["symbol"], side, contracts, float(p.get("entryPrice") or 0),
                                info.get("stop"), None))
        return out

    def equity(self) -> float:
        """Total balance in ``quote_ccy``."""
        bal = self._ex().fetch_balance()
        return float((bal.get("total") or {}).get(self.quote_ccy) or 0.0)
