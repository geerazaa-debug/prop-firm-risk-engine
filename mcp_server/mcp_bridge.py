"""MCP bridge: exposes the orchestrator's tools to Claude.

Tools:
  * ``fetch_ohlcv`` — synthetic, CSV, or public ccxt market data.
  * ``backtest_strategy`` — full-stack backtest, optionally with a critic verdict.
  * ``get_risk_metrics`` — live risk posture (DLL/drawdown headroom, exposure, kill switch).
  * ``execute_order`` — risk-sized bracket order. **Paper by default.**
  * ``trip_kill_switch`` — emergency flatten.

Safety model for ``execute_order``:
  * The caller never chooses the size: the risk engine computes it from equity,
    stop distance, leverage and remaining headroom. The caller can only pick side,
    stop and target, and a bracket is mandatory.
  * Live execution requires *all* of: ``mode="live"``, ``confirm=True``, the environment
    variable ``QM_LIVE_TRADING=ENABLED`` set by the human operator, and a live broker
    attached to the service. Anything less is refused.
  * Nothing trades while the kill switch is tripped.

Transports: with the ``mcp`` package installed the server runs on its stdio server class
(``MCPServer`` in mcp >= 2, ``FastMCP`` in mcp 1.x or the standalone ``fastmcp`` package).
``--jsonrpc`` runs a minimal dependency-free line-delimited JSON-RPC 2.0 loop instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.coordinator import TradingCoordinator
from agent.critic import QCritic
from agent.generator import StrategyCandidate, candidate_id
from core.audit import AuditTrail
from core.backtest import BacktestConfig, run_backtest
from core.compliance import ComplianceGate, WeekendGuard
from core.instruments import default_asset
from core.market_data import (
    Bar,
    fetch_ccxt_public,
    load_csv,
    synthetic_ohlcv,
)
from core.news_filter import NewsFilter, load_calendar
from core.risk_engine import (
    AccountState,
    AssetSpec,
    BracketError,
    Side,
    hard_bracket,
    mark_to_market,
    size_position,
)
from core.strategy_brain import (
    STRATEGY_REGISTRY,
    TrendBreakoutStrategy,
    build_strategy,
)
from execution.base import (
    BracketOrder,
    BrokerAdapter,
    KillSwitchEngaged,
    OrderRejected,
    PaperBroker,
)
from execution.kill_switch import KillSwitch

DEFAULT_STATE_DIR = Path(os.environ.get("QM_STATE_DIR", ROOT / "state"))
MAX_BARS = 5_000
LIVE_ENV_FLAG = "QM_LIVE_TRADING"
NEWS_CALENDAR_ENV = "QM_NEWS_CALENDAR"


def default_compliance() -> ComplianceGate:
    """Weekend guard always; news filter when ``QM_NEWS_CALENDAR`` points to a calendar file."""
    path = os.environ.get(NEWS_CALENDAR_ENV)
    news = NewsFilter(load_calendar(path)) if path else None
    return ComplianceGate(WeekendGuard(), news)


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses/enums/datetimes/non-finite floats into JSON-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


class OrchestratorService:
    """Tool implementations, independent of any MCP transport (unit-testable).

    Args:
        state_dir: Directory for the audit log and kill-switch sentinel.
        symbol: Default symbol.
        asset: Contract spec for sizing (default: XAUUSD-like CFD, 100 oz, 1:30).
        starting_cash: Paper account size.
        live_broker: Optional live adapter (CEX/MT5). None = paper only.
    """

    def __init__(self, state_dir: str | Path | None = DEFAULT_STATE_DIR, symbol: str = "XAUUSD",
                 asset: AssetSpec | None = None, starting_cash: float = 100_000.0,
                 live_broker: BrokerAdapter | None = None, compliance: ComplianceGate | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        sd = Path(state_dir) if state_dir else None
        self.clock = clock
        self.compliance = compliance if compliance is not None else default_compliance()
        self.symbol = symbol
        self.asset = asset or default_asset(symbol)
        self.audit = AuditTrail(sd / "audit.jsonl" if sd else None)
        self.kill_switch = KillSwitch(sentinel_path=sd / "KILL_SWITCH" if sd else None, audit=self.audit.record)
        self.paper = PaperBroker(starting_cash, unit_value=self.asset.contract_size, guard=self.kill_switch)
        self.live_broker = live_broker
        if live_broker is not None:
            self.kill_switch.register(live_broker)
        self.coordinator = TradingCoordinator(symbol, self.asset, self.paper, self.kill_switch,
                                              [TrendBreakoutStrategy()], audit=self.audit,
                                              compliance=self.compliance)
        self._bars: dict[str, list[Bar]] = {}

    # ---------------------------------------------------------------------------------
    def _load(self, symbol: str, timeframe: str, limit: int, source: str, seed: int,
              exchange: str, path: str | None) -> list[Bar]:
        limit = max(1, min(int(limit), MAX_BARS))
        if source == "synthetic":
            bars = synthetic_ohlcv(limit, seed=seed)
        elif source == "csv":
            if not path:
                raise ValueError("source='csv' requires path")
            bars = load_csv(path)[-limit:]
        elif source == "ccxt":
            bars = fetch_ccxt_public(exchange, symbol, timeframe, limit)
        else:
            raise ValueError("source must be 'synthetic', 'csv' or 'ccxt'")
        self._bars[symbol] = bars
        if bars:
            self.paper.mark(symbol, bars[-1].close)
        return bars

    def fetch_ohlcv(self, symbol: str = "XAUUSD", timeframe: str = "1d", limit: int = 300,
                    source: str = "synthetic", seed: int = 7, exchange: str = "binance",
                    path: str | None = None, include_bars: bool = False) -> dict[str, Any]:
        """Load OHLCV bars and cache them for later tools. Returns a summary (+ bars on request)."""
        bars = self._load(symbol, timeframe, limit, source, seed, exchange, path)
        out: dict[str, Any] = {"symbol": symbol, "source": source, "timeframe": timeframe, "count": len(bars)}
        if bars:
            out.update(first=bars[0].ts.isoformat(), last=bars[-1].ts.isoformat(), last_close=bars[-1].close)
        if include_bars:
            out["bars"] = [b.to_dict() for b in bars[-500:]]
        return out

    def backtest_strategy(self, strategy: str = "trend_breakout", params: dict[str, Any] | None = None,
                          symbol: str = "XAUUSD", bars: int = 1_500, source: str = "synthetic", seed: int = 7,
                          use_regime_filter: bool = True, critic: bool = True) -> dict[str, Any]:
        """Backtest one strategy through the full risk stack; optionally run the critic."""
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"unknown strategy {strategy!r}; known {sorted(STRATEGY_REGISTRY)}")
        data = self._bars.get(symbol) if source == "cache" else self._load(symbol, "1d", bars, source, seed,
                                                                           "binance", None)
        if not data:
            raise ValueError("no bars available")
        cfg = BacktestConfig(use_regime_filter=use_regime_filter)
        res = run_backtest(data, build_strategy(strategy, params), cfg, symbol)
        out: dict[str, Any] = {"strategy": strategy, "params": params or {}, "bars": len(data),
                               "metrics": res.metrics.to_dict(), "breached": res.breached,
                               "risk_events": list(res.risk_events),
                               "signals_blocked_by_regime": res.signals_blocked_by_regime}
        if critic:
            cand = StrategyCandidate(candidate_id(strategy, params or {}), strategy, params or {}, "ad-hoc", 1)
            out["critic"] = QCritic().review(cand, data, cfg, trials=1, symbol=symbol).to_dict()
        self.audit.record("mcp_backtest", {"strategy": strategy, "params": params or {},
                                           "metrics": out["metrics"]})
        return out

    def get_risk_metrics(self) -> dict[str, Any]:
        """Current risk posture of the paper account (plus limits)."""
        self._refresh_state()
        snap = self.coordinator.risk_snapshot()
        snap["limits"] = to_jsonable(self.coordinator.limits)
        snap["live_trading_enabled"] = os.environ.get(LIVE_ENV_FLAG) == "ENABLED" and self.live_broker is not None
        snap["compliance"] = self.compliance.evaluate(self.clock(), self.symbol).to_dict()
        return to_jsonable(snap)

    def _refresh_state(self) -> AccountState:
        coord = self.coordinator
        now = self.clock()
        if coord.state is None:
            coord.state = AccountState.open(self.paper.equity(), now, coord.limits)
        ev = mark_to_market(coord.state, coord.broker.equity(), now, coord.limits)
        coord.state = ev.state
        return coord.state

    def execute_order(self, side: str, stop_loss: float, take_profit: float, symbol: str | None = None,
                      mode: str = "paper", confirm: bool = False) -> dict[str, Any]:
        """Submit a risk-sized bracket order. Size is always computed by the risk engine."""
        symbol = symbol or self.symbol
        if self.kill_switch.tripped:
            return {"accepted": False, "reason": "kill switch engaged"}
        if mode == "live":
            if os.environ.get(LIVE_ENV_FLAG) != "ENABLED":
                return {"accepted": False, "reason": f"live trading disabled (operator must set {LIVE_ENV_FLAG}=ENABLED)"}
            if not confirm:
                return {"accepted": False, "reason": "live order requires confirm=true"}
            if self.live_broker is None:
                return {"accepted": False, "reason": "no live broker attached to this server"}
            broker: BrokerAdapter = self.live_broker
        elif mode == "paper":
            broker = self.paper
        else:
            return {"accepted": False, "reason": "mode must be 'paper' or 'live'"}

        bars = self._bars.get(symbol)
        if not bars:
            return {"accepted": False, "reason": f"no market data for {symbol}; call fetch_ohlcv first"}
        entry = bars[-1].close
        try:
            trade_side = Side(side.lower())
        except ValueError:
            return {"accepted": False, "reason": "side must be 'long' or 'short'"}

        gate = self.compliance.evaluate(self.clock(), symbol)
        if not gate.allow_new_orders:
            self.audit.record("mcp_order_vetoed", {"symbol": symbol, "reasons": list(gate.reasons)})
            return {"accepted": False, "reason": "compliance: " + "; ".join(gate.reasons)}
        try:
            levels = hard_bracket(trade_side, entry, stop_loss, take_profit, self.asset)
        except BracketError as exc:
            return {"accepted": False, "reason": f"bracket invalid: {exc}"}

        state = self._refresh_state()
        size = size_position(state, self.coordinator.limits, self.asset, levels.entry, levels.stop,
                             self.coordinator.exposure())
        if not size.approved:
            self.audit.record("mcp_order_vetoed", {"symbol": symbol, "reasons": list(size.reasons)})
            return {"accepted": False, "reason": "; ".join(size.reasons) or "risk veto"}
        try:
            order = BracketOrder(symbol, trade_side, size.units, levels.entry, levels.stop, levels.take_profit,
                                 strategy="mcp")
            result = broker.place_bracket(order)
        except (OrderRejected, KillSwitchEngaged) as exc:
            return {"accepted": False, "reason": str(exc)}
        self.audit.record("mcp_order", {"mode": mode, "symbol": symbol, "side": trade_side.value,
                                        "units": size.units, "risk": size.risk_amount,
                                        "accepted": result.accepted, "client_id": result.client_id})
        return to_jsonable({"accepted": result.accepted, "mode": mode, "units": size.units,
                            "risk_amount": size.risk_amount, "binding_constraint": size.binding_constraint,
                            "fill_price": result.fill_price, "client_id": result.client_id,
                            "message": result.message})

    def trip_kill_switch(self, reason: str = "manual trip via MCP") -> dict[str, Any]:
        """Cancel all orders and flatten all positions on every attached broker."""
        return to_jsonable(self.kill_switch.trip(reason, source="mcp"))

    def tools(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Name → callable map used by both transports."""
        return {"fetch_ohlcv": self.fetch_ohlcv, "backtest_strategy": self.backtest_strategy,
                "get_risk_metrics": self.get_risk_metrics, "execute_order": self.execute_order,
                "trip_kill_switch": self.trip_kill_switch}


# --------------------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------------------
SERVER_NAME = "quantmind-propfirm-orchestrator"
# mcp 2.x renamed FastMCP to MCPServer and removed mcp.server.fastmcp; add_tool is unchanged.
MCP_SERVER_CLASSES: tuple[tuple[str, str], ...] = (
    ("mcp.server.mcpserver", "MCPServer"),  # mcp >= 2
    ("mcp.server.fastmcp", "FastMCP"),      # mcp 1.x
    ("fastmcp", "FastMCP"),                 # standalone fastmcp package
)


class McpUnavailableError(ImportError):
    """No usable MCP server class: the package is missing or its API is unsupported."""


def load_mcp_server_class() -> type:
    """Return the first available MCP server class, or raise :class:`McpUnavailableError`
    saying whether ``mcp`` is missing or installed but unsupported."""
    failures: list[str] = []
    for module_name, attr in MCP_SERVER_CLASSES:
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            failures.append(f"{module_name}.{attr}: {type(exc).__name__}: {exc}")
    if importlib.util.find_spec("mcp") is None and importlib.util.find_spec("fastmcp") is None:
        raise McpUnavailableError("the 'mcp' package is not installed: pip install mcp  (or run with --jsonrpc)")
    raise McpUnavailableError(
        "the installed MCP package has no supported server class (upgrade it, or run with --jsonrpc). "
        "Tried: " + "; ".join(failures))


def build_mcp_server(service: OrchestratorService) -> Any:
    """Register the service's tools on an MCP stdio server (requires ``pip install mcp``)."""
    server = load_mcp_server_class()(SERVER_NAME)
    for name, fn in service.tools().items():
        server.add_tool(fn, name=name, description=(fn.__doc__ or "").strip())
    return server


build_fastmcp = build_mcp_server  # backward-compatible name used by the README and older callers


def handle_jsonrpc(service: OrchestratorService, request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one JSON-RPC 2.0 request (``method`` = tool name or ``tools/list``)."""
    rid = request.get("id")
    method = request.get("method", "")
    tools = service.tools()
    try:
        if method == "tools/list":
            result: Any = [{"name": n, "description": (f.__doc__ or "").strip()} for n, f in tools.items()]
        elif method in tools:
            result = to_jsonable(tools[method](**(request.get("params") or {})))
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    except Exception as exc:  # noqa: BLE001 - reported to the client
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def serve_jsonrpc(service: OrchestratorService) -> None:
    """Line-delimited JSON-RPC 2.0 over stdio (no third-party dependency)."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except ValueError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle_jsonrpc(service, req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m mcp_server.mcp_bridge [--jsonrpc]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jsonrpc", action="store_true", help="use the built-in JSON-RPC stdio loop")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    args = parser.parse_args(argv)
    service = OrchestratorService(args.state_dir)
    if args.jsonrpc:
        serve_jsonrpc(service)
        return 0
    try:
        server = build_mcp_server(service)
    except McpUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
