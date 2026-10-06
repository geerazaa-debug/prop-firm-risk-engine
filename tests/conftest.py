"""Shared fixtures and bar builders."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.market_data import Bar

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def bars_from_closes(closes: Sequence[float], wick: float = 0.1, start: datetime = T0,
                     step: timedelta = timedelta(days=1)) -> list[Bar]:
    """Build bars whose open is the previous close and whose wicks extend ``wick`` beyond the body."""
    out: list[Bar] = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        out.append(Bar(start + i * step, o, max(o, c) + wick, min(o, c) - wick, c, 1.0))
        prev = c
    return out


class FakeMt5Transport:
    """In-memory stand-in for the MT5 transport (ZeroMQ EA or native package)."""

    def __init__(self, trade_allowed: bool = True, stops_level: int = 10) -> None:
        self.trade_allowed = trade_allowed
        self.stops_level = stops_level
        self.sent: list[dict[str, Any]] = []
        self.open: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.fail_ping = False
        self.timeout_send: bool = False
        self.fill_on_timeout: bool = False
        self.reconnects = 0
        self.trade_mode = 0  # MT5 ACCOUNT_TRADE_MODE: 0 demo, 1 contest, 2 real
        self.equity = 50_000.0
        self.currency = "USD"
        self.server_offset = 3 * 3600  # server clock = UTC+3 (typical EET broker)
        self.rate_rows: list[list[float]] = []  # [t_server, o, h, l, c, v]
        self.fail_rates = False

    def ping(self) -> dict[str, Any]:
        from execution.mt5_bridge import Mt5Error

        self.calls.append("PING")
        if self.fail_ping:
            raise Mt5Error("no heartbeat")
        return {"trade_allowed": self.trade_allowed, "server_time": 0, "trade_mode": self.trade_mode,
                "login": 123456}

    def reconnect(self) -> None:
        self.reconnects += 1

    def account_info(self) -> dict[str, Any]:
        return {"trade_allowed": self.trade_allowed, "trade_expert": True, "equity": self.equity,
                "trade_mode": self.trade_mode, "trade_mode_name": {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(
                    self.trade_mode, "UNKNOWN"), "login": 123456, "server": "Demo-Server",
                "currency": self.currency, "leverage": 30}

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        return {"volume_step": 0.01, "volume_min": 0.01, "volume_max": 100.0, "digits": 2, "point": 0.01,
                "trade_tick_size": 0.01, "trade_contract_size": 100.0,
                "ask": 2000.0, "bid": 1999.8, "trade_stops_level": self.stops_level}

    def rates(self, symbol: str, timeframe: str, count: int) -> dict[str, Any]:
        from execution.mt5_bridge import Mt5Error

        self.calls.append("RATES")
        if self.fail_rates:
            raise Mt5Error("RATES failed")
        return {"server_offset": self.server_offset, "bars": [list(r) for r in self.rate_rows[-count:]]}

    def add_bars(self, bars: Sequence[Bar]) -> None:
        """Queue closed bars (UTC) as the EA would report them (server time)."""
        for b in bars:
            self.rate_rows.append([int(b.ts.timestamp()) + self.server_offset, b.open, b.high, b.low, b.close,
                                   b.volume])

    def send_bracket(self, fields: dict[str, Any]) -> dict[str, Any]:
        from execution.mt5_bridge import RETCODE_DONE, Mt5Timeout

        self.calls.append("SEND_BRACKET")
        self.sent.append(dict(fields))
        pos = {"ticket": len(self.sent), "symbol": fields["symbol"], "type": 0 if fields["side"] == "BUY" else 1,
               "volume": fields["volume"], "price_open": 2000.0, "sl": fields["sl"], "tp": fields["tp"],
               "comment": fields["comment"], "magic": fields["magic"]}
        if self.timeout_send:
            if self.fill_on_timeout:
                self.open.append(pos)
            raise Mt5Timeout("EA did not answer SEND_BRACKET")
        self.open.append(pos)
        return {"retcode": RETCODE_DONE, "order": pos["ticket"], "deal": pos["ticket"], "price": 2000.0}

    def modify_sltp(self, ticket: int, symbol: str, sl: float, tp: float) -> dict[str, Any]:
        from execution.mt5_bridge import RETCODE_DONE

        for p in self.open:
            if p["ticket"] == ticket:
                p["sl"], p["tp"] = sl, tp
        return {"retcode": RETCODE_DONE}

    def positions(self, magic: int | None) -> list[dict[str, Any]]:
        return [dict(p) for p in self.open]

    def pending_orders(self, magic: int | None) -> list[dict[str, Any]]:
        return []

    def cancel_all(self, magic: int | None) -> dict[str, Any]:
        self.calls.append("CANCEL_ALL")
        return {"cancelled": 0, "failed": []}

    def close_position(self, ticket: int, deviation: int) -> dict[str, Any]:
        from execution.mt5_bridge import RETCODE_DONE

        self.open = [p for p in self.open if p["ticket"] != ticket]
        return {"retcode": RETCODE_DONE}

    def flatten_all(self, magic: int | None, deviation: int) -> dict[str, Any]:
        self.calls.append("FLATTEN_ALL")
        n, self.open = len(self.open), []
        return {"closed": n, "cancelled": 0, "failed": []}


class AlwaysLongStrategy:
    """Test strategy that signals a long on every bar with deliberately unrounded levels."""

    name = "always_long"
    warmup = 2

    def __init__(self, stop_offset: float = 19.876543, tp_offset: float = 40.987654) -> None:
        self.stop_offset = stop_offset
        self.tp_offset = tp_offset

    def params(self) -> dict[str, Any]:
        return {"stop_offset": self.stop_offset, "tp_offset": self.tp_offset}

    def evaluate(self, symbol: str, bars: Sequence[Bar]) -> Any:
        from core.risk_engine import Side
        from core.strategy_brain import Signal, SignalKind

        c = bars[-1].close
        return Signal(self.name, symbol, Side.LONG, SignalKind.DIRECTIONAL, c, c - self.stop_offset,
                      c + self.tp_offset, 0.9, "test", bars[-1].ts)


def trend_bars(n: int = 200, start: float = 2000.0, step: float = 0.5) -> list[Bar]:
    """Steady uptrend: always passes the ER gate."""
    return bars_from_closes([start + i * step for i in range(n)], wick=0.2)


def open_regime_filter() -> Any:
    """Regime filter that never blocks (isolates compliance/execution behaviour in tests)."""
    from core.regime_filter import RegimeFilter, RegimeFilterConfig

    return RegimeFilter(RegimeFilterConfig(er_threshold=0.0, block_extreme_vol=False))
