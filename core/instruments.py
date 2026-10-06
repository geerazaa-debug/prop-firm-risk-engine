"""Default contract specifications, shared by the CLI and the MCP server.

These are conservative defaults for paper trading and first connection tests. For live
trading, pass the broker's real values (``SYMBOL_TRADE_TICK_SIZE``, stops level, lot
limits, contract size) — ``python -m app.main mt5-ping`` prints them from the terminal.
"""
from __future__ import annotations

from core.risk_engine import AssetSpec

#: Spot gold as quoted on typical MT5 CFD accounts: 100 oz per lot, $0.01 price tick.
XAUUSD_TICK = 0.01


def default_asset(symbol: str) -> AssetSpec:
    """Return the default :class:`AssetSpec` for ``symbol``.

    * ``XAU*``: 100 oz/lot, 0.01-lot steps, 1:30 leverage, **0.01 price tick**, so
      :func:`~core.risk_engine.hard_bracket` rounds SL/TP to 2 decimals.
    * ``BASE/QUOTE`` crypto (e.g. ``BTC/USDT``): 1 unit, 0.001 steps, 1:5 leverage.
    * Anything else: an FX-style 100k-unit lot at 1:30 (no tick rounding).
    """
    upper = symbol.upper()
    if upper.startswith("XAU"):
        return AssetSpec(symbol, leverage=30, contract_size=100, lot_step=0.01, min_lot=0.01,
                         price_tick=XAUUSD_TICK)
    if "/" in symbol and upper.endswith(("USDT", "USD")):
        return AssetSpec(symbol, leverage=5, contract_size=1, lot_step=0.001, min_lot=0.001)
    return AssetSpec(symbol, leverage=30, contract_size=100_000, lot_step=0.01, min_lot=0.01)
