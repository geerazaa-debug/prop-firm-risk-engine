"""Core pure business logic: risk engine, regime filter, strategies, backtester.

Nothing in ``core`` performs I/O against a broker or imports a UI library. Every
function here is deterministic given its inputs so it can be unit-tested and
reasoned about in isolation.
"""
from __future__ import annotations
