"""Broker adapters, order pipeline and emergency circuit breakers.

* :mod:`execution.base` — broker-neutral order types, the ``BrokerAdapter`` protocol
  and a thread-safe :class:`~execution.base.PaperBroker`.
* :mod:`execution.cctv_executor` — CEX order manager (ccxt) with atomic-or-rollback
  bracket orders and trade-only credential verification.
* :mod:`execution.mt5_bridge` — MetaTrader 5 bridge (native package or ZeroMQ EA).
* :mod:`execution.kill_switch` — thread-safe, persistent emergency flatten.
"""
from __future__ import annotations
