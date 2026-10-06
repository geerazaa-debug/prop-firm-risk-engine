"""Self-test of the real pyzmq path against a mock EA that speaks the EA protocol.

Exercises the actual sockets (no fakes) without MetaTrader: bracket round-trip, token
auth, hung-EA detection, order refusal while the link is down, no-retry of
SEND_BRACKET, reconnect after recovery, kill-switch FLATTEN_ALL, and PUB events.

Usage (requires ``pip install pyzmq``; uses ports 15555/15556)::

    python scripts/mt5_bridge_selftest.py
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zmq

from core.risk_engine import Side
from execution.base import BracketOrder
from execution.kill_switch import KillSwitch
from execution.mt5_adapter import Mt5Adapter
from execution.mt5_bridge import (
    HeartbeatConfig,
    HeartbeatMonitor,
    LinkState,
    Mt5EventSubscriber,
    Mt5Timeout,
    ZmqMt5Transport,
)

REP, PUB, TOKEN = "tcp://127.0.0.1:15555", "tcp://127.0.0.1:15556", "e2e-secret"


class MockEA(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stall = threading.Event()
        self.stop = threading.Event()
        self.positions: list[dict] = []
        self.seen_ids: dict[str, dict] = {}
        self.sends = 0
        self.log: list[str] = []
        self.trade_mode = 0  # 0 demo, 2 real
        self.server_offset = 3 * 3600
        start = int(time.time()) // 3600 * 3600 - 400 * 3600
        # 400 closed H1 bars in *server* time, gently trending up
        self.rates = []
        for i in range(400):
            o = 2000 + i * 0.5
            wick = 0.4 + (i * 37 % 17) / 10  # varied ranges: realistic ATR percentiles
            self.rates.append([start + i * 3600 + self.server_offset, o, o + 0.5 + wick, o - wick, o + 0.3, 100])

    def run(self) -> None:
        ctx = zmq.Context.instance()
        rep, pub = ctx.socket(zmq.REP), ctx.socket(zmq.PUB)
        rep.bind(REP)
        pub.bind(PUB)
        rep.setsockopt(zmq.RCVTIMEO, 50)
        while not self.stop.is_set():
            pub.send_string(json.dumps({"type": "heartbeat", "positions": len(self.positions)}))
            try:
                raw = rep.recv_string()
            except zmq.error.Again:
                continue
            req = json.loads(raw)
            self.log.append(req["action"])
            if self.stall.is_set():  # simulate a hung terminal: never reply, drop the socket
                rep.close(linger=0)
                while self.stall.is_set() and not self.stop.is_set():
                    time.sleep(0.02)
                rep = ctx.socket(zmq.REP)
                rep.bind(REP)
                rep.setsockopt(zmq.RCVTIMEO, 50)
                continue
            rep.send_string(json.dumps(self.handle(req)))
            if req["action"] in ("SEND_BRACKET", "FLATTEN_ALL"):
                pub.send_string(json.dumps({"type": "deal", "action": req["action"]}))
        rep.close(linger=0)
        pub.close(linger=0)

    def handle(self, r: dict) -> dict:
        rid = r["id"]
        if r.get("token") != TOKEN:
            return {"ok": False, "id": rid, "error": "authentication failed"}
        a = r["action"]
        if a == "PING":
            return {"ok": True, "id": rid, "data": {"trade_allowed": True, "trade_mode": self.trade_mode}}
        if a == "ACCOUNT_INFO":
            return {"ok": True, "id": rid, "data": {"trade_allowed": True, "trade_expert": True, "equity": 100000,
                                                    "trade_mode": self.trade_mode, "login": 42, "currency": "USD",
                                                    "leverage": 30,
                                                    "trade_mode_name": "REAL" if self.trade_mode == 2 else "DEMO"}}
        if a == "RATES":
            return {"ok": True, "id": rid, "data": {"server_offset": self.server_offset,
                                                    "bars": self.rates[-int(r.get("count", 500)):]}}
        if a == "SYMBOL_INFO":
            return {"ok": True, "id": rid, "data": {"volume_step": 0.01, "volume_min": 0.01, "volume_max": 50,
                                                    "digits": 2, "point": 0.01, "ask": 2000.0, "bid": 1999.8,
                                                    "trade_stops_level": 10, "trade_tick_size": 0.01,
                                                    "trade_contract_size": 100.0}}
        if a == "SEND_BRACKET":
            if not (r.get("sl", 0) > 0 and r.get("tp", 0) > 0):
                return {"ok": False, "id": rid, "error": "SL and TP are mandatory"}
            if r["client_id"] in self.seen_ids:
                return {"ok": True, "id": rid, "data": self.seen_ids[r["client_id"]]}
            self.sends += 1
            self.positions.append({"ticket": self.sends, "symbol": r["symbol"], "type": 0, "volume": r["volume"],
                                   "price_open": 2000.0, "sl": r["sl"], "tp": r["tp"], "comment": r["comment"]})
            data = {"retcode": 10009, "order": self.sends, "deal": self.sends, "price": 2000.0}
            self.seen_ids[r["client_id"]] = data
            return {"ok": True, "id": rid, "data": data}
        if a == "POSITIONS":
            return {"ok": True, "id": rid, "data": self.positions}
        if a == "CANCEL_ALL":
            return {"ok": True, "id": rid, "data": {"cancelled": 0, "failed": []}}
        if a == "FLATTEN_ALL":
            n, self.positions = len(self.positions), []
            return {"ok": True, "id": rid, "data": {"closed": n, "cancelled": 0, "failed": []}}
        return {"ok": False, "id": rid, "error": "unknown action"}


def main() -> None:
    ea = MockEA()
    ea.start()
    time.sleep(0.3)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond, detail))

    events: list[dict] = []
    sub = Mt5EventSubscriber(events.append, endpoint=PUB)
    sub.start()

    t = ZmqMt5Transport(REP, token=TOKEN, timeout_ms=300, retries=1)
    mon = HeartbeatMonitor(t, HeartbeatConfig(interval_s=0.1, max_missed=2, backoff_initial_s=0.1, backoff_max_s=0.4))
    adapter = Mt5Adapter(t, monitor=mon)
    adapter.connect()
    check("connect + heartbeat UP", mon.state is LinkState.UP)

    order = BracketOrder("XAUUSD", Side.LONG, 0.3, 2000.0, 1980.0, 2040.0, client_id="qm-e2e-1")
    res = adapter.place_bracket(order)
    check("bracket accepted over real TCP", res.accepted, res.message)
    check("EA received SL and TP in the same request", ea.positions and ea.positions[0]["sl"] == 1980.0
          and ea.positions[0]["tp"] == 2040.0)

    bad = ZmqMt5Transport(REP, token="wrong", timeout_ms=300)
    try:
        bad.ping()
        check("wrong token rejected", False)
    except Exception as exc:  # noqa: BLE001
        check("wrong token rejected", "authentication failed" in str(exc), str(exc))
    bad.close()

    # Hung EA: heartbeat must go DOWN, orders refused, then recover with a fresh socket.
    ea.stall.set()
    deadline = time.monotonic() + 5
    while mon.state is not LinkState.DOWN and time.monotonic() < deadline:
        time.sleep(0.05)
    check("hung EA detected (DOWN)", mon.state is LinkState.DOWN, mon.state.value)
    refused = adapter.place_bracket(BracketOrder("XAUUSD", Side.LONG, 0.3, 2000.0, 1980.0, 2040.0,
                                                 client_id="qm-e2e-2"))
    check("orders refused while link down", not refused.accepted and "not healthy" in refused.message,
          refused.message)
    try:
        t.send_bracket({"symbol": "XAUUSD", "side": "BUY", "volume": 0.1, "sl": 1980.0, "tp": 2040.0,
                        "deviation": 20, "magic": 1, "client_id": "qm-e2e-3", "comment": "qm-e2e-3"})
        check("SEND_BRACKET times out, not retried", False)
    except Mt5Timeout:
        check("SEND_BRACKET times out, not retried", True)
    ea.stall.clear()
    deadline = time.monotonic() + 5
    while mon.state is not LinkState.UP and time.monotonic() < deadline:
        time.sleep(0.05)
    check("reconnected after EA recovery (UP)", mon.state is LinkState.UP,
          f"reconnects={t.reconnects}")

    # Kill switch -> single FLATTEN_ALL round trip.
    kill = KillSwitch([adapter], retry_delay_s=0)
    t0 = time.perf_counter()
    rep = kill.trip("e2e kill")
    ms = (time.perf_counter() - t0) * 1000
    check("kill switch flattened via FLATTEN_ALL", rep.clean and rep.flattened == 1 and not ea.positions,
          f"{ms:.1f} ms cancel+flatten round trips")
    time.sleep(0.3)
    check("PUB events received", any(e.get("type") == "heartbeat" for e in events)
          and any(e.get("type") == "deal" for e in events), f"{len(events)} events")

    # Live loop over real TCP: bar streaming + demo guard.
    from agent.coordinator import TradingCoordinator
    from core.regime_filter import RegimeFilter, RegimeFilterConfig
    from core.strategy_brain import TrendBreakoutStrategy
    from execution.live_runner import DemoAccountGuardError, LiveConfig, LiveRunner, Mt5BarFeed, asset_from_symbol_info

    kill.rearm("REARM")
    asset = asset_from_symbol_info("XAUUSD", t.symbol_info("XAUUSD"), t.account_info())
    feed = Mt5BarFeed(t, "XAUUSD", "1h", history=300)
    coord = TradingCoordinator("XAUUSD", asset, adapter, kill, [TrendBreakoutStrategy()],
                               regime_filter=RegimeFilter(RegimeFilterConfig(er_threshold=0.0)))
    runner = LiveRunner(coord, feed, t, kill, mon, LiveConfig(poll_seconds=0.05))
    runner.preflight()
    check("RATES streamed closed bars, server time -> UTC",
          len(runner.bars) == 300 and runner.bars[-1].ts.timestamp() == ea.rates[-1][0] - ea.server_offset)
    sends_before = ea.sends
    last_close = ea.rates[-1][4]  # a normal-sized bar that closes just above the 20-bar high
    channel_high = max(row[2] for row in ea.rates[-20:])
    ea.rates.append([ea.rates[-1][0] + 3600, last_close, channel_high + 0.6, last_close - 0.3,
                     channel_high + 0.5, 100])
    rep = runner.step()
    check("live step traded the new closed bar with SL/TP", rep.new_bars == 1 and ea.sends == sends_before + 1
          and ea.positions[-1]["sl"] > 0 and ea.positions[-1]["tp"] > 0, rep.cycle.decision if rep.cycle else "")
    ea.trade_mode = 2  # terminal switched to a REAL account
    deadline = time.monotonic() + 3
    while mon.last_info.get("trade_mode") != 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    flat_calls = ea.log.count("FLATTEN_ALL")
    try:
        runner.step()
        check("demo guard stops on REAL account (no orders, no flatten)", False)
    except DemoAccountGuardError:
        check("demo guard stops on REAL account (no orders, no flatten)",
              ea.sends == sends_before + 1 and ea.log.count("FLATTEN_ALL") == flat_calls)

    mon.stop()
    sub.stop()
    t.close()
    ea.stop.set()
    ea.join(timeout=2)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
