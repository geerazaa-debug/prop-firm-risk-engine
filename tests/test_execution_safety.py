"""Execution safety: bracket atomicity, trade-only credentials, kill switch, audit chain."""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from core.audit import AuditTrail
from core.risk_engine import Side
from execution.base import (
    BracketOrder,
    BracketRollbackError,
    KillSwitchEngaged,
    OrderRejected,
    PaperBroker,
)
from execution.cctv_executor import (
    CexCredentials,
    CexExecutor,
    CredentialSecurityError,
    PermissionReport,
    enforce_trade_only,
)
from execution.kill_switch import KillSwitch
from execution.mt5_adapter import Mt5Adapter
from tests.conftest import FakeMt5Transport


def long_order(**kw: Any) -> BracketOrder:
    base: dict[str, Any] = {"symbol": "BTC/USDT:USDT", "side": Side.LONG, "units": 0.5,
                            "entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0}
    base.update(kw)
    return BracketOrder(**base)


# ------------------------------------------------------------------ bracket validation
class TestBracketOrder:
    @pytest.mark.parametrize("kw", [
        {"stop_loss": 101.0}, {"take_profit": 99.0}, {"units": 0.0}, {"units": float("nan")},
        {"entry_type": "stop"}, {"side": Side.SHORT},
    ])
    def test_invalid_brackets_rejected(self, kw: dict) -> None:
        with pytest.raises(OrderRejected):
            long_order(**kw)


# ------------------------------------------------------------------ paper broker
class TestPaperBroker:
    def test_stop_fills_before_target_when_both_touched(self) -> None:
        b = PaperBroker(10_000, slippage_bps=0, commission_bps=0)
        b.mark("X", 100.0)
        b.place_bracket(long_order(symbol="X"))
        fills = b.on_bar("X", 100.0, 111.0, 94.0, 105.0)
        assert fills[0].reason == "stop_loss" and fills[0].exit == pytest.approx(95.0)

    def test_gap_through_stop_fills_at_open(self) -> None:
        b = PaperBroker(10_000, slippage_bps=0, commission_bps=0)
        b.mark("X", 100.0)
        b.place_bracket(long_order(symbol="X"))
        fills = b.on_bar("X", 90.0, 91.0, 89.0, 90.5)
        assert fills[0].exit == pytest.approx(90.0)

    def test_amend_stop_never_loosens(self) -> None:
        b = PaperBroker(10_000)
        b.mark("X", 100.0)
        b.place_bracket(long_order(symbol="X"))
        stop = b.positions()[0].stop_loss
        assert b.amend_stop("X", stop + 1.0)
        assert not b.amend_stop("X", stop - 5.0)


# ------------------------------------------------------------------ kill switch
class FlakyBroker(PaperBroker):
    def __init__(self, failures: int) -> None:
        super().__init__(10_000)
        self.failures = failures
        self.flatten_calls = 0

    def flatten_all(self) -> int:
        self.flatten_calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("venue timeout")
        return super().flatten_all()


class TestKillSwitch:
    def test_trip_flattens_and_blocks_orders(self, tmp_path) -> None:
        kill = KillSwitch(sentinel_path=tmp_path / "KILL")
        broker = PaperBroker(10_000, guard=kill)
        kill.register(broker)
        broker.mark("X", 100.0)
        broker.place_bracket(long_order(symbol="X"))
        report = kill.trip("test", source="pytest")
        assert report.flattened == 1 and report.clean
        assert broker.positions() == []
        with pytest.raises(KillSwitchEngaged):
            broker.place_bracket(long_order(symbol="X"))

    def test_trip_persists_across_restart(self, tmp_path) -> None:
        KillSwitch(sentinel_path=tmp_path / "KILL").trip("crash")
        assert KillSwitch(sentinel_path=tmp_path / "KILL").tripped

    def test_external_sentinel_trips_running_engine(self, tmp_path) -> None:
        kill = KillSwitch(sentinel_path=tmp_path / "KILL")
        broker = PaperBroker(10_000)
        kill.register(broker)
        broker.mark("X", 100.0)
        broker.place_bracket(long_order(symbol="X"))
        KillSwitch.write_sentinel(tmp_path / "KILL", "from another terminal", source="cli")
        report = kill.check_external_trigger()
        assert report is not None and report.reason == "from another terminal"
        assert kill.tripped and broker.positions() == []

    def test_rearm_requires_phrase_and_flat_books(self, tmp_path) -> None:
        kill = KillSwitch(sentinel_path=tmp_path / "KILL")
        broker = PaperBroker(10_000)
        kill.register(broker)
        kill.trip("x")
        with pytest.raises(PermissionError):
            kill.rearm("yes please")
        broker.mark("X", 100.0)
        broker.place_bracket(long_order(symbol="X"))  # paper broker has no guard here
        with pytest.raises(PermissionError):
            kill.rearm("REARM")
        broker.flatten_all()
        assert kill.rearm("REARM") and not kill.tripped
        assert not (tmp_path / "KILL").exists()

    def test_retries_transient_failures_and_reports_persistent_ones(self) -> None:
        flaky = FlakyBroker(failures=2)
        kill = KillSwitch([flaky], retry_delay_s=0)
        assert kill.trip("t").clean and flaky.flatten_calls == 3
        dead = FlakyBroker(failures=99)
        report = KillSwitch([dead], retry_delay_s=0, max_retries=3).trip("t")
        assert not report.clean and "venue timeout" in report.errors[0]

    def test_concurrent_trips_are_safe(self) -> None:
        broker = PaperBroker(10_000)
        for i in range(20):
            broker.mark(f"S{i}", 100.0)
            broker.place_bracket(long_order(symbol=f"S{i}"))
        kill = KillSwitch([broker], retry_delay_s=0)
        barrier = threading.Barrier(16)
        reports: list = []

        def worker() -> None:
            barrier.wait()
            reports.append(kill.trip("race"))

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert kill.tripped and broker.positions() == []
        assert sum(r.flattened for r in reports) == 20  # every position closed exactly once
        assert all(r.clean for r in reports)


# ------------------------------------------------------------------ CEX executor
class FakeExchange:
    """Minimal ccxt stand-in recording every call."""

    def __init__(self, restrictions: dict | None = None, fail_on_call: int | None = None) -> None:
        self.restrictions = restrictions if restrictions is not None else {
            "enableWithdrawals": False, "enableFutures": True}
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.fail_on_call = fail_on_call

    def load_markets(self) -> None: ...
    def amount_to_precision(self, _s: str, a: float) -> str: return f"{a:.3f}"
    def price_to_precision(self, _s: str, p: float) -> str: return f"{p:.2f}"
    def sapi_get_account_apirestrictions(self) -> dict: return self.restrictions

    def create_order(self, symbol, type_, side, amount, price=None, params=None) -> dict:
        n = len(self.orders) + 1
        if self.fail_on_call == n:
            self.orders.append({"failed": True})
            raise RuntimeError("exchange rejected child order")
        self.orders.append({"symbol": symbol, "type": type_, "side": side, "amount": amount,
                            "price": price, "params": params or {}})
        return {"id": f"o{n}", "average": 100.0}

    def cancel_order(self, oid: str, _symbol: str) -> None:
        self.cancelled.append(oid)

    def fetch_open_orders(self, symbol: str | None = None) -> list: return []
    def fetch_positions(self) -> list: return []
    def fetch_balance(self) -> dict: return {"total": {"USDT": 1234.5}}


class TestCexExecutor:
    def test_withdrawal_enabled_key_is_refused_even_on_testnet(self) -> None:
        ex = FakeExchange({"enableWithdrawals": True, "enableFutures": True})
        with pytest.raises(CredentialSecurityError, match="WITHDRAWAL"):
            CexExecutor("binanceusdm", exchange=ex, sandbox=True).connect()

    def test_unverifiable_key_refused_on_mainnet_only(self) -> None:
        unverified = PermissionReport(False, False, False, "no endpoint")
        with pytest.raises(CredentialSecurityError):
            enforce_trade_only(unverified, sandbox=False)
        enforce_trade_only(unverified, sandbox=True)  # tolerated on testnet

    def test_key_without_trade_permission_refused(self) -> None:
        with pytest.raises(CredentialSecurityError):
            enforce_trade_only(PermissionReport(True, False, False, "read-only"), sandbox=False)

    def test_orders_refused_before_connect(self) -> None:
        with pytest.raises(OrderRejected):
            CexExecutor("binanceusdm", exchange=FakeExchange()).place_bracket(long_order())

    def test_sequential_bracket_places_reduce_only_children(self) -> None:
        ex = FakeExchange()
        cex = CexExecutor("binanceusdm", exchange=ex, sandbox=False)
        cex.connect()
        res = cex.place_bracket(long_order())
        assert res.accepted and len(ex.orders) == 3
        sl, tp = ex.orders[1], ex.orders[2]
        assert sl["side"] == "sell" and sl["params"]["reduceOnly"] and sl["params"]["stopLossPrice"] == 95.0
        assert tp["params"]["reduceOnly"] and tp["params"]["takeProfitPrice"] == 110.0

    def test_failed_child_rolls_back_entry(self) -> None:
        ex = FakeExchange(fail_on_call=3)  # entry ok, SL ok, TP fails
        cex = CexExecutor("binanceusdm", exchange=ex, sandbox=False)
        cex.connect()
        with pytest.raises(BracketRollbackError):
            cex.place_bracket(long_order())
        assert ex.cancelled == ["o2"]  # the stop that was placed is cancelled
        flatten = ex.orders[-1]
        assert flatten["side"] == "sell" and flatten["params"] == {"reduceOnly": True}
        assert flatten["amount"] == pytest.approx(0.5)

    def test_attached_mode_is_single_atomic_request(self) -> None:
        ex = FakeExchange()
        ex.private_get_v5_user_query_api = lambda: {"result": {"readOnly": 0,  # type: ignore[attr-defined]
                                                               "permissions": {"ContractTrade": ["Order"]}}}
        cex = CexExecutor("bybit", exchange=ex, sandbox=False)
        cex.connect()
        res = cex.place_bracket(long_order())
        assert res.atomic and len(ex.orders) == 1
        assert ex.orders[0]["params"]["stopLoss"] == {"triggerPrice": 95.0}
        assert ex.orders[0]["params"]["takeProfit"] == {"triggerPrice": 110.0}

    def test_bybit_withdraw_permission_detected(self) -> None:
        ex = FakeExchange()
        ex.private_get_v5_user_query_api = lambda: {"result": {"readOnly": 0,  # type: ignore[attr-defined]
                                                               "permissions": {"Wallet": ["Withdraw"]}}}
        with pytest.raises(CredentialSecurityError):
            CexExecutor("bybit", exchange=ex).connect()

    def test_kill_guard_blocks_orders(self, tmp_path) -> None:
        kill = KillSwitch(sentinel_path=tmp_path / "K")
        cex = CexExecutor("binanceusdm", exchange=FakeExchange(), kill_guard=kill)
        cex.connect()
        kill.trip("x")
        with pytest.raises(KillSwitchEngaged):
            cex.place_bracket(long_order())

    def test_credentials_never_leak_in_repr(self, monkeypatch) -> None:
        monkeypatch.setenv("QM_CEX_API_KEY", "AKIA-very-secret-key")
        monkeypatch.setenv("QM_CEX_API_SECRET", "super-secret-value")
        creds = CexCredentials.from_env()
        assert "secret" not in repr(creds) and "AKIA" not in repr(creds)
        monkeypatch.delenv("QM_CEX_API_SECRET")
        with pytest.raises(CredentialSecurityError):
            CexCredentials.from_env()


# ------------------------------------------------------------------ MT5 bridge
class TestMt5Adapter:
    def test_sl_tp_travel_with_entry(self) -> None:
        fake = FakeMt5Transport()
        adapter = Mt5Adapter(fake)
        adapter.connect()
        res = adapter.place_bracket(BracketOrder("XAUUSD", Side.LONG, 0.137, 2000.0, 1980.0, 2050.0))
        assert res.accepted and res.atomic
        assert len(fake.sent) == 1  # one request carries entry + SL + TP
        sent = fake.sent[0]
        assert (sent["symbol"], sent["side"], sent["volume"], sent["sl"], sent["tp"]) == \
            ("XAUUSD", "BUY", 0.13, 1980.0, 2050.0)

    def test_investor_login_refused(self) -> None:
        with pytest.raises(PermissionError):
            Mt5Adapter(FakeMt5Transport(trade_allowed=False)).connect()

    def test_stops_inside_stops_level_rejected(self) -> None:
        adapter = Mt5Adapter(FakeMt5Transport())
        adapter.connect()
        res = adapter.place_bracket(BracketOrder("XAUUSD", Side.LONG, 0.1, 2000.0, 1999.95, 2050.0))
        assert not res.accepted and "stops level" in res.message


# ------------------------------------------------------------------ audit trail
class TestAuditTrail:
    def test_chain_verifies_and_detects_tampering(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        trail = AuditTrail(path)
        for i in range(5):
            trail.record("evt", {"i": i})
        assert trail.verify() == (True, None)
        lines = path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[2])
        rec["payload"]["i"] = 999
        lines[2] = json.dumps(rec)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok, bad = AuditTrail(path).verify()
        assert not ok and bad == 3

    def test_resume_continues_chain(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        AuditTrail(path).record("a")
        AuditTrail(path).record("b")
        assert AuditTrail(path).verify() == (True, None)

    def test_secrets_are_redacted(self) -> None:
        rec = AuditTrail().record("cfg", {"api_key": "abc", "nested": {"secret": "x"}, "symbol": "BTC"})
        assert rec.payload == {"api_key": "***", "nested": {"secret": "***"}, "symbol": "BTC"}
