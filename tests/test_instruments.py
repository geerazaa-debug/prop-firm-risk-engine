"""Gold tick-size defaults, 0.01 SL/TP rounding, and the mt5-ping connectivity helper."""
from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

import pytest

import app.main as cli
from core.instruments import XAUUSD_TICK, default_asset
from core.risk_engine import Side, hard_bracket
from execution.mt5_bridge import Mt5Error, Mt5Timeout
from mcp_server.mcp_bridge import OrchestratorService
from tests.conftest import FakeMt5Transport


def cents(x: float) -> bool:
    """True if ``x`` is an exact multiple of $0.01 (checked in decimal, not float)."""
    return Decimal(repr(x)).quantize(Decimal("0.01")) == Decimal(repr(x))


class TestGoldDefaults:
    def test_every_default_gold_spec_has_cent_tick(self, tmp_path) -> None:
        assert XAUUSD_TICK == 0.01
        assert default_asset("XAUUSD").price_tick == 0.01
        assert cli.asset_for("XAUUSD").price_tick == 0.01
        assert OrchestratorService(tmp_path).asset.price_tick == 0.01

    def test_other_symbols_unchanged(self) -> None:
        assert default_asset("BTC/USDT").price_tick == 0.0
        assert default_asset("EURUSD").price_tick == 0.0


class TestGoldBracketRounding:
    def test_worked_example(self) -> None:
        gold = default_asset("XAUUSD")
        long_ = hard_bracket(Side.LONG, 2350.0, 2331.12789, 2391.45612, gold)
        assert (long_.stop, long_.take_profit) == (2331.12, 2391.45)  # stop away, TP toward entry
        short = hard_bracket(Side.SHORT, 2350.0, 2368.87211, 2308.54388, gold)
        assert (short.stop, short.take_profit) == (2368.88, 2308.55)

    def test_already_on_tick_is_untouched(self) -> None:
        lv = hard_bracket(Side.LONG, 2350.0, 2330.25, 2390.5, default_asset("XAUUSD"))
        assert (lv.stop, lv.take_profit, lv.adjusted) == (2330.25, 2390.5, False)

    @pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
    def test_property_two_decimals_and_conservative(self, side: Side) -> None:
        gold = default_asset("XAUUSD")
        rng = random.Random(7)
        for _ in range(5_000):
            entry = round(rng.uniform(1_500, 5_000), 2)
            stop = entry - side.sign * rng.uniform(0.5, 80)
            tp = entry + side.sign * rng.uniform(0.5, 160)
            lv = hard_bracket(side, entry, stop, tp, gold)
            assert cents(lv.stop) and cents(lv.take_profit), (stop, tp, lv)
            assert abs(lv.stop - entry) >= abs(stop - entry) - 1e-9  # never a tighter stop
            assert abs(lv.take_profit - entry) <= abs(tp - entry) + 1e-9  # never a further target
            assert abs(lv.stop - stop) < 0.01 + 1e-9 and abs(lv.take_profit - tp) < 0.01 + 1e-9

    def test_cli_paper_order_carries_cent_levels(self, tmp_path) -> None:
        from datetime import UTC, datetime

        svc = OrchestratorService(tmp_path, clock=lambda: datetime(2026, 9, 23, 12, tzinfo=UTC))
        svc.fetch_ohlcv("XAUUSD", limit=300)
        px = svc._bars["XAUUSD"][-1].close
        sent: list[Any] = []
        real = svc.paper.place_bracket
        svc.paper.place_bracket = lambda o: (sent.append(o), real(o))[1]  # type: ignore[method-assign]
        assert svc.execute_order("long", px * 0.98731, px * 1.04219)["accepted"]
        (order,) = sent
        assert cents(order.stop_loss) and cents(order.take_profit)
        assert order.stop_loss <= px * 0.98731 and order.take_profit <= px * 1.04219


class TestMt5PingHelper:
    def test_healthy_terminal_passes(self) -> None:
        ok, lines = cli.mt5_ping_report(FakeMt5Transport(), "XAUUSD", default_asset("XAUUSD"))
        text = "\n".join(lines)
        assert ok, text
        assert "price_tick 0.01 is a multiple of broker tick" in text

    def test_three_digit_gold_broker_is_compatible(self) -> None:
        fake = FakeMt5Transport()
        base = fake.symbol_info
        fake.symbol_info = lambda s: {**base(s), "digits": 3, "point": 0.001, "trade_tick_size": 0.001}  # type: ignore
        ok, _ = cli.mt5_ping_report(fake, "XAUUSD", default_asset("XAUUSD"))
        assert ok

    def test_coarser_broker_tick_fails(self) -> None:
        fake = FakeMt5Transport()
        base = fake.symbol_info
        fake.symbol_info = lambda s: {**base(s), "trade_tick_size": 0.05}  # type: ignore
        ok, lines = cli.mt5_ping_report(fake, "XAUUSD", default_asset("XAUUSD"))
        assert not ok and any("NOT a multiple" in ln for ln in lines)

    def test_trading_disabled_fails(self) -> None:
        ok, lines = cli.mt5_ping_report(FakeMt5Transport(trade_allowed=False), "XAUUSD", default_asset("XAUUSD"))
        assert not ok and any("trading not allowed" in ln for ln in lines)

    def test_spec_mismatch_is_a_warning(self) -> None:
        fake = FakeMt5Transport()
        base = fake.symbol_info
        fake.symbol_info = lambda s: {**base(s), "trade_contract_size": 10.0}  # type: ignore
        ok, lines = cli.mt5_ping_report(fake, "XAUUSD", default_asset("XAUUSD"))
        assert ok and any("contract size" in ln and "WARN" in ln for ln in lines)


class TestMt5PingCommand:
    def run(self, monkeypatch, capsys, behaviour: Any, *argv: str) -> tuple[int, str]:
        class StubTransport(FakeMt5Transport):
            def __init__(self, *a: Any, **kw: Any) -> None:
                super().__init__()
                self.closed = False

            def ping(self) -> dict[str, Any]:
                if isinstance(behaviour, Exception):
                    raise behaviour
                return super().ping()

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(cli, "ZmqMt5Transport", StubTransport)
        code = cli.main(["mt5-ping", *argv])
        return code, capsys.readouterr().out

    def test_requires_token(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv(cli.MT5_TOKEN_ENV, raising=False)
        code, out = self.run(monkeypatch, capsys, None)
        assert code == 2 and "QM_MT5_TOKEN" in out

    def test_success(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv(cli.MT5_TOKEN_ENV, "tok")
        code, out = self.run(monkeypatch, capsys, None)
        assert code == 0 and "MT5 link OK" in out and "tok" not in out  # token never printed

    def test_unreachable_ea_gives_checklist(self, monkeypatch, capsys) -> None:
        code, out = self.run(monkeypatch, capsys, Mt5Timeout("no answer"), "--token", "tok")
        assert code == 1 and "not reachable" in out and "Allow DLL imports" in out

    def test_bad_token_hint(self, monkeypatch, capsys) -> None:
        code, out = self.run(monkeypatch, capsys, Mt5Error("EA rejected PING: authentication failed"),
                             "--token", "wrong")
        assert code == 1 and "InpAuthToken" in out
