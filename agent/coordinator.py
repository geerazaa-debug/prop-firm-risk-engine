"""Multi-agent routing and the live regime gatekeeper.

:class:`ResearchCoordinator` runs the Generator → Critic loop and keeps only promoted
strategies.

:class:`TradingCoordinator` is the per-bar decision pipeline for paper/live trading.
Every cycle goes through the same fixed chain, and any link can veto:

    kill switch ─► risk engine (mark-to-market) ─► compliance (weekend flatten) ─►
    ratchet stops ─► regime gate + signal aggregation ─► compliance (weekend/news veto) ─►
    hard SL/TP levels (risk engine) ─► position sizing ─► bracket order

Every decision — including every veto — is written to the audit trail.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agent.critic import CriticVerdict, QCritic
from agent.generator import HypothesisGenerator
from core.audit import AuditTrail
from core.backtest import BacktestConfig
from core.compliance import ComplianceDecision, ComplianceGate
from core.indicators import atr_series
from core.market_data import Bar
from core.regime_filter import RegimeFilter
from core.risk_engine import (
    DEFAULT_RATCHET,
    AccountState,
    AssetSpec,
    BracketError,
    BracketLevels,
    ExposureSnapshot,
    PositionSize,
    RatchetConfig,
    RatchetStop,
    RiskAction,
    RiskEvaluation,
    RiskLimits,
    hard_bracket,
    mark_to_market,
    ratchet_stop,
    risk_report,
    size_position,
)
from core.strategy_brain import AggregatedDecision, Signal, SignalAggregator, Strategy
from execution.base import (
    BracketOrder,
    BracketResult,
    BracketRollbackError,
    BrokerAdapter,
    KillSwitchEngaged,
    OrderRejected,
    Position,
)
from execution.kill_switch import KillSwitch


# --------------------------------------------------------------------------------------
# Research: Generator -> Critic
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ResearchReport:
    """Outcome of one research round."""

    symbol: str
    trials: int
    verdicts: tuple[CriticVerdict, ...]

    @property
    def promoted(self) -> list[CriticVerdict]:
        """Promoted verdicts, best first."""
        return sorted((v for v in self.verdicts if v.promoted), key=lambda v: v.score, reverse=True)


class ResearchCoordinator:
    """Routes candidates from the generator to the critic and records the outcome."""

    def __init__(self, generator: HypothesisGenerator | None = None, critic: QCritic | None = None,
                 backtest: BacktestConfig | None = None, audit: AuditTrail | None = None) -> None:
        self.generator = generator or HypothesisGenerator()
        self.critic = critic or QCritic()
        self.backtest = backtest or BacktestConfig()
        self.audit = audit or AuditTrail()

    def run(self, bars: Sequence[Bar], symbol: str = "SYM", max_candidates: int | None = None,
            prior_trials: int = 0) -> ResearchReport:
        """Generate candidates, review each one, return the report.

        The snooping hurdle used for *every* candidate in the round is based on the
        generator's trial count after the whole batch is drawn, so candidates reviewed
        first do not get an easier bar than those reviewed last. ``prior_trials`` adds
        tests already run elsewhere in the same research family (e.g. the same grid on
        another timeframe), so the hurdle reflects every test performed.
        """
        candidates = self.generator.generate(max_candidates)
        trials = self.generator.trials + max(0, prior_trials)
        verdicts: list[CriticVerdict] = []
        for cand in candidates:
            verdict = self.critic.review(cand, bars, self.backtest, trials, symbol)
            verdicts.append(verdict)
            self.audit.record("critic_verdict", verdict.to_dict())
        report = ResearchReport(symbol, trials, tuple(verdicts))
        self.audit.record("research_complete", {"symbol": symbol, "trials": trials,
                                                "promoted": [v.candidate.candidate_id for v in report.promoted]})
        return report


# --------------------------------------------------------------------------------------
# Live / paper decision pipeline
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CycleReport:
    """What happened in one trading cycle."""

    ts: datetime
    equity: float
    status: str
    risk_action: str
    decision: str
    order: BracketResult | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


class TradingCoordinator:
    """Per-bar gatekeeper that owns the account risk state for one symbol.

    Args:
        symbol: Instrument traded.
        asset: Contract specification used for sizing.
        broker: Execution adapter (paper, CEX, MT5).
        kill_switch: Kill switch protecting ``broker``.
        strategies: Promoted strategies.
        limits: Risk limits.
        regime_filter: Regime gate (KER + volatility).
        ratchet: Trailing-stop configuration (None disables trailing).
        audit: Audit trail.
        compliance: Weekend/news gate (None disables session compliance).
    """

    def __init__(self, symbol: str, asset: AssetSpec, broker: BrokerAdapter, kill_switch: KillSwitch,
                 strategies: Sequence[Strategy], limits: RiskLimits | None = None,
                 regime_filter: RegimeFilter | None = None, ratchet: RatchetConfig | None = DEFAULT_RATCHET,
                 audit: AuditTrail | None = None, compliance: ComplianceGate | None = None) -> None:
        self.symbol = symbol
        self.asset = asset
        self.broker = broker
        self.kill_switch = kill_switch
        self.limits = limits or RiskLimits()
        self.aggregator = SignalAggregator(strategies, regime_filter or RegimeFilter())
        self.ratchet_cfg = ratchet
        self.audit = audit or AuditTrail()
        self.state: AccountState | None = None
        self.ratchets: dict[str, RatchetStop] = {}
        self.last_decision: AggregatedDecision | None = None
        self.compliance = compliance
        self.last_compliance: ComplianceDecision | None = None
        kill_switch.register(broker)

    @property
    def warmup(self) -> int:
        """Bars required before entries are possible."""
        return self.aggregator.warmup

    def exposure(self) -> ExposureSnapshot:
        """Aggregate open risk/notional/margin from the broker's positions."""
        uv = self.asset.contract_size * self.asset.quote_to_account
        risk = notional = 0.0
        for p in self.broker.positions():
            r = self.ratchets.get(p.symbol)
            stop = r.stop if r is not None else p.stop_loss
            risk += max(0.0, (p.entry_price - stop) * p.side.sign) * p.units * uv if stop else float("inf")
            notional += p.entry_price * p.units * uv
        return ExposureSnapshot(risk, notional, notional / self.asset.leverage)

    def risk_snapshot(self) -> dict[str, Any]:
        """Risk metrics for dashboards and the MCP ``get_risk_metrics`` tool."""
        if self.state is None:
            return {"status": "uninitialised", "kill_switch": self.kill_switch.tripped}
        rep = risk_report(self.state, self.limits)
        exp = self.exposure()
        out: dict[str, Any] = {k: getattr(rep, k) for k in rep.__dataclass_fields__ if k != "extras"}
        out.update(kill_switch=self.kill_switch.tripped, open_risk=exp.open_risk, open_notional=exp.open_notional,
                   positions=[{"symbol": p.symbol, "side": p.side.value, "units": p.units, "entry": p.entry_price,
                               "stop": (self.ratchets[p.symbol].stop if p.symbol in self.ratchets else p.stop_loss),
                               "take_profit": p.take_profit} for p in self.broker.positions()])
        return out

    def _emergency_flatten(self, reason: str, source: str = "risk_engine") -> list[str]:
        """Non-latching cancel+flatten through the kill switch; escalate to a trip if it fails."""
        rep = self.kill_switch.soft_flatten(reason, source=source)
        notes = [f"flattened {rep.flattened}, cancelled {rep.cancelled}"]
        if not rep.clean:
            notes.append(f"FLATTEN FAILED: {list(rep.errors)}")
            self.kill_switch.trip(f"{reason}; flatten failed: {list(rep.errors)}", source=source)
        self.ratchets.clear()
        return notes

    def _mark(self, ts: datetime, notes: list[str]) -> RiskEvaluation:
        """External kill check + risk mark-to-market at ``ts``; applies freeze/terminate actions."""
        ext = self.kill_switch.check_external_trigger()
        if ext is not None:
            notes.append(f"external kill: {ext.reason}")
        equity = self.broker.equity()
        if self.state is None:
            self.state = AccountState.open(equity, ts, self.limits)
        ev = mark_to_market(self.state, equity, ts, self.limits)
        self.state = ev.state
        if ev.action is RiskAction.FREEZE_AND_FLATTEN:
            notes += self._emergency_flatten(ev.message)
            self.audit.record("daily_loss_freeze", {"message": ev.message, "equity": equity, "notes": notes})
        elif ev.action is RiskAction.TERMINATE_AND_FLATTEN:
            self.kill_switch.trip(ev.message, source="risk_engine")
            self.ratchets.clear()
            self.audit.record("drawdown_breach", {"message": ev.message, "equity": equity})
        return ev

    def _enforce_compliance(self, now: datetime, notes: list[str]) -> ComplianceDecision | None:
        """Evaluate session compliance at ``now`` and flatten if it demands it."""
        compliance = self.compliance.evaluate(now, self.symbol) if self.compliance else None
        self.last_compliance = compliance
        if compliance is not None and compliance.must_flatten and self.broker.positions():
            reason = "; ".join(compliance.reasons)
            notes += self._emergency_flatten(reason, source="compliance")
            self.audit.record("compliance_flatten", {"symbol": self.symbol, "reason": reason, "now": now.isoformat(),
                                                     "notes": notes})
        return compliance

    def supervise(self, now: datetime) -> list[str]:
        """Between-bar supervision for live loops (no new entries are considered).

        Marks the account to market at wall-clock ``now`` (so the daily-loss and drawdown
        limits react within one poll, not one bar) and applies session compliance, e.g.
        the Friday 19:30 UTC flatten even when a bar does not close exactly then.
        """
        notes: list[str] = []
        self._mark(now, notes)
        if not self.kill_switch.tripped:
            self._enforce_compliance(now, notes)
        return notes

    @staticmethod
    def decision_time(bars: Sequence[Bar]) -> datetime:
        """Close time of the last bar (``ts`` is the open time), inferred from bar spacing."""
        step = bars[-1].ts - bars[-2].ts if len(bars) >= 2 else timedelta(0)
        return bars[-1].ts + step

    def _report(self, bar: Bar, ev: RiskEvaluation, notes: list[str], decision: str,
                order: BracketResult | None = None) -> CycleReport:
        assert self.state is not None
        return CycleReport(bar.ts, self.state.equity, self.state.status.value, ev.action.value, decision,
                           order, tuple(notes))

    def _sync_ratchets(self) -> dict[str, Position]:
        """Keep ratchet state in sync with the broker (SL/TP may have filled)."""
        open_syms = {p.symbol: p for p in self.broker.positions()}
        for sym in list(self.ratchets):
            if sym not in open_syms:
                del self.ratchets[sym]
        return open_syms

    def _trail_stop(self, bars: Sequence[Bar], bar: Bar, notes: list[str]) -> None:
        """Advance this symbol's ratchet stop; amend the broker's SL only when it moves."""
        if self.symbol not in self.ratchets or self.ratchet_cfg is None:
            return
        current = self.ratchets[self.symbol]
        atr = atr_series(bars[-(self.aggregator.regime_filter.config.atr_period * 3):],
                         self.aggregator.regime_filter.config.atr_period)[-1]
        nxt = ratchet_stop(current, bar.high, bar.low, atr, self.ratchet_cfg)
        if nxt.stop == current.stop:
            self.ratchets[self.symbol] = nxt
            return
        if self.broker.amend_stop(self.symbol, nxt.stop):
            self.ratchets[self.symbol] = nxt
            self.audit.record("stop_ratcheted", {"symbol": self.symbol, "from": round(current.stop, 5),
                                                 "to": round(nxt.stop, 5),
                                                 "locked_r": round(nxt.locked_r(), 3)})
        else:
            notes.append("broker refused stop amendment")

    def _entry_block_reason(self, bars: Sequence[Bar], open_syms: dict[str, Position]) -> str | None:
        """Why no new entry may be considered on this bar, or None if one may."""
        assert self.state is not None
        if not self.state.trading_allowed:
            return f"no entries: {self.state.status.value}"
        if self.symbol in open_syms:
            return "position open - managing"
        if len(bars) < self.warmup:
            return f"warming up ({len(bars)}/{self.warmup})"
        return None

    def _decide(self, bars: Sequence[Bar], funding_rates: Sequence[float] | None) -> AggregatedDecision:
        decision = self.aggregator.aggregate(self.symbol, bars, funding_rates)
        self.last_decision = decision
        if decision.signal is None and decision.rejected:
            self.audit.record("signal_vetoed", {"symbol": self.symbol, "summary": decision.summary,
                                                "er": decision.regime.efficiency_ratio})
        return decision

    def _compliance_veto(self, sig: Signal, compliance: ComplianceDecision | None, now: datetime) -> str | None:
        if compliance is None or compliance.allow_new_orders:
            return None
        self.audit.record("compliance_veto", {"signal": sig.strategy, "side": sig.side.value,
                                              "reasons": list(compliance.reasons), "now": now.isoformat()})
        return f"compliance veto: {'; '.join(compliance.reasons)}"

    def _prepare_order(self, sig: Signal) -> tuple[BracketLevels, PositionSize] | str:
        """Hard SL/TP from the risk engine, then a risk-engine size; a str is the veto reason."""
        assert self.state is not None
        try:
            levels = hard_bracket(sig.side, sig.entry, sig.stop, sig.take_profit, self.asset)
        except BracketError as exc:
            self.audit.record("bracket_rejected", {"signal": sig.strategy, "error": str(exc)})
            return f"bracket veto: {exc}"
        size = size_position(self.state, self.limits, self.asset, levels.entry, levels.stop, self.exposure())
        if not size.approved:
            self.audit.record("size_rejected", {"signal": sig.strategy, "reasons": list(size.reasons)})
            return f"sizing veto: {'; '.join(size.reasons)}"
        return levels, size

    def _submit(self, sig: Signal, decision: AggregatedDecision, levels: BracketLevels,
                size: PositionSize) -> tuple[str, BracketResult | None]:
        """Place the bracket; on acceptance start a ratchet from the actual fill."""
        try:
            order = BracketOrder(self.symbol, sig.side, size.units, levels.entry, levels.stop, levels.take_profit,
                                 strategy=sig.strategy)
        except OrderRejected as exc:  # invalid order: a veto, never an exception out of the decision loop
            self.audit.record("order_invalid", {"signal": sig.strategy, "error": str(exc)})
            return f"order invalid: {exc}", None
        try:
            result = self.broker.place_bracket(order)
        except (KillSwitchEngaged, OrderRejected, BracketRollbackError) as exc:
            self.audit.record("order_failed", {"client_id": order.client_id, "error": str(exc)})
            if isinstance(exc, BracketRollbackError) and "UNPROTECTED" in str(exc):
                self.kill_switch.trip(str(exc), source="executor")
            return f"order failed: {exc}", None

        self.audit.record("bracket_submitted", {
            "client_id": order.client_id, "strategy": sig.strategy, "side": sig.side.value, "units": size.units,
            "entry": round(levels.entry, 5), "stop": round(levels.stop, 5),
            "take_profit": round(levels.take_profit, 5), "levels_adjusted": levels.adjusted,
            "risk": round(size.risk_amount, 2),
            "binding_constraint": size.binding_constraint, "er": round(decision.regime.efficiency_ratio, 3),
            "accepted": result.accepted, "message": result.message})
        if result.accepted:
            fill = result.fill_price or levels.entry
            self.ratchets[self.symbol] = RatchetStop.open(sig.side, fill, levels.stop + (fill - levels.entry))
        return f"{decision.summary} -> {size.units} units", result

    def on_bar(self, bars: Sequence[Bar], funding_rates: Sequence[float] | None = None,
               now: datetime | None = None) -> CycleReport:
        """Run one decision cycle on the latest completed bar of ``bars``.

        ``now`` is the decision time used by session compliance; it defaults to the last
        bar's close time.
        """
        bar = bars[-1]
        now = now or self.decision_time(bars)
        notes: list[str] = []
        ev = self._mark(bar.ts, notes)
        if self.kill_switch.tripped:
            return self._report(bar, ev, notes, "halted: kill switch engaged")
        compliance = self._enforce_compliance(now, notes)
        if self.kill_switch.tripped:
            return self._report(bar, ev, notes, "halted: compliance flatten failed, kill switch engaged")

        open_syms = self._sync_ratchets()
        self._trail_stop(bars, bar, notes)
        blocked = self._entry_block_reason(bars, open_syms)
        if blocked:
            return self._report(bar, ev, notes, blocked)

        decision = self._decide(bars, funding_rates)
        if decision.signal is None:
            return self._report(bar, ev, notes, decision.summary)
        veto = self._compliance_veto(decision.signal, compliance, now)
        if veto:
            return self._report(bar, ev, notes, veto)
        prepared = self._prepare_order(decision.signal)
        if isinstance(prepared, str):
            return self._report(bar, ev, notes, prepared)
        outcome, result = self._submit(decision.signal, decision, *prepared)
        return self._report(bar, ev, notes, outcome, result)
