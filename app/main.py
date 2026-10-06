"""QuantMind-PropFirm-Orchestrator — terminal dashboard, CLI and audit trail.

Usage (from the project root)::

    python -m app.main research                 # Generator -> Critic round
    python -m app.main backtest --strategy trend_breakout --param channel=20
    python -m app.main dashboard                # live paper-replay dashboard
    python -m app.main shell                    # interactive session (step/run/kill/...)
    python -m app.main status                   # kill switch + audit tail
    python -m app.main kill --reason "news"     # trips a running engine via sentinel
    python -m app.main rearm --confirm REARM
    python -m app.main audit                    # verify the hash-chained audit log
    python -m app.main mt5-ping                 # check the MT5 ZeroMQ EA (needs QM_MT5_TOKEN)
    python -m app.main live --symbol XAUUSD --timeframe 1h   # DEMO-ONLY live loop
    python -m app.main mcp [--jsonrpc]          # start the MCP server

Everything here runs in PAPER mode. Live adapters are wired only through code that
constructs a :class:`~execution.cctv_executor.CexExecutor` or
:class:`~execution.mt5_bridge.Mt5Bridge` explicitly.
"""
from __future__ import annotations

import argparse
import cmd
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.coordinator import (
    CycleReport,
    ResearchCoordinator,
    TradingCoordinator,
)
from agent.critic import QCritic
from agent.generator import HypothesisGenerator, StrategyCandidate, candidate_id
from core.audit import AuditTrail
from core.backtest import BacktestConfig, run_backtest
from core.compliance import FRIDAY_FLATTEN_UTC, ComplianceGate, WeekendGuard, WeekendGuardConfig, parse_hhmm
from core.instruments import default_asset
from core.market_data import Bar, load_csv, resample, synthetic_ohlcv
from core.news_filter import NewsFilter, fetch_ff_calendar, load_calendar
from core.risk_engine import AssetSpec
from core.strategy_brain import (
    Strategy,
    TrendBreakoutStrategy,
    build_strategy,
)
from execution.base import PaperBroker
from execution.kill_switch import REARM_PHRASE, KillSwitch
from execution.live_runner import (
    DemoAccountGuardError,
    LiveConfig,
    LiveRunner,
    Mt5BarFeed,
    StepReport,
    asset_from_symbol_info,
    enforce_demo_account,
    parse_timeframe,
)
from execution.mt5_adapter import Mt5Adapter
from execution.mt5_bridge import (
    DEFAULT_REQ_ENDPOINT,
    HeartbeatMonitor,
    Mt5Error,
    Mt5Timeout,
    Mt5Transport,
    ZmqMt5Transport,
)

STATE_DIR = Path(os.environ.get("QM_STATE_DIR", ROOT / "state"))
SENTINEL = "KILL_SWITCH"
AUDIT_FILE = "audit.jsonl"

# ------------------------------------------------------------------------------ styling
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
if os.name == "nt":
    os.system("")  # enable VT100 escape processing on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass


def c(text: str, code: str) -> str:
    """Wrap ``text`` in an ANSI colour when the terminal supports it."""
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def gauge(fraction: float, width: int = 26) -> str:
    """Horizontal usage bar coloured green / yellow / red."""
    frac = max(0.0, min(1.0, fraction))
    filled = round(frac * width)
    bar = "█" * filled + "·" * (width - filled)
    return c(bar, "32" if frac < 0.5 else "33" if frac < 0.8 else "31")


def asset_for(symbol: str) -> AssetSpec:
    """Default contract specs for the demo symbols (see :mod:`core.instruments`)."""
    return default_asset(symbol)


def load_bars(args: argparse.Namespace) -> list[Bar]:
    """Bars from ``--csv`` or the synthetic generator, optionally session-filtered and resampled.

    ``--drop-weekends`` removes bars inside the weekend-guard window (Fri 19:30 - Sun 22:00
    UTC), so 24/7 proxies such as PAXG match the gold CFD session. ``--resample 4h``
    aggregates to a higher timeframe (applied after the weekend filter).
    """
    bars = load_csv(args.csv) if getattr(args, "csv", None) else synthetic_ohlcv(args.bars, seed=args.seed)
    if getattr(args, "drop_weekends", False):
        guard = WeekendGuard()
        bars = [b for b in bars if not guard.in_blackout(b.ts)]
    if getattr(args, "resample", None):
        bars = resample(bars, parse_timeframe(args.resample)[1])
    return bars


# ------------------------------------------------------------------------------ session
@dataclass
class PaperSession:
    """A paper-trading replay: bars are fed one at a time through the full pipeline."""

    symbol: str
    bars: list[Bar]
    coordinator: TradingCoordinator
    broker: PaperBroker
    kill_switch: KillSwitch
    audit: AuditTrail
    cursor: int = 0
    last: CycleReport | None = None
    strategies_note: str = ""

    @classmethod
    def build(cls, symbol: str, bars: list[Bar], state_dir: Path, research_fraction: float = 0.5,
              seed: int = 42, starting_cash: float = 100_000.0,
              compliance: ComplianceGate | None = None) -> PaperSession:
        """Research on the first part of ``bars``; replay the rest with promoted strategies."""
        audit = AuditTrail(state_dir / AUDIT_FILE)
        kill = KillSwitch(sentinel_path=state_dir / SENTINEL, audit=audit.record)
        asset = asset_for(symbol)
        broker = PaperBroker(starting_cash, unit_value=asset.contract_size * asset.quote_to_account, guard=kill)
        split = int(len(bars) * research_fraction)
        report = ResearchCoordinator(HypothesisGenerator(seed=seed), audit=audit).run(bars[:split], symbol)
        promoted = report.promoted[:2]
        strategies: list[Strategy] = [v.candidate.build() for v in promoted]
        note = ", ".join(f"{v.candidate.strategy_name}{dict(v.candidate.params)}" for v in promoted)
        if not strategies:
            strategies = [TrendBreakoutStrategy()]
            note = "NO candidate promoted - running default trend_breakout (paper only)"
            audit.record("fallback_strategy", {"reason": note})
        coord = TradingCoordinator(symbol, asset, broker, kill, strategies, audit=audit, compliance=compliance)
        # Pre-roll the research window so indicators are warm when replay begins.
        return cls(symbol, bars, coord, broker, kill, audit, cursor=split, strategies_note=note)

    @property
    def done(self) -> bool:
        """True when every bar has been replayed."""
        return self.cursor >= len(self.bars)

    def step(self) -> CycleReport | None:
        """Advance one bar: broker fills SL/TP on the bar, then the coordinator decides."""
        if self.done:
            return None
        bar = self.bars[self.cursor]
        self.broker.on_bar(self.symbol, bar.open, bar.high, bar.low, bar.close)
        window = self.bars[max(0, self.cursor + 1 - 400): self.cursor + 1]
        self.last = self.coordinator.on_bar(window)
        self.cursor += 1
        return self.last


def render(session: PaperSession) -> str:
    """Render the dashboard as a string."""
    coord = session.coordinator
    snap = coord.risk_snapshot()
    lines: list[str] = []
    ks = c(" KILL SWITCH ENGAGED ", "41;97") if session.kill_switch.tripped else c(" ARMED ", "42;30")
    bar = session.bars[session.cursor - 1] if session.cursor else session.bars[0]
    lines.append(c("QuantMind-PropFirm-Orchestrator", "1;36") + c("  [PAPER]", "33") +
                 f"  {session.symbol}  {bar.ts:%Y-%m-%d}  bar {session.cursor}/{len(session.bars)}  {ks}")
    lines.append(c("strategies: ", "2") + session.strategies_note)
    lines.append("─" * 96)
    if snap.get("status") != "uninitialised":
        dll_used = max(0.0, -snap["daily_pnl_pct"]) / snap["daily_limit_pct"]
        dd_used = snap["drawdown_pct"] / snap["drawdown_limit_pct"]
        status = snap["status"]
        status_txt = c(status.upper(), "32" if status == "active" else "31")
        lines.append(f"equity  {snap['equity']:>14,.2f}   HWM {snap['high_water_mark']:>14,.2f}   status {status_txt}")
        lines.append(f"daily   {snap['daily_pnl_pct']:>+13.2%}   {gauge(dll_used)} {dll_used:5.0%} of "
                     f"{snap['daily_limit_pct']:.1%} DLL   headroom ${snap['daily_headroom']:,.0f}")
        lines.append(f"drawdn  {snap['drawdown_pct']:>13.2%}   {gauge(dd_used)} {dd_used:5.0%} of "
                     f"{snap['drawdown_limit_pct']:.1%} max    headroom ${snap['drawdown_headroom']:,.0f}")
        lines.append(f"open risk ${snap['open_risk']:,.0f}   notional ${snap['open_notional']:,.0f}")
    dec = coord.last_decision
    if dec is not None:
        er = dec.regime.efficiency_ratio
        thr = coord.aggregator.regime_filter.config.er_threshold
        er_txt = c(f"{er:.2f}", "32" if er >= thr else "31")
        lines.append(f"regime  ER {er_txt} (min {thr:.2f})  {gauge(er, 20)}  vol {dec.regime.vol_regime.value}")
    comp = coord.last_compliance
    if comp is not None:
        state = c("OPEN", "32") if comp.allow_new_orders else c("BLOCKED", "31")
        lines.append(f"session {state}  {'; '.join(comp.reasons)[:80]}")
    lines.append("─" * 96)
    for p in snap.get("positions", []):
        lines.append(f"POSITION {p['side'].upper():5} {p['units']} @ {p['entry']:.2f}  stop {p['stop']:.2f}  "
                     f"tp {p['take_profit']:.2f}")
    if session.last is not None:
        lines.append(c("decision: ", "1") + session.last.decision)
        for n in session.last.notes:
            lines.append(c("  note: ", "33") + n)
    lines.append(c("audit trail (latest)", "1;36"))
    for rec in session.audit.tail(8):
        payload = json.dumps(rec.payload, default=str)
        lines.append(f"  #{rec.seq:<5} {rec.ts[11:19]} {rec.event:<20} {payload[:62]}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ commands
def cmd_research(args: argparse.Namespace) -> int:
    bars = load_bars(args)
    audit = AuditTrail(STATE_DIR / AUDIT_FILE)
    report = ResearchCoordinator(HypothesisGenerator(seed=args.seed), audit=audit,
                                 backtest=BacktestConfig(asset=asset_for(args.symbol))).run(
        bars, args.symbol, args.max_candidates, prior_trials=args.prior_trials)
    print(c(f"Research on {len(bars)} bars — {report.trials} trials (snooping hurdle applies to all)", "1"))
    print(f"{'':2}{'strategy':<15}{'params':<52}{'OOS n':>6}{'Lo t':>7}{'hurdle':>8}  verdict")
    for v in sorted(report.verdicts, key=lambda v: v.lo.t_stat, reverse=True):
        mark = c("✔", "32") if v.promoted else c("✘", "31")
        params = json.dumps(dict(v.candidate.params))
        why = "PROMOTED" if v.promoted else v.reasons[0]
        print(f"{mark} {v.candidate.strategy_name:<15}{params:<52}{v.oos_result.metrics.n_trades:>6}"
              f"{v.lo.t_stat:>7.2f}{v.hurdle:>8.2f}  {why}")
    print(c(f"\n{len(report.promoted)} promoted.", "1"))
    return 0


def _parse_params(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def cmd_backtest(args: argparse.Namespace) -> int:
    bars = load_bars(args)
    strat = build_strategy(args.strategy, _parse_params(args.param))
    for flt in (True, False):
        res = run_backtest(bars, strat, BacktestConfig(use_regime_filter=flt), args.symbol)
        label = "with KER gate   " if flt else "without KER gate"
        print(f"{label}: {json.dumps(res.metrics.to_dict())}  blocked={res.signals_blocked_by_regime} "
              f"breached={res.breached}")
        for ev in res.risk_events:
            print(c(f"   risk event: {ev}", "33"))
    return 0


def weekend_guard_from_args(args: argparse.Namespace) -> WeekendGuard:
    """Weekend guard with the Friday flatten time from ``--friday-flatten`` (default 19:30 UTC)."""
    return WeekendGuard(WeekendGuardConfig(flatten_time=getattr(args, "friday_flatten", FRIDAY_FLATTEN_UTC)))


def compliance_from_args(args: argparse.Namespace) -> ComplianceGate | None:
    """Build the session compliance gate from ``--weekend-guard`` / ``--news``."""
    weekend = weekend_guard_from_args(args) if getattr(args, "weekend_guard", False) else None
    news = NewsFilter(load_calendar(args.news)) if getattr(args, "news", None) else None
    return ComplianceGate(weekend, news) if (weekend or news) else None


def cmd_dashboard(args: argparse.Namespace) -> int:
    session = PaperSession.build(args.symbol, load_bars(args), STATE_DIR, seed=args.seed,
                                 compliance=compliance_from_args(args))
    try:
        while not session.done:
            session.step()
            if session.cursor % args.refresh == 0 or session.done:
                sys.stdout.write("\x1b[2J\x1b[H" if _COLOR else "\n")
                sys.stdout.write(render(session) + "\n")
                sys.stdout.flush()
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\ninterrupted - positions left as-is in the paper broker (use `kill` to flatten)")
    return 0


class Shell(cmd.Cmd):
    """Interactive paper-trading shell."""

    intro = "QuantMind shell (paper). Commands: step [n] | run | status | kill <reason> | rearm | audit | quit"
    prompt = "qm> "

    def __init__(self, session: PaperSession) -> None:
        super().__init__()
        self.session = session

    def do_step(self, arg: str) -> None:
        """step [n] — replay n bars (default 1)."""
        for _ in range(int(arg or 1)):
            if self.session.step() is None:
                print("end of data")
                break
        print(render(self.session))

    def do_run(self, _arg: str) -> None:
        """run — replay all remaining bars."""
        while self.session.step() is not None:
            pass
        print(render(self.session))

    def do_status(self, _arg: str) -> None:
        """status — show the dashboard."""
        print(render(self.session))

    def do_risk(self, _arg: str) -> None:
        """risk — raw risk metrics as JSON."""
        print(json.dumps(self.session.coordinator.risk_snapshot(), indent=2, default=str))

    def do_kill(self, arg: str) -> None:
        """kill <reason> — cancel all orders and flatten all positions."""
        rep = self.session.kill_switch.trip(arg or "manual shell kill", source="shell")
        print(c(f"KILL SWITCH: cancelled={rep.cancelled} flattened={rep.flattened} errors={list(rep.errors)}", "31"))

    def do_rearm(self, arg: str) -> None:
        """rearm REARM — re-enable trading (books must be flat)."""
        try:
            self.session.kill_switch.rearm(arg.strip())
            print(c("re-armed", "32"))
        except PermissionError as exc:
            print(c(str(exc), "31"))

    def do_audit(self, _arg: str) -> None:
        """audit — verify the audit chain."""
        ok, bad = self.session.audit.verify()
        print(c("audit chain OK", "32") if ok else c(f"audit chain BROKEN at seq {bad}", "31"))

    def do_quit(self, _arg: str) -> bool:
        """quit — leave the shell."""
        return True

    do_exit = do_quit
    do_EOF = do_quit


def cmd_shell(args: argparse.Namespace) -> int:
    Shell(PaperSession.build(args.symbol, load_bars(args), STATE_DIR, seed=args.seed,
                             compliance=compliance_from_args(args))).cmdloop()
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    sentinel = STATE_DIR / SENTINEL
    if sentinel.exists():
        print(c(f"KILL SWITCH ENGAGED: {sentinel.read_text(encoding='utf-8')}", "31"))
    else:
        print(c("kill switch armed", "32"))
    for rec in AuditTrail(STATE_DIR / AUDIT_FILE).tail(15):
        print(f"#{rec.seq:<5} {rec.ts} {rec.event:<22} {json.dumps(rec.payload, default=str)[:90]}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    KillSwitch.write_sentinel(STATE_DIR / SENTINEL, args.reason, source="cli")
    AuditTrail(STATE_DIR / AUDIT_FILE).record("kill_requested_cli", {"reason": args.reason})
    print(c("Kill sentinel written. Running engines flatten on their next cycle and stay halted.", "31"))
    return 0


def cmd_rearm(args: argparse.Namespace) -> int:
    try:
        KillSwitch(sentinel_path=STATE_DIR / SENTINEL,
                   audit=AuditTrail(STATE_DIR / AUDIT_FILE).record).rearm(args.confirm)
    except PermissionError as exc:
        print(c(str(exc), "31"))
        return 1
    print(c("Sentinel cleared. (An engine already halted in memory must be re-armed from its own shell.)", "32"))
    return 0


MT5_TOKEN_ENV = "QM_MT5_TOKEN"


def _is_multiple(value: float, step: float) -> bool:
    ratio = value / step
    return ratio >= 1 - 1e-9 and abs(ratio - round(ratio)) < 1e-6


def mt5_ping_report(transport: Mt5Transport, symbol: str, asset: AssetSpec) -> tuple[bool, list[str]]:
    """Query the EA and check the terminal against our :class:`AssetSpec` for ``symbol``.

    Returns ``(ok, lines)``. ``ok`` is False if trading is not permitted or our price
    tick would produce prices the broker rejects. Lot/contract differences are warnings.
    """
    lines: list[str] = []
    ok = True

    def line(status: str, text: str) -> None:
        colour = {"PASS": "32", "WARN": "33", "FAIL": "31", "INFO": "36"}[status]
        lines.append(f"{c(status, colour)}  {text}")

    t0 = time.perf_counter()
    ping = transport.ping()
    rtt = (time.perf_counter() - t0) * 1000
    line("PASS", f"PING round trip {rtt:.1f} ms")
    if ping.get("trade_allowed"):
        line("PASS", "terminal reports trading allowed")
    else:
        ok = False
        line("FAIL", f"trading not allowed: {ping.get('reason') or 'see terminal'}")
    if ping.get("weekend_blackout"):
        line("INFO", "EA weekend guard active - SEND_BRACKET will be refused until Sun 22:00 UTC")

    acc = transport.account_info()
    line("INFO", f"account {acc.get('login')} @ {acc.get('server')}  equity {acc.get('equity')} "
                 f"{acc.get('currency', '')}  leverage 1:{acc.get('leverage')}")
    if not acc.get("trade_expert", True):
        ok = False
        line("FAIL", "expert (algorithmic) trading is disabled for this account/terminal")

    sym = transport.symbol_info(symbol)
    broker_tick = float(sym.get("trade_tick_size") or sym.get("point") or 0.0)
    line("INFO", f"{symbol}: digits {sym.get('digits')}  point {sym.get('point')}  tick {broker_tick}  "
                 f"bid/ask {sym.get('bid')}/{sym.get('ask')}  stops level {sym.get('trade_stops_level')} pts")
    if asset.price_tick <= 0:
        line("WARN", "our price_tick is 0: SL/TP are only rounded to digits at send time (not conservatively)")
    elif broker_tick > 0 and _is_multiple(asset.price_tick, broker_tick):
        line("PASS", f"price_tick {asset.price_tick} is a multiple of broker tick {broker_tick} "
                     "- rounded SL/TP are valid broker prices")
    else:
        ok = False
        line("FAIL", f"price_tick {asset.price_tick} is NOT a multiple of broker tick {broker_tick} - fix AssetSpec")
    for name, ours, theirs in (("lot step", asset.lot_step, sym.get("volume_step")),
                               ("min lot", asset.min_lot, sym.get("volume_min")),
                               ("contract size", asset.contract_size, sym.get("trade_contract_size"))):
        if theirs is None:
            continue
        if abs(float(theirs) - ours) > 1e-9:
            line("WARN", f"{name}: ours {ours} vs broker {theirs} - pass the broker value in AssetSpec")
        else:
            line("PASS", f"{name} {ours} matches broker")
    stops_pts = float(sym.get("trade_stops_level") or 0)
    if stops_pts and asset.min_stop_distance < stops_pts * float(sym.get("point") or 0):
        line("WARN", f"broker stops level is {stops_pts * float(sym.get('point') or 0)} in price; "
                     f"set AssetSpec.min_stop_distance to at least that")
    return ok, lines


def cmd_mt5_ping(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get(MT5_TOKEN_ENV, "")
    if not token:
        print(c(f"No token: set {MT5_TOKEN_ENV} (same value as the EA's InpAuthToken) or pass --token", "31"))
        return 2
    transport = ZmqMt5Transport(args.endpoint, token=token, timeout_ms=args.timeout_ms, retries=0)
    try:
        ok, lines = mt5_ping_report(transport, args.symbol, asset_for(args.symbol))
    except Mt5Timeout:
        print(c(f"EA not reachable at {args.endpoint} within {args.timeout_ms} ms. Check that:", "31"))
        for tip in ("MT5 is running and ZeroMQ_Listener is attached to a chart (smiley face, not a sad face)",
                    "'Allow DLL imports' and 'Algo Trading' are enabled (Tools > Options > Expert Advisors)",
                    "the Experts tab shows no 'failed to bind' / DLL errors",
                    f"--endpoint matches the EA's InpRepEndpoint (default {DEFAULT_REQ_ENDPOINT})"):
            print(f"  - {tip}")
        return 1
    except Mt5Error as exc:
        print(c(f"EA answered with an error: {exc}", "31"))
        if "authentication" in str(exc):
            print(f"  - {MT5_TOKEN_ENV} / --token must equal the EA's InpAuthToken")
        return 1
    except ImportError:
        print(c("pyzmq is not installed: pip install pyzmq", "31"))
        return 2
    finally:
        transport.close()
    print("\n".join(lines))
    print(c("MT5 link OK - ready for demo trading" if ok else "MT5 link reachable but NOT ready (see FAIL lines)",
            "1;32" if ok else "1;31"))
    return 0 if ok else 1


def select_live_strategies(args: argparse.Namespace, bars: list[Bar], asset: AssetSpec,
                           audit: AuditTrail) -> tuple[list[Strategy], str] | None:
    """Pick what the live loop may trade. Only Critic-promoted strategies, unless overridden.

    With ``--strategy`` the named strategy is reviewed by the Critic on the broker's own
    history; without it a full research round runs. Returns None (refuse to start) when
    nothing is promoted and ``--allow-unpromoted`` was not given.
    """
    cfg = BacktestConfig(asset=asset)
    if args.strategy:
        params = _parse_params(args.param)
        cand = StrategyCandidate(candidate_id(args.strategy, params), args.strategy, params, "operator choice", 1)
        verdict = QCritic().review(cand, bars, cfg, trials=1, symbol=args.symbol)
        audit.record("live_strategy_review", verdict.to_dict())
        label = f"{args.strategy}{params}"
        if verdict.promoted:
            return [cand.build()], f"{label} (Critic: PROMOTED, t={verdict.lo.t_stat:.2f})"
        why = "; ".join(verdict.reasons)
        if args.allow_unpromoted:
            return [cand.build()], f"{label} (NOT promoted: {why}) - running by operator override"
        print(c(f"Refusing to trade {label}: Critic rejected it on MT5 history: {why}", "31"))
        return None
    report = ResearchCoordinator(HypothesisGenerator(seed=args.seed), audit=audit, backtest=cfg).run(bars, args.symbol)
    if report.promoted:
        best = report.promoted[:2]
        return ([v.candidate.build() for v in best],
                ", ".join(f"{v.candidate.strategy_name}{dict(v.candidate.params)}" for v in best))
    if args.allow_unpromoted:
        return [TrendBreakoutStrategy()], "NO candidate promoted - default trend_breakout by operator override"
    print(c(f"Refusing to start: none of {report.trials} candidates passed the Critic on MT5 history. "
            "Use --strategy/--param for a specific setup, or --allow-unpromoted to override (demo only).", "31"))
    return None


def _print_step(rep: StepReport) -> None:
    stamp = rep.now.strftime("%Y-%m-%d %H:%M:%S")
    if rep.cycle is not None:
        order = ""
        if rep.cycle.order is not None:
            order = c(f"  ORDER {'OK' if rep.cycle.order.accepted else 'REJECTED'}: {rep.cycle.order.message}",
                      "32" if rep.cycle.order.accepted else "31")
        print(f"{stamp}  [{rep.state.value}] bar {rep.cycle.ts:%Y-%m-%d %H:%M} equity {rep.cycle.equity:,.2f}  "
              f"{rep.cycle.decision}{order}")
    for note in rep.notes:
        print(f"{stamp}  [{rep.state.value}] {note}")


def cmd_live(args: argparse.Namespace) -> int:
    """Continuous demo-only loop against the MT5 EA."""
    token = args.token or os.environ.get(MT5_TOKEN_ENV, "")
    if not token:
        print(c(f"No token: set {MT5_TOKEN_ENV} (same value as the EA's InpAuthToken) or pass --token", "31"))
        return 2
    parse_timeframe(args.timeframe)  # validate early
    transport = ZmqMt5Transport(args.endpoint, token=token, timeout_ms=3_000, retries=1)
    adapter: Mt5Adapter | None = None
    try:
        account = transport.account_info()
        enforce_demo_account(account)  # before any other request
        print(c(f"Account {account.get('login')} @ {account.get('server')}: {account.get('trade_mode_name')} "
                "- demo guard passed", "32"))
        asset = asset_from_symbol_info(args.symbol, transport.symbol_info(args.symbol), account)
        audit = AuditTrail(STATE_DIR / AUDIT_FILE)
        feed = Mt5BarFeed(transport, args.symbol, args.timeframe, history=args.history)
        research_bars = feed.fetch(args.research_bars)
        print(f"Loaded {len(research_bars)} closed {feed.code} bars from MT5 for strategy selection...")
        chosen = select_live_strategies(args, research_bars, asset, audit)
        if chosen is None:
            return 1
        strategies, note = chosen
        print(c(f"Strategies: {note}", "36"))

        news: NewsFilter | None = None
        if args.news == "auto":
            try:
                news = NewsFilter(fetch_ff_calendar())
            except Exception as exc:  # noqa: BLE001 - network failure must be explicit
                print(c(f"Could not download the news calendar ({exc}). Pass --news FILE or --news off.", "31"))
                return 1
        elif args.news != "off":
            news = NewsFilter(load_calendar(args.news))
        if news is None:
            print(c("WARNING: news filter disabled (--news off)", "33"))
        gate = ComplianceGate(weekend_guard_from_args(args), news)
        print(f"Weekend guard: flatten Fri {args.friday_flatten:%H:%M} UTC, entries blocked until Sun 22:00 UTC")

        kill = KillSwitch(sentinel_path=STATE_DIR / SENTINEL, audit=audit.record)
        monitor = HeartbeatMonitor(transport)
        adapter = Mt5Adapter(transport, monitor=monitor, kill_guard=kill)
        adapter.connect()
        coord = TradingCoordinator(args.symbol, asset, adapter, kill, strategies, audit=audit, compliance=gate)
        runner = LiveRunner(coord, feed, transport, kill, monitor,
                            LiveConfig(poll_seconds=args.poll_seconds, max_history=args.history), audit)
        print(c(f"LIVE (DEMO) loop: {args.symbol} {feed.code}, polling every {args.poll_seconds}s. "
                "Ctrl+C stops the loop (positions keep their SL/TP); `python -m app.main kill` flattens.", "1"))
        runner.run(max_steps=args.max_steps, on_step=_print_step)
        return 0
    except DemoAccountGuardError as exc:
        print(c(str(exc), "1;41;97"))
        return 3
    except Mt5Timeout:
        print(c(f"EA not reachable at {args.endpoint}. Run `python -m app.main mt5-ping` for the checklist.", "31"))
        return 1
    except Mt5Error as exc:
        print(c(f"MT5 error: {exc}", "31"))
        return 1
    finally:
        if adapter is not None:
            adapter.close()
        transport.close()


def cmd_audit(_args: argparse.Namespace) -> int:
    ok, bad = AuditTrail(STATE_DIR / AUDIT_FILE).verify()
    print(c("audit chain OK", "32") if ok else c(f"audit chain BROKEN at seq {bad}", "31"))
    return 0 if ok else 1


def cmd_mcp(args: argparse.Namespace) -> int:
    from mcp_server.mcp_bridge import main as mcp_main

    return mcp_main(["--jsonrpc"] if args.jsonrpc else [])


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all subcommands."""
    p = argparse.ArgumentParser(prog="quantmind", description="Prop-firm orchestrator (paper mode CLI)")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp: argparse.ArgumentParser, bars: int = 1_500) -> None:
        sp.add_argument("--symbol", default="XAUUSD")
        sp.add_argument("--bars", type=int, default=bars)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--csv", help="CSV with ts,open,high,low,close[,volume]")
        sp.add_argument("--drop-weekends", action="store_true", help="drop bars in the Fri 19:30-Sun 22:00 UTC window")
        sp.add_argument("--resample", help="aggregate bars to a timeframe, e.g. 4h")

    sp = sub.add_parser("research", help="run a Generator -> Critic round")
    data_args(sp)
    sp.add_argument("--max-candidates", type=int, default=None)
    sp.add_argument("--prior-trials", type=int, default=0,
                    help="tests already run in this research family (raises the snooping hurdle)")
    sp.set_defaults(func=cmd_research)

    sp = sub.add_parser("backtest", help="backtest one strategy with and without the KER gate")
    data_args(sp)
    sp.add_argument("--strategy", default="trend_breakout")
    sp.add_argument("--param", action="append", default=[], help="key=value (repeatable)")
    sp.set_defaults(func=cmd_backtest)

    def friday_flatten_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--friday-flatten", type=parse_hhmm, default=FRIDAY_FLATTEN_UTC, metavar="HH:MM",
                        help=f"Friday auto-flatten time, UTC (default {FRIDAY_FLATTEN_UTC:%H:%M})")

    def compliance_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--weekend-guard", action="store_true",
                        help="flatten Friday (see --friday-flatten) and block entries until Sun 22:00 UTC")
        friday_flatten_arg(sp)
        sp.add_argument("--news", help="news calendar (.json Forex-Factory format or .csv) for blackouts")

    sp = sub.add_parser("dashboard", help="live paper-replay dashboard")
    data_args(sp, 2_400)
    compliance_args(sp)
    sp.add_argument("--refresh", type=int, default=1, help="render every N bars")
    sp.add_argument("--delay", type=float, default=0.05, help="seconds between renders")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("shell", help="interactive paper session")
    data_args(sp, 2_400)
    compliance_args(sp)
    sp.set_defaults(func=cmd_shell)

    sub.add_parser("status", help="kill-switch state and audit tail").set_defaults(func=cmd_status)
    sp = sub.add_parser("kill", help="trip the kill switch of running engines")
    sp.add_argument("--reason", default="manual CLI kill")
    sp.set_defaults(func=cmd_kill)
    sp = sub.add_parser("rearm", help="clear the kill sentinel")
    sp.add_argument("--confirm", required=True, help=f"must be exactly {REARM_PHRASE}")
    sp.set_defaults(func=cmd_rearm)
    sub.add_parser("audit", help="verify the hash-chained audit trail").set_defaults(func=cmd_audit)
    sp = sub.add_parser("mt5-ping", help="check connectivity and symbol specs of the MT5 ZeroMQ EA")
    sp.add_argument("--endpoint", default=DEFAULT_REQ_ENDPOINT)
    sp.add_argument("--token", help=f"shared secret (default: ${MT5_TOKEN_ENV})")
    sp.add_argument("--symbol", default="XAUUSD")
    sp.add_argument("--timeout-ms", type=int, default=2_000)
    sp.set_defaults(func=cmd_mt5_ping)

    sp = sub.add_parser("live", help="continuous DEMO-ONLY trading loop against the MT5 EA")
    sp.add_argument("--symbol", default="XAUUSD")
    sp.add_argument("--timeframe", default="1h", help="1m 5m 15m 30m 1h 4h 1d")
    sp.add_argument("--endpoint", default=DEFAULT_REQ_ENDPOINT)
    sp.add_argument("--token", help=f"shared secret (default: ${MT5_TOKEN_ENV})")
    sp.add_argument("--poll-seconds", type=float, default=5.0)
    sp.add_argument("--history", type=int, default=1_000, help="bars kept for indicators")
    sp.add_argument("--research-bars", type=int, default=5_000, help="MT5 bars used for strategy selection")
    sp.add_argument("--strategy", help="trade this strategy (still reviewed by the Critic)")
    sp.add_argument("--param", action="append", default=[], help="key=value (repeatable)")
    sp.add_argument("--allow-unpromoted", action="store_true",
                    help="trade even if the Critic rejects it (demo accounts only; logged)")
    sp.add_argument("--news", default="auto", help="'auto' (download weekly calendar), a file, or 'off'")
    friday_flatten_arg(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_live)
    sp = sub.add_parser("mcp", help="start the MCP server")
    sp.add_argument("--jsonrpc", action="store_true")
    sp.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
