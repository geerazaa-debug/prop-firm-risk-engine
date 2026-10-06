"""Hard Risk Engine — pure math, no I/O, no UI dependencies.

This module is the single source of truth for whether the account may trade and
how large any new position may be. It encodes prop-firm challenge rules with a
wide safety margin:

* **Daily Loss Limit (DLL)** — 2.5% of the equity at the start of the trading day
  (the firm's own limit is typically 5%). Touching it freezes trading and demands a
  flatten for the rest of that day.
* **Maximum Trailing Drawdown** — 6.0% below the equity high-water mark (firm limit
  typically 10%). Touching it terminates the account: trading never resumes without
  a human creating a new ``AccountState``.
* **Leverage-aware position sizing** — fixed-fractional risk, scaled down for highly
  leveraged assets, capped by notional exposure, margin utilisation and the
  *remaining* DLL / drawdown headroom so no single stop-out can breach a limit.
* **Immutable ratchet trailing stop** — a non-mutating state machine whose stop can
  only move in the trade's favour (breakeven lock at +1R, decaying ATR multiple).

All state objects are frozen dataclasses; every transition returns a new object.
The hard ceilings below are module constants: a ``RiskLimits`` instance may be
*tighter* than them but construction fails if it tries to be looser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import Enum
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------------------
# Non-negotiable ceilings. RiskLimits may tighten these, never loosen them.
# --------------------------------------------------------------------------------------
HARD_DAILY_LOSS_LIMIT_PCT: Final[float] = 0.025
HARD_MAX_TRAILING_DRAWDOWN_PCT: Final[float] = 0.06
HARD_MAX_RISK_PER_TRADE_PCT: Final[float] = 0.01
HARD_MIN_BUDGET_FLOOR_PCT: Final[float] = 0.0025

# Reference values of typical prop-firm rules, used only for reporting the buffer.
PROP_FIRM_DAILY_LOSS_PCT: Final[float] = 0.05
PROP_FIRM_MAX_LOSS_PCT: Final[float] = 0.10

_EPS: Final[float] = 1e-12


class RiskConfigError(ValueError):
    """Raised when a risk configuration would loosen a hard ceiling or is incoherent."""


class Side(str, Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> Side:
        """The closing direction."""
        return Side.SHORT if self is Side.LONG else Side.LONG


class AccountStatus(str, Enum):
    """Lifecycle of the account from the risk engine's point of view."""

    ACTIVE = "active"
    DAILY_FROZEN = "daily_frozen"  # DLL hit: no trading until the next trading day
    BREACHED = "breached"  # trailing drawdown hit: terminal


class RiskAction(str, Enum):
    """What the caller must do after a mark-to-market evaluation."""

    NONE = "none"
    FREEZE_AND_FLATTEN = "freeze_and_flatten"
    TERMINATE_AND_FLATTEN = "terminate_and_flatten"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Risk configuration. Every field is validated against the hard ceilings.

    Attributes:
        daily_loss_limit_pct: Max loss vs. day-start equity before a daily freeze.
        max_trailing_drawdown_pct: Max loss vs. equity high-water mark before termination.
        risk_per_trade_pct: Fixed-fractional risk budget of a single new trade.
        max_open_risk_pct: Cap on the summed stop-risk of all open positions. Must be
            strictly below the DLL so that every open stop being hit on the same day
            still cannot breach the daily limit on its own.
        max_notional_exposure_multiple: Cap on gross notional as a multiple of equity.
        reference_leverage: Leverage at which risk is *not* scaled down. Assets offering
            more leverage than this get their risk budget scaled by
            ``reference_leverage / asset_leverage`` (they gap and liquidate faster).
        max_margin_utilisation: Fraction of equity that may be posted as margin.
        headroom_buffer: A new trade's worst-case loss may consume at most this fraction
            of the *spendable* DLL / drawdown headroom (after open risk and the floor).
        min_budget_floor_pct: Reserved cushion, as a fraction of equity, kept between
            equity and each loss limit. Once the remaining daily or overall loss budget
            (headroom minus open risk) falls below it, no new trades are opened, and it is
            never spendable by a new trade — so a full stop-out still leaves at least this
            cushion. Default 0.5% = one standard trade; may not be set below 0.25%.
        min_trade_risk_fraction: A trade whose allowed risk is below this fraction of a
            standard trade (``risk_per_trade_pct * equity``) is not opened. This stops the
            engine from creeping towards the floor with ever-smaller micro-lot trades.
        day_rollover_tz: IANA zone whose midnight starts a new trading day
            (FTMO uses ``"Europe/Prague"``).
    """

    daily_loss_limit_pct: float = HARD_DAILY_LOSS_LIMIT_PCT
    max_trailing_drawdown_pct: float = HARD_MAX_TRAILING_DRAWDOWN_PCT
    risk_per_trade_pct: float = 0.005
    max_open_risk_pct: float = 0.015
    max_notional_exposure_multiple: float = 3.0
    reference_leverage: float = 30.0
    max_margin_utilisation: float = 0.5
    headroom_buffer: float = 0.8
    min_budget_floor_pct: float = 0.005
    min_trade_risk_fraction: float = 0.25
    day_rollover_tz: str = "UTC"

    def __post_init__(self) -> None:
        checks: list[tuple[bool, str]] = [
            (0 < self.daily_loss_limit_pct <= HARD_DAILY_LOSS_LIMIT_PCT,
             f"daily_loss_limit_pct must be in (0, {HARD_DAILY_LOSS_LIMIT_PCT}]"),
            (0 < self.max_trailing_drawdown_pct <= HARD_MAX_TRAILING_DRAWDOWN_PCT,
             f"max_trailing_drawdown_pct must be in (0, {HARD_MAX_TRAILING_DRAWDOWN_PCT}]"),
            (0 < self.risk_per_trade_pct <= HARD_MAX_RISK_PER_TRADE_PCT,
             f"risk_per_trade_pct must be in (0, {HARD_MAX_RISK_PER_TRADE_PCT}]"),
            (self.risk_per_trade_pct <= self.max_open_risk_pct < self.daily_loss_limit_pct,
             "require risk_per_trade_pct <= max_open_risk_pct < daily_loss_limit_pct"),
            (self.daily_loss_limit_pct <= self.max_trailing_drawdown_pct,
             "daily_loss_limit_pct must not exceed max_trailing_drawdown_pct"),
            (self.max_notional_exposure_multiple > 0, "max_notional_exposure_multiple must be > 0"),
            (self.reference_leverage >= 1, "reference_leverage must be >= 1"),
            (0 < self.max_margin_utilisation <= 1, "max_margin_utilisation must be in (0, 1]"),
            (0 < self.headroom_buffer <= 1, "headroom_buffer must be in (0, 1]"),
            (HARD_MIN_BUDGET_FLOOR_PCT <= self.min_budget_floor_pct
             < self.daily_loss_limit_pct - self.risk_per_trade_pct,
             (f"min_budget_floor_pct must be in [{HARD_MIN_BUDGET_FLOOR_PCT}, "
              "daily_loss_limit_pct - risk_per_trade_pct)")),
            (0 < self.min_trade_risk_fraction <= 1, "min_trade_risk_fraction must be in (0, 1]"),
        ]
        for ok, message in checks:
            if not ok:
                raise RiskConfigError(message)
        _zone(self.day_rollover_tz)  # fail fast on an unknown zone


@lru_cache(maxsize=32)
def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception as exc:  # ZoneInfoNotFoundError, ValueError
        raise RiskConfigError(f"unknown day_rollover_tz {name!r} (install 'tzdata' on Windows)") from exc


def trading_day_of(ts: datetime, tz_name: str = "UTC") -> date:
    """Return the trading day a timestamp belongs to. Naive timestamps are treated as UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(_zone(tz_name)).date()


# --------------------------------------------------------------------------------------
# Account state machine
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AccountState:
    """Immutable snapshot of the account as seen by the risk engine."""

    trading_day: date
    initial_balance: float
    day_start_equity: float
    high_water_mark: float
    equity: float
    status: AccountStatus = AccountStatus.ACTIVE
    breach_reason: str | None = None

    @classmethod
    def open(cls, equity: float, ts: datetime, limits: RiskLimits) -> AccountState:
        """Create the initial state for a fresh account/challenge."""
        if equity <= 0 or not math.isfinite(equity):
            raise ValueError("starting equity must be a positive finite number")
        return cls(
            trading_day=trading_day_of(ts, limits.day_rollover_tz),
            initial_balance=equity,
            day_start_equity=equity,
            high_water_mark=equity,
            equity=equity,
        )

    # ---- derived metrics ------------------------------------------------------------
    def daily_floor(self, limits: RiskLimits) -> float:
        """Equity level at which the DLL triggers today."""
        return self.day_start_equity * (1.0 - limits.daily_loss_limit_pct)

    def drawdown_floor(self, limits: RiskLimits) -> float:
        """Equity level at which the trailing drawdown triggers."""
        return self.high_water_mark * (1.0 - limits.max_trailing_drawdown_pct)

    def daily_pnl_pct(self) -> float:
        """Today's PnL as a fraction of day-start equity (negative = loss)."""
        return (self.equity - self.day_start_equity) / self.day_start_equity

    def drawdown_pct(self) -> float:
        """Current drawdown from the high-water mark as a positive fraction."""
        return max(0.0, (self.high_water_mark - self.equity) / self.high_water_mark)

    def daily_headroom(self, limits: RiskLimits) -> float:
        """Dollars that can still be lost today before the DLL triggers."""
        return max(0.0, self.equity - self.daily_floor(limits))

    def drawdown_headroom(self, limits: RiskLimits) -> float:
        """Dollars that can still be lost before the trailing drawdown triggers."""
        return max(0.0, self.equity - self.drawdown_floor(limits))

    @property
    def trading_allowed(self) -> bool:
        """True only while ACTIVE."""
        return self.status is AccountStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class RiskEvaluation:
    """Result of :func:`mark_to_market`: the new state plus the action the caller owes."""

    state: AccountState
    action: RiskAction
    message: str = ""


def mark_to_market(state: AccountState, equity: float, ts: datetime, limits: RiskLimits) -> RiskEvaluation:
    """Advance the account state machine with a fresh equity mark.

    Order of operations:

    1. A BREACHED account stays breached forever (equity is still recorded).
    2. If ``ts`` falls on a later trading day, the day rolls: day-start equity becomes
       the last mark of the previous day and a DAILY_FROZEN account is re-activated.
    3. The high-water mark ratchets up.
    4. Trailing drawdown is checked first (terminal), then the DLL (daily freeze).
       Touching a floor counts as a breach.

    Args:
        state: Previous state.
        equity: Current account equity (balance + floating PnL).
        ts: Timestamp of the mark.
        limits: Risk limits.

    Returns:
        A :class:`RiskEvaluation`. ``action`` is non-NONE only on the transition into a
        frozen/breached status, so callers flatten exactly once per event.
    """
    if not math.isfinite(equity):
        # A NaN/inf equity feed is treated as the worst case: terminate and flatten.
        breached = replace(state, status=AccountStatus.BREACHED, breach_reason="non-finite equity mark")
        return RiskEvaluation(breached, RiskAction.TERMINATE_AND_FLATTEN, "non-finite equity mark")

    if state.status is AccountStatus.BREACHED:
        return RiskEvaluation(replace(state, equity=equity), RiskAction.NONE, "account breached (terminal)")

    day = trading_day_of(ts, limits.day_rollover_tz)
    if day > state.trading_day:
        state = replace(
            state,
            trading_day=day,
            day_start_equity=state.equity,
            status=AccountStatus.ACTIVE,
            breach_reason=None,
        )

    hwm = max(state.high_water_mark, equity)
    state = replace(state, equity=equity, high_water_mark=hwm)

    if equity <= state.drawdown_floor(limits) + _EPS:
        reason = (f"trailing drawdown {state.drawdown_pct():.2%} >= "
                  f"{limits.max_trailing_drawdown_pct:.2%} (HWM {hwm:,.2f})")
        return RiskEvaluation(
            replace(state, status=AccountStatus.BREACHED, breach_reason=reason),
            RiskAction.TERMINATE_AND_FLATTEN,
            reason,
        )

    if equity <= state.daily_floor(limits) + _EPS:
        if state.status is AccountStatus.DAILY_FROZEN:
            return RiskEvaluation(state, RiskAction.NONE, "daily limit already frozen")
        reason = (f"daily loss {-state.daily_pnl_pct():.2%} >= "
                  f"{limits.daily_loss_limit_pct:.2%} (day start {state.day_start_equity:,.2f})")
        return RiskEvaluation(
            replace(state, status=AccountStatus.DAILY_FROZEN, breach_reason=reason),
            RiskAction.FREEZE_AND_FLATTEN,
            reason,
        )

    return RiskEvaluation(state, RiskAction.NONE, "")


# --------------------------------------------------------------------------------------
# Pre-trade checks and position sizing
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Contract specification needed to translate price distance into account currency.

    Attributes:
        symbol: Instrument name.
        leverage: Maximum leverage the broker offers on this asset (e.g. 100 for FX majors).
        contract_size: Units of the underlying per 1.0 lot/contract (100 oz for XAUUSD).
        lot_step: Minimum volume increment.
        min_lot: Minimum tradable volume.
        quote_to_account: Multiplier converting quote-currency PnL into account currency.
        commission_bps: Estimated commission per side, in basis points of notional.
        slippage_bps: Estimated adverse slippage per side, in basis points of price.
            Both default to a conservative non-zero estimate so that forgetting to set
            them over-estimates risk rather than under-estimating it.
        price_tick: Broker price increment (0 = no rounding). SL/TP are rounded to it.
        min_stop_distance: Broker minimum distance of SL/TP from entry, in price
            (MT5 ``SYMBOL_TRADE_STOPS_LEVEL * point``). 0 = no constraint.
    """

    symbol: str
    leverage: float = 1.0
    contract_size: float = 1.0
    lot_step: float = 0.001
    min_lot: float = 0.001
    quote_to_account: float = 1.0
    commission_bps: float = 2.0
    slippage_bps: float = 1.0
    price_tick: float = 0.0
    min_stop_distance: float = 0.0

    def __post_init__(self) -> None:
        numeric = (self.leverage, self.contract_size, self.lot_step, self.min_lot, self.quote_to_account,
                   self.commission_bps, self.slippage_bps, self.price_tick, self.min_stop_distance)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numeric):
            # NaN slips past every `<` check below and inf leverage zeroes the margin per unit.
            raise RiskConfigError(f"AssetSpec for {self.symbol} has a non-finite field")
        if self.leverage < 1 or self.contract_size <= 0 or self.lot_step <= 0 or self.min_lot <= 0:
            raise RiskConfigError(f"invalid AssetSpec for {self.symbol}")
        if self.quote_to_account <= 0:
            raise RiskConfigError("quote_to_account must be > 0")
        if not (0 <= self.commission_bps < 1_000 and 0 <= self.slippage_bps < 1_000):
            raise RiskConfigError("commission_bps / slippage_bps must be in [0, 1000)")
        if self.price_tick < 0 or self.min_stop_distance < 0:
            raise RiskConfigError("price_tick / min_stop_distance must be >= 0")

    @property
    def unit_value(self) -> float:
        """Account-currency value of a 1.0 price move on one unit (lot/contract)."""
        return self.contract_size * self.quote_to_account

    def round_trip_cost_per_unit(self, entry: float, stop: float) -> float:
        """Estimated round-trip cost of one unit that enters at ``entry`` and exits at ``stop``.

        Commission and slippage are charged on both legs:
        ``unit_value * (entry + stop) * (commission + slippage)``.
        """
        per_side = (self.commission_bps + self.slippage_bps) / 10_000
        return self.unit_value * (entry + stop) * per_side

    def loss_per_unit_at_stop(self, entry: float, stop: float) -> float:
        """Worst-case (non-gap) loss of one unit: stop distance plus round-trip costs."""
        return abs(entry - stop) * self.unit_value + self.round_trip_cost_per_unit(entry, stop)


@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    """Aggregate exposure of currently open positions (in account currency)."""

    open_risk: float = 0.0  # sum of |entry - stop| * units * contract * fx
    open_notional: float = 0.0
    open_margin: float = 0.0


NO_EXPOSURE = ExposureSnapshot()


@dataclass(frozen=True, slots=True)
class TradeApproval:
    """Outcome of :func:`pre_trade_check`."""

    approved: bool
    max_risk: float
    reasons: tuple[str, ...] = ()
    code: str = "ok"  # "ok" | "status" | "budget_floor" | "min_trade_size" | "no_budget"


def pre_trade_check(state: AccountState, limits: RiskLimits,
                    exposure: ExposureSnapshot = NO_EXPOSURE) -> TradeApproval:
    """Compute the largest worst-case loss (in account currency) a new trade may carry.

    Let ``floor = min_budget_floor_pct * equity`` and, for each loss limit,
    ``remaining = headroom - open_risk``. Checks, in order:

    1. **Budget floor** — if the remaining daily *or* overall budget is below ``floor``,
       block. Trading halts while a cushion of at least ``floor`` is still left.
    2. The budget is the minimum of:

       * ``risk_per_trade_pct * equity``
       * ``max_open_risk_pct * equity - open_risk``
       * ``headroom_buffer * (remaining_daily - floor)``
       * ``headroom_buffer * (remaining_overall - floor)``

       The floor is reserved, so even if every open stop *and* the new stop are hit,
       equity stays at least ``floor`` above both loss limits (gaps aside, which the
       buffer and the cushion absorb).
    3. **Minimum trade size** — if that budget is below
       ``min_trade_risk_fraction * risk_per_trade_pct * equity``, block rather than open
       a token-sized trade.
    """
    if not state.trading_allowed:
        reason = f"account status {state.status.value}: {state.breach_reason or ''}".strip()
        return TradeApproval(False, 0.0, (reason,), "status")

    eq = state.equity
    floor = limits.min_budget_floor_pct * eq
    remaining_daily = state.daily_headroom(limits) - exposure.open_risk
    remaining_overall = state.drawdown_headroom(limits) - exposure.open_risk
    for label, remaining in (("daily", remaining_daily), ("overall", remaining_overall)):
        if remaining < floor:
            msg = (f"remaining {label} loss budget {remaining:,.2f} < floor {floor:,.2f} "
                   f"({limits.min_budget_floor_pct:.2%} of equity) - new trades halted")
            return TradeApproval(False, 0.0, (msg,), "budget_floor")

    budget = min(
        limits.risk_per_trade_pct * eq,
        limits.max_open_risk_pct * eq - exposure.open_risk,
        limits.headroom_buffer * (remaining_daily - floor),
        limits.headroom_buffer * (remaining_overall - floor),
    )
    if budget <= 0:
        return TradeApproval(False, 0.0, ("no remaining risk budget (open risk consumes the limit)",), "no_budget")
    min_trade = limits.min_trade_risk_fraction * limits.risk_per_trade_pct * eq
    if budget < min_trade:
        msg = (f"allowed risk {budget:,.2f} < minimum meaningful trade {min_trade:,.2f} "
               f"({limits.min_trade_risk_fraction:.0%} of a standard trade)")
        return TradeApproval(False, 0.0, (msg,), "min_trade_size")
    return TradeApproval(True, budget, ())


@dataclass(frozen=True, slots=True)
class PositionSize:
    """Result of :func:`size_position`."""

    units: float
    risk_amount: float
    notional: float
    margin: float
    binding_constraint: str
    approved: bool
    reasons: tuple[str, ...] = ()


def _floor_to_step(value: float, step: float) -> float:
    steps = math.floor(value / step + 1e-9)
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    return round(steps * step, decimals + 2)


def size_position(
    state: AccountState,
    limits: RiskLimits,
    asset: AssetSpec,
    entry: float,
    stop: float,
    exposure: ExposureSnapshot = NO_EXPOSURE,
) -> PositionSize:
    """Leverage-aware fixed-fractional position sizing.

    ``units = risk_budget * leverage_scalar / per_unit_risk``, then capped by
    notional exposure and margin utilisation, then floored to the lot step.

    * ``risk_budget`` comes from :func:`pre_trade_check` (already headroom-aware).
    * ``per_unit_risk`` is the stop distance **plus** estimated round-trip commission
      and slippage (:meth:`AssetSpec.loss_per_unit_at_stop`), so ``risk_amount`` is the
      full loss if the stop is hit, not just the price move.
    * ``leverage_scalar = min(1, reference_leverage / asset.leverage)``: an asset that
      offers 100:1 leverage gets a smaller risk budget than one offering 30:1 because
      gap and liquidation risk scale with leverage.
    * **Floors:** ``pre_trade_check`` halts new trades once the remaining daily or
      overall loss budget is below ``min_budget_floor_pct`` of equity
      (``binding_constraint == "budget_floor"``) or the allowed risk is below a minimum
      meaningful trade (``"min_trade_size"``). Additionally, if the budget cannot fund
      even one minimum lot including costs, the trade is blocked (``"min_lot_floor"``).

    Args:
        state: Current account state.
        limits: Risk limits.
        asset: Contract specification.
        entry: Intended entry price.
        stop: Protective stop price.
        exposure: Existing open exposure.

    Returns:
        A :class:`PositionSize`; ``approved`` is False with ``units == 0`` when no
        compliant size exists.
    """
    reasons: list[str] = []
    if entry <= 0 or stop <= 0 or not (math.isfinite(entry) and math.isfinite(stop)):
        return PositionSize(0, 0, 0, 0, "invalid_prices", False, ("entry/stop must be positive finite",))
    per_unit_price_risk = abs(entry - stop)
    if per_unit_price_risk <= _EPS:
        return PositionSize(0, 0, 0, 0, "zero_stop_distance", False, ("stop equals entry",))

    approval = pre_trade_check(state, limits, exposure)
    if not approval.approved:
        binding = "risk_gate" if approval.code in ("status", "no_budget") else approval.code
        return PositionSize(0, 0, 0, 0, binding, False, approval.reasons)

    unit_value = asset.unit_value
    per_unit_risk = asset.loss_per_unit_at_stop(entry, stop)
    min_lot_risk = asset.min_lot * per_unit_risk
    if approval.max_risk < min_lot_risk:
        reasons.append(f"remaining risk budget {approval.max_risk:,.2f} < one min-lot trade risk "
                       f"{min_lot_risk:,.2f} (incl. costs)")
        return PositionSize(0, 0, 0, 0, "min_lot_floor", False, tuple(reasons))

    per_unit_notional = entry * unit_value
    per_unit_margin = per_unit_notional / asset.leverage

    leverage_scalar = min(1.0, limits.reference_leverage / asset.leverage)
    candidates: dict[str, float] = {
        "risk_budget": approval.max_risk * leverage_scalar / per_unit_risk,
        "notional_cap": max(0.0, limits.max_notional_exposure_multiple * state.equity
                            - exposure.open_notional) / per_unit_notional,
        "margin_cap": max(0.0, limits.max_margin_utilisation * state.equity
                          - exposure.open_margin) / per_unit_margin,
    }
    binding = min(candidates, key=candidates.__getitem__)
    units = _floor_to_step(candidates[binding], asset.lot_step)

    if units < asset.min_lot:
        reasons.append(f"size {candidates[binding]:.6f} below min lot {asset.min_lot} ({binding})")
        return PositionSize(0, 0, 0, 0, binding, False, tuple(reasons))

    return PositionSize(
        units=units,
        risk_amount=units * per_unit_risk,
        notional=units * per_unit_notional,
        margin=units * per_unit_margin,
        binding_constraint=binding,
        approved=True,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------------------
# Hard bracket levels (SL/TP are mandatory and computed here, never by the transport)
# --------------------------------------------------------------------------------------
class BracketError(ValueError):
    """A bracket cannot be built: missing, non-finite or wrong-side SL/TP."""


@dataclass(frozen=True, slots=True)
class BracketLevels:
    """Broker-ready, validated stop-loss and take-profit prices for one entry."""

    side: Side
    entry: float
    stop: float
    take_profit: float
    adjusted: bool = False  # True if rounding / stops-level widened a level

    @property
    def stop_distance(self) -> float:
        """Price distance from entry to stop."""
        return abs(self.entry - self.stop)

    @property
    def reward_to_risk(self) -> float:
        """Target distance divided by stop distance."""
        return abs(self.take_profit - self.entry) / self.stop_distance


def _round_away(price: float, tick: float, entry: float) -> float:
    """Round ``price`` to ``tick`` moving *away* from ``entry``."""
    steps = price / tick
    steps = math.floor(steps + 1e-9) if price < entry else math.ceil(steps - 1e-9)
    return round(steps * tick, 10)


def _round_toward(price: float, tick: float, entry: float) -> float:
    """Round ``price`` to ``tick`` moving *toward* ``entry``."""
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if price < entry else math.floor(steps + 1e-9)
    return round(steps * tick, 10)


def _broker_levels(sign: int, entry: float, stop: float, take_profit: float,
                   asset: AssetSpec) -> tuple[float, float]:
    """Round SL/TP to the tick (stop away from entry, target toward it) and widen both to the
    broker's minimum stop distance."""
    new_stop, new_tp = stop, take_profit
    tick = asset.price_tick
    if tick > 0:
        new_stop = _round_away(stop, tick, entry)
        new_tp = _round_toward(take_profit, tick, entry)
        if (new_tp - entry) * sign <= 0:  # target collapsed onto entry: step one tick out
            new_tp = round(entry + sign * tick, 10) if tick else new_tp
    min_d = asset.min_stop_distance
    if min_d > 0:
        if abs(entry - new_stop) < min_d:
            new_stop = entry - sign * min_d
            new_stop = _round_away(new_stop, tick, entry) if tick > 0 else new_stop
        if abs(new_tp - entry) < min_d:
            new_tp = entry + sign * min_d
            new_tp = _round_away(new_tp, tick, entry) if tick > 0 else new_tp
    return new_stop, new_tp


def _require_tradable_levels(new_stop: float, new_tp: float) -> None:
    """Rounded SL/TP must be finite and positive; anything else is a BracketError, never an order."""
    if not (math.isfinite(new_stop) and math.isfinite(new_tp)):
        raise BracketError("bracket levels are not finite after tick rounding")
    if new_stop <= 0:
        raise BracketError("stop rounded to a non-positive price")
    if new_tp <= 0:  # e.g. a short whose target collapsed onto entry and was stepped a whole tick below 0
        raise BracketError("take-profit rounded to a non-positive price")


def hard_bracket(side: Side, entry: float, stop: float | None, take_profit: float | None,
                 asset: AssetSpec) -> BracketLevels:
    """Validate and normalise SL/TP into broker-ready hard levels.

    * SL and TP are mandatory: ``None``, non-finite, non-positive or wrong-side values
      raise :class:`BracketError` — a bracket without both levels can never be built.
    * The stop is rounded to the price tick *away* from entry (never tighter than the
      strategy asked for, so it is not hit by rounding noise); the target is rounded
      *toward* entry (never further than asked).
    * Both are widened to the broker's minimum stop distance if needed.

    Size the position with the returned ``stop``: rounding changes the risk per unit.
    """
    for name, value in (("entry", entry), ("stop", stop), ("take_profit", take_profit)):
        if value is None or not math.isfinite(value) or value <= 0:
            raise BracketError(f"{name} must be a positive finite price, got {value!r}")
    assert stop is not None and take_profit is not None
    sign = side.sign
    if (entry - stop) * sign <= 0:
        raise BracketError(f"{side.value} stop {stop} is not on the losing side of entry {entry}")
    if (take_profit - entry) * sign <= 0:
        raise BracketError(f"{side.value} take-profit {take_profit} is not on the winning side of entry {entry}")

    try:
        new_stop, new_tp = _broker_levels(sign, entry, stop, take_profit, asset)
    except OverflowError as exc:  # price / tick beyond float range: not a tradable level
        raise BracketError(f"price/tick out of range for {asset.symbol}") from exc
    _require_tradable_levels(new_stop, new_tp)
    adjusted = not (math.isclose(new_stop, stop) and math.isclose(new_tp, take_profit))
    return BracketLevels(side, entry, new_stop, new_tp, adjusted)


# --------------------------------------------------------------------------------------
# Immutable ratchet trailing stop
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RatchetConfig:
    """Parameters of the ratchet trailing stop.

    Attributes:
        breakeven_trigger_r: Max favourable excursion (in R) that locks breakeven.
        breakeven_offset_r: Lock the stop this many R beyond entry (covers costs).
        trail_activation_r: MFE (in R) at which the ATR chandelier trail starts.
        initial_atr_multiple: Chandelier ATR multiple at activation.
        min_atr_multiple: Floor of the decaying multiple.
        decay_per_r: Multiple shrinks by this much per extra R of MFE, so the trail
            tightens as the trade runs (locking more open profit).
    """

    breakeven_trigger_r: float = 1.0
    breakeven_offset_r: float = 0.05
    trail_activation_r: float = 1.0
    initial_atr_multiple: float = 3.0
    min_atr_multiple: float = 1.5
    decay_per_r: float = 0.5

    def __post_init__(self) -> None:
        if not (0 < self.min_atr_multiple <= self.initial_atr_multiple):
            raise RiskConfigError("need 0 < min_atr_multiple <= initial_atr_multiple")
        if self.decay_per_r < 0 or self.breakeven_trigger_r <= 0 or self.breakeven_offset_r < 0:
            raise RiskConfigError("invalid ratchet parameters")

    def atr_multiple(self, mfe_r: float) -> float:
        """Decay curve: ``max(min, initial - decay * (mfe_r - activation))``."""
        extra = max(0.0, mfe_r - self.trail_activation_r)
        return max(self.min_atr_multiple, self.initial_atr_multiple - self.decay_per_r * extra)


DEFAULT_RATCHET = RatchetConfig()


@dataclass(frozen=True, slots=True)
class RatchetStop:
    """Immutable trailing-stop state for one open position."""

    side: Side
    entry: float
    initial_stop: float
    stop: float
    best_price: float
    breakeven_locked: bool = False

    @classmethod
    def open(cls, side: Side, entry: float, stop: float) -> RatchetStop:
        """Create the initial state, validating that the stop is on the losing side."""
        for name, value in (("entry", entry), ("stop", stop)):
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                # A NaN stop fails every comparison, so is_hit() would never fire: no stop at all.
                raise ValueError(f"{name} must be a positive finite price, got {value!r}")
        if (entry - stop) * side.sign <= 0:
            raise ValueError(f"{side.value} stop {stop} must be on the losing side of entry {entry}")
        return cls(side=side, entry=entry, initial_stop=stop, stop=stop, best_price=entry)

    @property
    def r_unit(self) -> float:
        """Initial risk distance (1R) in price."""
        return abs(self.entry - self.initial_stop)

    def r_multiple(self, price: float) -> float:
        """Open PnL at ``price`` expressed in R."""
        return (price - self.entry) * self.side.sign / self.r_unit

    def locked_r(self) -> float:
        """R guaranteed if the current stop is hit (negative = loss)."""
        return self.r_multiple(self.stop)

    def is_hit(self, bar_high: float, bar_low: float) -> bool:
        """Whether a bar's range touched the stop."""
        return bar_low <= self.stop if self.side is Side.LONG else bar_high >= self.stop

    def _tighter(self, a: float, b: float) -> float:
        return max(a, b) if self.side is Side.LONG else min(a, b)


def ratchet_stop(state: RatchetStop, bar_high: float, bar_low: float, atr: float,
                 config: RatchetConfig | None = None) -> RatchetStop:
    """Return the next ratchet state after observing a completed bar.

    The returned stop is ``tighter(previous_stop, candidates)``, so it is monotone:
    it never moves against the trade, regardless of inputs (including a NaN ATR, which
    is ignored). Call this only *after* checking whether the bar hit the existing stop,
    so the new level applies from the next bar (no look-ahead).
    """
    config = config or DEFAULT_RATCHET
    sign = state.side.sign
    extreme = bar_high if state.side is Side.LONG else bar_low
    best = state._tighter(state.best_price, extreme) if math.isfinite(extreme) else state.best_price
    mfe_r = (best - state.entry) * sign / state.r_unit

    candidate = state.stop
    locked = state.breakeven_locked
    if mfe_r >= config.breakeven_trigger_r:
        breakeven = state.entry + sign * config.breakeven_offset_r * state.r_unit
        candidate = state._tighter(candidate, breakeven)
        locked = True
    if mfe_r >= config.trail_activation_r and math.isfinite(atr) and atr > 0:
        chandelier = best - sign * config.atr_multiple(mfe_r) * atr
        candidate = state._tighter(candidate, chandelier)

    new_stop = state._tighter(state.stop, candidate)
    return replace(state, stop=new_stop, best_price=best, breakeven_locked=locked)


@dataclass(frozen=True, slots=True)
class RiskReport:
    """Human/JSON-friendly summary of the account's risk posture."""

    status: str
    equity: float
    day_start_equity: float
    high_water_mark: float
    daily_pnl_pct: float
    drawdown_pct: float
    daily_headroom: float
    drawdown_headroom: float
    daily_limit_pct: float
    drawdown_limit_pct: float
    buffer_vs_firm_daily_pct: float
    buffer_vs_firm_max_pct: float
    breach_reason: str | None = None
    extras: dict[str, float] = field(default_factory=dict)


def risk_report(state: AccountState, limits: RiskLimits) -> RiskReport:
    """Build a :class:`RiskReport` from state and limits."""
    return RiskReport(
        status=state.status.value,
        equity=state.equity,
        day_start_equity=state.day_start_equity,
        high_water_mark=state.high_water_mark,
        daily_pnl_pct=state.daily_pnl_pct(),
        drawdown_pct=state.drawdown_pct(),
        daily_headroom=state.daily_headroom(limits),
        drawdown_headroom=state.drawdown_headroom(limits),
        daily_limit_pct=limits.daily_loss_limit_pct,
        drawdown_limit_pct=limits.max_trailing_drawdown_pct,
        buffer_vs_firm_daily_pct=PROP_FIRM_DAILY_LOSS_PCT - limits.daily_loss_limit_pct,
        buffer_vs_firm_max_pct=PROP_FIRM_MAX_LOSS_PCT - limits.max_trailing_drawdown_pct,
        breach_reason=state.breach_reason,
    )
