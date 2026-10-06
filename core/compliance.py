"""Prop-firm session compliance: weekend auto-flatten and news blackouts.

:class:`WeekendGuard` defines a weekly blackout from Friday 19:30 UTC to Sunday 22:00
UTC (both configurable). Inside it:

* every open position must be flattened (``must_flatten``) — weekend gaps are the one
  loss the per-trade stop cannot bound, and many firms forbid holding over the weekend;
* no new orders may be opened.

:class:`ComplianceGate` combines the weekend guard with the optional
:class:`~core.news_filter.NewsFilter` into one decision the coordinator, the MCP
server and the MT5 EA's own backup guard all agree on.

Everything here is pure and clock-injected: callers pass ``now``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any

from core.news_filter import NewsFilter
from core.risk_engine import RiskConfigError, _zone

MINUTES_PER_WEEK = 7 * 24 * 60

#: Default Friday auto-flatten time (UTC). MetaQuotes-Demo closes XAUUSD at 23:00 server
#: time (20:00 UTC); flattening at 19:30 UTC leaves a 30-minute buffer so the closing
#: orders still reach an open market. Keep in sync with the EA's InpFlattenHourUTC/MinuteUTC.
FRIDAY_FLATTEN_UTC = time(19, 30)


def parse_hhmm(text: str) -> time:
    """Parse ``"HH:MM"`` (24h) into a :class:`datetime.time`."""
    try:
        hh, mm = text.strip().split(":")
        return time(int(hh), int(mm))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"expected HH:MM (24h), got {text!r}") from exc
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True, slots=True)
class WeekendGuardConfig:
    """Weekly blackout window. Weekdays follow ``datetime.weekday()`` (Mon=0 ... Sun=6).

    Attributes:
        flatten_weekday / flatten_time: Start of the blackout (default Fri 19:30).
        reopen_weekday / reopen_time: End of the blackout (default Sun 22:00, FX reopen).
        tz: IANA zone the times are expressed in (default UTC).
        enabled: Master switch.
    """

    flatten_weekday: int = 4
    flatten_time: time = FRIDAY_FLATTEN_UTC
    reopen_weekday: int = 6
    reopen_time: time = time(22, 0)
    tz: str = "UTC"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not (0 <= self.flatten_weekday <= 6 and 0 <= self.reopen_weekday <= 6):
            raise RiskConfigError("weekdays must be in 0..6")
        if self._minute(self.flatten_weekday, self.flatten_time) == self._minute(self.reopen_weekday,
                                                                                  self.reopen_time):
            raise RiskConfigError("flatten and reopen instants must differ")
        _zone(self.tz)

    @staticmethod
    def _minute(weekday: int, t: time) -> float:
        return weekday * 1440 + t.hour * 60 + t.minute + t.second / 60


@dataclass(frozen=True, slots=True)
class ComplianceDecision:
    """Outcome of a compliance evaluation."""

    allow_new_orders: bool
    must_flatten: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {"allow_new_orders": self.allow_new_orders, "must_flatten": self.must_flatten,
                "reasons": list(self.reasons)}


ALLOW = ComplianceDecision(True, False)


class WeekendGuard:
    """Friday auto-flatten / weekend no-trade window."""

    def __init__(self, config: WeekendGuardConfig | None = None) -> None:
        self.config = config or WeekendGuardConfig()

    def minute_of_week(self, now: datetime) -> float:
        """Minutes since Monday 00:00 in the guard's timezone."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        local = now.astimezone(_zone(self.config.tz))
        return local.weekday() * 1440 + local.hour * 60 + local.minute + local.second / 60

    def in_blackout(self, now: datetime) -> bool:
        """True from the flatten instant (inclusive) to the reopen instant (exclusive)."""
        cfg = self.config
        if not cfg.enabled:
            return False
        m = self.minute_of_week(now)
        start = cfg._minute(cfg.flatten_weekday, cfg.flatten_time)
        end = cfg._minute(cfg.reopen_weekday, cfg.reopen_time)
        return start <= m < end if start < end else (m >= start or m < end)

    def evaluate(self, now: datetime) -> ComplianceDecision:
        """Flatten and block inside the blackout; allow otherwise."""
        if not self.in_blackout(now):
            return ALLOW
        cfg = self.config
        reason = (f"weekend guard: flat from {WEEKDAY_NAMES[cfg.flatten_weekday]} {cfg.flatten_time:%H:%M} "
                  f"to {WEEKDAY_NAMES[cfg.reopen_weekday]} {cfg.reopen_time:%H:%M} {cfg.tz}")
        return ComplianceDecision(False, True, (reason,))


class ComplianceGate:
    """Single decision point combining the weekend guard and the news filter."""

    def __init__(self, weekend: WeekendGuard | None = None, news: NewsFilter | None = None) -> None:
        self.weekend = weekend
        self.news = news

    def evaluate(self, now: datetime, symbol: str) -> ComplianceDecision:
        """Aggregate: any component can block entries; only the weekend guard flattens."""
        reasons: list[str] = []
        allow, flatten = True, False
        if self.weekend is not None:
            wk = self.weekend.evaluate(now)
            allow &= wk.allow_new_orders
            flatten |= wk.must_flatten
            reasons += wk.reasons
        if self.news is not None:
            nd = self.news.check(now, symbol)
            if nd.blocked:
                allow = False
                reasons.append(nd.reason)
        return ComplianceDecision(allow, flatten, tuple(reasons))
