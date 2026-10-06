"""High-impact economic news blackout filter.

Blocks *new* entries from ``before`` minutes ahead of a high-impact event until
``after`` minutes past it (default ±5 minutes) for any event whose currency affects
the traded symbol. Existing positions are left alone — flattening inside the news
window can itself violate some firms' rules.

Calendar sources:

* Forex-Factory-style JSON (the public ``ff_calendar_thisweek.json`` feed): a list of
  ``{"title", "country", "date", "impact"}`` with ISO dates carrying a UTC offset.
* CSV with columns ``ts,currency,impact,title``.
* :func:`fetch_ff_calendar` downloads the weekly JSON (standard library only).

The filter fails **closed** by default: with no calendar, or a calendar whose coverage
ended more than ``stale_after`` ago, every entry is blocked — trading blind through an
unknown NFP print is exactly what the filter exists to prevent.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

FF_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

#: Titles treated as high impact even if the feed labels them lower (or omits impact).
HIGH_IMPACT_KEYWORDS: tuple[str, ...] = (
    "non-farm", "nonfarm", "nfp", "cpi", "consumer price index", "fomc", "federal funds rate",
    "interest rate decision", "fed chair", "powell", "ecb press conference", "boe", "boj",
)

_CURRENCY_ALIASES = {"USDT": "USD", "USDC": "USD", "XAU": "XAU", "XAG": "XAG"}
#: Metals are priced in USD and react to USD data.
_EXTRA_CURRENCIES = {"XAU": {"USD"}, "XAG": {"USD"}}


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def currencies_for(symbol: str) -> frozenset[str]:
    """Currencies whose news moves ``symbol`` (``XAUUSD`` -> {XAU, USD}; ``BTC/USDT`` -> {USD})."""
    raw = symbol.upper().split(":")[0].replace("/", "").replace(".", "")
    letters = "".join(ch for ch in raw if ch.isalpha())
    parts = {letters[:3], letters[3:6]} if len(letters) >= 6 else {letters}
    out: set[str] = set()
    for part in parts:
        cur = _CURRENCY_ALIASES.get(part, part)
        out.add(cur)
        out |= _EXTRA_CURRENCIES.get(cur, set())
    return frozenset(c for c in out if c)


@dataclass(frozen=True, slots=True)
class NewsEvent:
    """One calendar event, timestamped in UTC."""

    ts: datetime
    currency: str
    title: str
    impact: str  # "high" | "medium" | "low" | "holiday" | ""

    @property
    def is_high_impact(self) -> bool:
        """High by label, or by title keyword (NFP/CPI/FOMC...)."""
        title = self.title.lower()
        return self.impact == "high" or any(k in title for k in HIGH_IMPACT_KEYWORDS)


def _norm_impact(value: Any) -> str:
    v = str(value or "").strip().lower()
    return {"red": "high", "orange": "medium", "yellow": "low"}.get(v, v)


def parse_ff_calendar(rows: Iterable[dict[str, Any]]) -> list[NewsEvent]:
    """Parse Forex-Factory-style JSON rows. Unparseable rows (tentative/all-day) are skipped."""
    out: list[NewsEvent] = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(str(row["date"]))
        except (KeyError, ValueError):
            continue
        out.append(NewsEvent(_aware(ts).astimezone(UTC), str(row.get("country", "")).upper(),
                             str(row.get("title", "")), _norm_impact(row.get("impact"))))
    return sorted(out, key=lambda e: e.ts)


def load_calendar(path: str | Path) -> list[NewsEvent]:
    """Load a calendar from ``.json`` (Forex-Factory format) or ``.csv`` (``ts,currency,impact,title``)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return parse_ff_calendar(json.loads(p.read_text(encoding="utf-8")))
    events: list[NewsEvent] = []
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = _aware(datetime.fromisoformat(row["ts"].strip())).astimezone(UTC)
            events.append(NewsEvent(ts, row["currency"].strip().upper(), row.get("title", "").strip(),
                                    _norm_impact(row.get("impact"))))
    return sorted(events, key=lambda e: e.ts)


def fetch_ff_calendar(url: str = FF_THIS_WEEK_URL, timeout_s: float = 10.0) -> list[NewsEvent]:
    """Download and parse the weekly Forex-Factory JSON feed."""
    req = urllib.request.Request(url, headers={"User-Agent": "quantmind-news-filter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return parse_ff_calendar(json.loads(resp.read().decode("utf-8")))


@dataclass(frozen=True, slots=True)
class NewsFilterConfig:
    """Blackout settings."""

    before: timedelta = timedelta(minutes=5)
    after: timedelta = timedelta(minutes=5)
    fail_closed: bool = True
    stale_after: timedelta = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class NewsDecision:
    """Whether new entries are blocked right now, and why."""

    blocked: bool
    reason: str = ""
    event: NewsEvent | None = None


@dataclass
class NewsFilter:
    """Blackout checker over a loaded calendar.

    Args:
        events: Calendar events (any order).
        config: Window and staleness settings.
        coverage_end: Last instant the calendar is known to cover; defaults to the last
            event's timestamp. Past ``coverage_end + stale_after`` the calendar is stale.
    """

    events: Sequence[NewsEvent]
    config: NewsFilterConfig = field(default_factory=NewsFilterConfig)
    coverage_end: datetime | None = None

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda e: e.ts)
        if self.coverage_end is None and self.events:
            self.coverage_end = self.events[-1].ts

    def relevant(self, symbol: str) -> list[NewsEvent]:
        """High-impact events that affect ``symbol``."""
        curs = currencies_for(symbol) | {"ALL"}
        return [e for e in self.events if e.is_high_impact and e.currency in curs]

    def check(self, now: datetime, symbol: str) -> NewsDecision:
        """Is ``now`` inside a blackout window for ``symbol``?"""
        now = _aware(now)
        cfg = self.config
        if self.coverage_end is None or now > self.coverage_end + cfg.stale_after:
            if cfg.fail_closed:
                return NewsDecision(True, "news calendar missing or stale - entries blocked (fail closed)")
            return NewsDecision(False, "news calendar missing or stale (fail open)")
        for ev in self.relevant(symbol):
            if ev.ts - cfg.before <= now <= ev.ts + cfg.after:
                return NewsDecision(True, f"news blackout: {ev.currency} {ev.title} at {ev.ts:%H:%M} UTC", ev)
        return NewsDecision(False)

    def upcoming(self, now: datetime, symbol: str, horizon: timedelta = timedelta(hours=24)) -> list[NewsEvent]:
        """Relevant high-impact events within ``horizon`` of ``now``."""
        now = _aware(now)
        return [e for e in self.relevant(symbol) if now <= e.ts <= now + horizon]
