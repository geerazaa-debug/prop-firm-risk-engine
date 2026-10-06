"""OHLCV data model and data sources (synthetic, CSV, public ccxt endpoints).

The synthetic generator is deterministic for a given seed and produces explicit
trending and mean-reverting (chop) regimes, which makes it useful both for the demo
dashboard and for testing that the regime filter actually suppresses chop.
"""
from __future__ import annotations

import csv
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

RegimeKind = Literal["trend", "chop"]


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. ``ts`` is the bar's open time (timezone-aware UTC)."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not (self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)):
            raise ValueError(f"inconsistent OHLC at {self.ts}: {self.open}/{self.high}/{self.low}/{self.close}")

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly representation."""
        return {"ts": self.ts.isoformat(), "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": self.volume}


def closes(bars: Sequence[Bar]) -> list[float]:
    """Extract close prices."""
    return [b.close for b in bars]


def _make_bar(ts: datetime, prev_close: float, close: float, rng: random.Random, wick: float) -> Bar:
    o = prev_close
    hi = max(o, close) * (1 + abs(rng.gauss(0, wick)))
    lo = min(o, close) * (1 - abs(rng.gauss(0, wick)))
    return Bar(ts, o, hi, lo, close, volume=round(1_000 + abs(rng.gauss(0, 400)), 2))


def synthetic_ohlcv(
    n_bars: int = 1_000,
    seed: int = 7,
    start_price: float = 2_000.0,
    start: datetime = datetime(2022, 1, 3, tzinfo=UTC),
    step: timedelta = timedelta(days=1),
    regime_length: int = 60,
    schedule: Sequence[RegimeKind] | None = None,
    trend_drift: float = 0.004,
    trend_vol: float = 0.009,
    chop_vol: float = 0.011,
    chop_reversion: float = 0.35,
) -> list[Bar]:
    """Generate a deterministic regime-switching price path.

    Trend regimes follow a drifted random walk whose direction is drawn per regime;
    chop regimes follow an Ornstein-Uhlenbeck process that reverts to the regime's
    anchor price, producing false breakouts on both sides.

    Args:
        n_bars: Number of bars.
        seed: RNG seed (same seed, same series).
        start_price: First open.
        start: Timestamp of the first bar.
        step: Bar duration.
        regime_length: Bars per regime block.
        schedule: Optional explicit regime sequence (cycled); default alternates
            ``trend``/``chop``.
        trend_drift: Absolute per-bar log drift in trend regimes.
        trend_vol: Per-bar volatility in trend regimes.
        chop_vol: Per-bar volatility in chop regimes.
        chop_reversion: OU pull-back speed per bar in chop regimes.
    """
    if n_bars <= 0:
        return []
    rng = random.Random(seed)
    plan: Sequence[RegimeKind] = schedule or ("trend", "chop")
    bars: list[Bar] = []
    price = start_price
    anchor = price
    direction = 1.0
    for i in range(n_bars):
        block = i // regime_length
        regime = plan[block % len(plan)]
        if i % regime_length == 0:
            anchor = price
            direction = 1.0 if rng.random() < 0.6 else -1.0
        if regime == "trend":
            log_ret = direction * trend_drift + rng.gauss(0, trend_vol)
        else:
            log_ret = -chop_reversion * math.log(price / anchor) + rng.gauss(0, chop_vol)
        close = max(price * math.exp(log_ret), 0.01)
        bars.append(_make_bar(start + i * step, price, close, rng, wick=0.003))
        price = close
    return bars


def load_csv(path: str | Path) -> list[Bar]:
    """Load bars from a CSV with columns ``ts,open,high,low,close[,volume]``.

    The time column may be named ``ts``, ``timestamp``, ``time``, ``datetime`` or ``date``
    and may hold ISO-8601 or epoch milliseconds. Extra columns are ignored.
    """
    out: list[Bar] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        ts_col = next((c for c in ("ts", "timestamp", "time", "datetime", "date")
                       if c in (reader.fieldnames or [])), None)
        if ts_col is None:
            raise ValueError(f"{path}: no time column (expected one of ts/timestamp/time/datetime/date)")
        for row in reader:
            raw_ts = row[ts_col].strip()
            if raw_ts.isdigit():
                ts = datetime.fromtimestamp(int(raw_ts) / 1000, tz=UTC)
            else:
                ts = datetime.fromisoformat(raw_ts)
                ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            out.append(Bar(ts, float(row["open"]), float(row["high"]), float(row["low"]),
                           float(row["close"]), float(row.get("volume") or 0.0)))
    out.sort(key=lambda b: b.ts)
    return out


def resample(bars: Sequence[Bar], period: timedelta) -> list[Bar]:
    """Aggregate bars into ``period`` buckets aligned to the Unix epoch (UTC).

    Open = first open, high = max, low = min, close = last close, volume = sum. A bucket
    is emitted only if it is complete, i.e. a later bar exists after it (the last,
    possibly still-forming bucket is dropped).
    """
    if not bars:
        return []
    step = int(period.total_seconds())
    if step <= 0:
        raise ValueError("period must be positive")
    out: list[Bar] = []
    bucket: list[Bar] = []
    key = None
    first_complete = int(bars[0].ts.timestamp()) % step == 0
    for b in bars:
        k = int(b.ts.timestamp()) // step
        if key is not None and k != key and bucket:
            out.append(_merge(bucket, key * step))
            bucket = []
        key = k
        bucket.append(b)
    if not first_complete and out:
        out = out[1:]  # data started mid-bucket: the first bucket is partial
    return out  # trailing bucket intentionally dropped (may be incomplete)


def _merge(bucket: Sequence[Bar], start_epoch: int) -> Bar:
    return Bar(datetime.fromtimestamp(start_epoch, tz=UTC), bucket[0].open, max(b.high for b in bucket),
               min(b.low for b in bucket), bucket[-1].close, sum(b.volume for b in bucket))


def from_ccxt_rows(rows: Iterable[Sequence[float]]) -> list[Bar]:
    """Convert ccxt ``fetch_ohlcv`` rows (``[ms, o, h, l, c, v]``) into bars."""
    return [Bar(datetime.fromtimestamp(r[0] / 1000, tz=UTC), float(r[1]), float(r[2]),
                float(r[3]), float(r[4]), float(r[5] or 0.0)) for r in rows]


def fetch_ccxt_public(exchange_id: str, symbol: str, timeframe: str = "1d", limit: int = 500) -> list[Bar]:
    """Fetch OHLCV from a public (unauthenticated) ccxt endpoint.

    No API keys are used or accepted here: market data never needs trading rights.
    """
    import ccxt  # local import: optional dependency

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    return from_ccxt_rows(exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
