"""Emergency circuit breaker: cancel everything, flatten everything, stay down.

Properties:

* **Thread-safe** — ``trip`` may be called concurrently from the trading loop, the
  CLI, the MCP server, or a signal handler. A re-entrant lock serialises flattening;
  the ``tripped`` flag is a ``threading.Event`` so readers never block.
* **Idempotent & retrying** — tripping again re-runs cancel+flatten (harmless if
  already flat, useful if a previous attempt partially failed).
* **Cancel before flatten** — pending orders are revoked first so nothing fills
  behind the flatten.
* **Persistent** — a sentinel file is written on trip. A new process that finds it
  starts tripped, and another process (``python -m app.main kill``) can trip a running
  engine by writing it. Re-arming requires an explicit confirmation phrase *and* flat
  books.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from execution.base import BrokerAdapter

REARM_PHRASE = "REARM"
AuditHook = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class KillReport:
    """Result of a trip attempt."""

    reason: str
    source: str
    ts: str
    cancelled: int
    flattened: int
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """True when every adapter cancelled and flattened without error."""
        return not self.errors


class KillSwitch:
    """Process-wide emergency stop over any number of broker adapters.

    Args:
        adapters: Adapters to cancel/flatten on trip.
        sentinel_path: File persisting the tripped state (None = memory only).
        audit: Optional callback ``(event, payload)``.
        max_retries: Attempts per adapter operation.
        retry_delay_s: Pause between attempts.
    """

    def __init__(self, adapters: Iterable[BrokerAdapter] = (), sentinel_path: str | Path | None = None,
                 audit: AuditHook | None = None, max_retries: int = 3, retry_delay_s: float = 0.2) -> None:
        self._adapters: list[BrokerAdapter] = list(adapters)
        self._lock = threading.RLock()
        self._event = threading.Event()
        self._sentinel = Path(sentinel_path) if sentinel_path else None
        self._audit = audit
        self._max_retries = max(1, max_retries)
        self._delay = retry_delay_s
        self.last_report: KillReport | None = None
        if self._sentinel is not None and self._sentinel.exists():
            self._event.set()  # persisted trip survives restarts

    # ------------------------------------------------------------------------------
    @property
    def tripped(self) -> bool:
        """Lock-free read of the tripped flag."""
        return self._event.is_set()

    def register(self, adapter: BrokerAdapter) -> None:
        """Add an adapter to protect."""
        with self._lock:
            if adapter not in self._adapters:
                self._adapters.append(adapter)

    def _retry(self, fn: Callable[[], int], label: str, errors: list[str]) -> int:
        for attempt in range(1, self._max_retries + 1):
            try:
                return int(fn())
            except Exception as exc:  # noqa: BLE001 - every failure must be reported, never raised
                if attempt == self._max_retries:
                    errors.append(f"{label}: {type(exc).__name__}: {exc}")
                else:
                    time.sleep(self._delay)
        return 0

    def trip(self, reason: str, source: str = "api") -> KillReport:
        """Halt trading: set the flag, persist it, cancel all orders, flatten all positions.

        The flag is set *before* any network call so concurrent order attempts are
        refused immediately.
        """
        self._event.set()
        with self._lock:
            ts = datetime.now(UTC).isoformat(timespec="seconds")
            self._write_sentinel(reason, source, ts)
            report = self._cancel_and_flatten(reason, source, ts)
            self.last_report = report
            if self._audit is not None:
                self._audit("kill_switch_tripped", {"reason": reason, "source": source, "cancelled": report.cancelled,
                                                    "flattened": report.flattened, "errors": list(report.errors)})
            return report

    def soft_flatten(self, reason: str, source: str = "compliance") -> KillReport:
        """Cancel and flatten everything *without* latching the switch.

        Used for scheduled, non-terminal flattens (Friday weekend guard, daily-loss
        freeze): trading may resume when the condition clears, so no sentinel is
        written and :attr:`tripped` is unchanged. Same retry/reporting semantics as
        :meth:`trip`; callers should escalate to :meth:`trip` if the report is not clean.
        """
        with self._lock:
            ts = datetime.now(UTC).isoformat(timespec="seconds")
            report = self._cancel_and_flatten(reason, source, ts)
            if self._audit is not None:
                self._audit("soft_flatten", {"reason": reason, "source": source, "cancelled": report.cancelled,
                                             "flattened": report.flattened, "errors": list(report.errors)})
            return report

    def _cancel_and_flatten(self, reason: str, source: str, ts: str) -> KillReport:
        errors: list[str] = []
        cancelled = flattened = 0
        for adapter in list(self._adapters):
            cancelled += self._retry(adapter.cancel_all_orders, f"{adapter.name}.cancel_all_orders", errors)
        for adapter in list(self._adapters):
            flattened += self._retry(adapter.flatten_all, f"{adapter.name}.flatten_all", errors)
        return KillReport(reason, source, ts, cancelled, flattened, tuple(errors))

    def check_external_trigger(self) -> KillReport | None:
        """Trip if another process wrote the sentinel. Call once per trading cycle."""
        if self._sentinel is None or self.tripped or not self._sentinel.exists():
            return None
        try:
            info = json.loads(self._sentinel.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        return self.trip(str(info.get("reason", "external sentinel")), source=str(info.get("source", "sentinel")))

    def rearm(self, confirmation: str) -> bool:
        """Clear the trip. Requires ``confirmation == "REARM"`` and flat books everywhere."""
        if confirmation != REARM_PHRASE:
            raise PermissionError(f"re-arm requires the exact confirmation phrase {REARM_PHRASE!r}")
        with self._lock:
            open_positions = [p for a in self._adapters for p in a.positions()]
            if open_positions:
                raise PermissionError(f"refusing to re-arm with {len(open_positions)} open position(s)")
            if self._sentinel is not None and self._sentinel.exists():
                self._sentinel.unlink()
            self._event.clear()
            if self._audit is not None:
                self._audit("kill_switch_rearmed", {})
            return True

    def _write_sentinel(self, reason: str, source: str, ts: str) -> None:
        if self._sentinel is None:
            return
        self.write_sentinel(self._sentinel, reason, source, ts)

    @staticmethod
    def write_sentinel(path: str | Path, reason: str, source: str = "cli", ts: str | None = None) -> None:
        """Write the sentinel file (used by the CLI to trip a running engine)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = ts or datetime.now(UTC).isoformat(timespec="seconds")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps({"reason": reason, "source": source, "ts": ts}), encoding="utf-8")
        tmp.replace(p)  # atomic on the same filesystem
