"""Tamper-evident, append-only decision audit trail.

Each record is one JSON line carrying ``prev_hash`` and its own ``hash``
(SHA-256 over the canonical JSON of the record without ``hash``). Editing or deleting
any historical line breaks the chain, which :meth:`AuditTrail.verify` detects.

Values under keys that look like credentials are redacted before they are written.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64
_SENSITIVE = ("secret", "password", "passphrase", "api_key", "apikey", "token", "signature", "private")


def redact(value: Any) -> Any:
    """Recursively replace values of credential-like keys with ``"***"``."""
    if isinstance(value, Mapping):
        return {k: ("***" if any(s in str(k).lower() for s in _SENSITIVE) else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One entry of the trail."""

    seq: int
    ts: str
    event: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def as_dict(self) -> dict[str, Any]:
        """Dict form, as written to disk."""
        return {"seq": self.seq, "ts": self.ts, "event": self.event, "payload": self.payload,
                "prev_hash": self.prev_hash, "hash": self.hash}


class AuditTrail:
    """Thread-safe hash-chained JSONL log with an in-memory tail for dashboards.

    Args:
        path: File to append to, or None for memory only (tests, dry runs).
        tail_size: Number of recent records kept in memory.
    """

    def __init__(self, path: str | Path | None = None, tail_size: int = 200) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._tail: deque[AuditRecord] = deque(maxlen=tail_size)
        self._seq = 0
        self._last_hash = GENESIS_HASH
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._resume()

    def _resume(self) -> None:
        assert self.path is not None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self._seq = rec["seq"]
                    self._last_hash = rec["hash"]
                    self._tail.append(AuditRecord(**rec))

    def record(self, event: str, payload: Mapping[str, Any] | None = None) -> AuditRecord:
        """Append an event and return the stored record."""
        with self._lock:
            body = {
                "seq": self._seq + 1,
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "event": event,
                "payload": redact(dict(payload or {})),
                "prev_hash": self._last_hash,
            }
            digest = hashlib.sha256(_canonical(body).encode()).hexdigest()
            rec = AuditRecord(hash=digest, **body)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(_canonical(rec.as_dict()) + "\n")
            self._seq = rec.seq
            self._last_hash = digest
            self._tail.append(rec)
            return rec

    def tail(self, n: int = 20) -> list[AuditRecord]:
        """Most recent ``n`` records, oldest first."""
        with self._lock:
            return list(self._tail)[-n:]

    def verify(self) -> tuple[bool, int | None]:
        """Re-hash the on-disk chain. Returns ``(ok, first_bad_seq)``."""
        if self.path is None or not self.path.exists():
            return True, None
        prev = GENESIS_HASH
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                claimed = rec.pop("hash")
                if rec["prev_hash"] != prev or hashlib.sha256(_canonical(rec).encode()).hexdigest() != claimed:
                    return False, rec.get("seq")
                prev = claimed
        return True, None
