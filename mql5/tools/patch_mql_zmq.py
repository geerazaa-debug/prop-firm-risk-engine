"""Patch dingmaotu/mql-zmq headers so they compile on current MetaTrader 5 builds.

Upstream ``mql-zmq`` (last commit Dec 2023) declares byte buffers and DLL string
parameters as ``char[]`` but fills them with ``StringToUtf8(..., uchar &[])``. Newer
MetaEditor compilers reject that implicit ``char[]`` <-> ``uchar[]`` conversion
(``error 246: parameter convertion type 'char[]' to 'uchar[] &' is not allowed``).

This script rewrites every ``char`` type token to ``uchar`` in the ``Zmq`` headers
(skipping ``//`` comments), preserving each file's encoding (several are UTF-16).
The change is ABI-neutral: both are 1-byte types passed to ``libzmq.dll`` by pointer.

Usage::

    python mql5/tools/patch_mql_zmq.py "<MT5 data folder>/MQL5/Include/Zmq"

It is idempotent and writes a ``.orig`` backup of each file it changes.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_CHAR = re.compile(r"(?<![A-Za-z0-9_])char(?![A-Za-z0-9_])")


def _detect(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def patch_text(text: str) -> tuple[str, int]:
    """Replace ``char`` type tokens outside ``//`` comments. Returns (text, replacements)."""
    out: list[str] = []
    count = 0
    for line in text.splitlines(keepends=True):
        code, sep, comment = line.partition("//")
        new_code, n = _CHAR.subn("uchar", code)
        count += n
        out.append(new_code + sep + comment)
    return "".join(out), count


def patch_dir(zmq_include: Path) -> dict[str, int]:
    """Patch every ``*.mqh`` under ``zmq_include``. Returns {file: replacements}."""
    changed: dict[str, int] = {}
    for path in sorted(zmq_include.glob("*.mqh")):
        raw = path.read_bytes()
        enc = _detect(raw)
        text = raw.decode(enc)
        new, n = patch_text(text)
        if n:
            backup = path.with_suffix(path.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_bytes(new.encode(enc))
            changed[path.name] = n
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zmq_include", type=Path, help="path to MQL5/Include/Zmq")
    args = parser.parse_args(argv)
    if not (args.zmq_include / "Zmq.mqh").exists():
        print(f"{args.zmq_include} does not contain Zmq.mqh", file=sys.stderr)
        return 2
    changed = patch_dir(args.zmq_include)
    for name, n in changed.items():
        print(f"patched {name}: {n} char -> uchar")
    print("already patched - nothing to do" if not changed else f"{len(changed)} file(s) patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
