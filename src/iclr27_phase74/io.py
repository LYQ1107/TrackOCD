"""Atomic and streaming IO helpers used by the Phase74 audit."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, records: Iterable[Any]) -> int:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                n += 1
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return n


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} is not an object")
                yield value


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream a top-level JSON array without retaining 1.2M Q0 rows."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as f:
        buf = ""; pos = 0; eof = False

        def fill() -> None:
            nonlocal buf, pos, eof
            if eof: return
            part = f.read(chunk_size)
            if part:
                if pos: buf = buf[pos:]; pos = 0
                buf += part
            else: eof = True

        fill()
        while pos < len(buf) and buf[pos].isspace(): pos += 1
        if pos >= len(buf) or buf[pos] != "[":
            raise ValueError(f"expected top-level JSON array: {path}")
        pos += 1; first = True
        while True:
            while True:
                while pos < len(buf) and buf[pos].isspace(): pos += 1
                if pos < len(buf): break
                if eof: raise ValueError(f"unterminated JSON array: {path}")
                fill()
            if not first:
                if buf[pos] == "]": return
                if buf[pos] != ",": raise ValueError(f"missing comma: {path}")
                pos += 1
                while True:
                    while pos < len(buf) and buf[pos].isspace(): pos += 1
                    if pos < len(buf): break
                    if eof: raise ValueError(f"unterminated JSON array: {path}")
                    fill()
                if buf[pos] == "]": return
            first = False
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    if eof: raise
                    fill()
            yield value; pos = end


def file_metadata(path: Path, *, record_count: int | None = None, schema_keys: Iterable[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "realpath": str(p.resolve()), "is_symlink": p.is_symlink(), "exists": p.exists()}
    if p.exists():
        st = p.stat(); out.update({"bytes": st.st_size, "mtime_epoch": st.st_mtime})
        # Directory roots are provenance anchors, not files to hash.  A
        # recursive image hash would be both unsafe and needlessly expensive;
        # individual annotation paths are hashed only when required.
        if p.is_file():
            out["sha256"] = sha256(p)
        else:
            out["kind"] = "directory"
    if record_count is not None: out["record_count"] = int(record_count)
    if schema_keys is not None: out["schema_keys"] = sorted(set(schema_keys))
    return out
