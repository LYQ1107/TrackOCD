"""Bounded/atomic IO helpers for the Phase73 audit."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                count += 1
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return count
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Yield a top-level JSON array without retaining the 1.2M-row stream."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        buf = ""
        pos = 0
        eof = False

        def fill() -> None:
            nonlocal buf, pos, eof
            if eof:
                return
            part = fh.read(chunk_size)
            if part:
                if pos:
                    buf = buf[pos:]
                    pos = 0
                buf += part
            else:
                eof = True

        fill()
        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        while pos >= len(buf) and not eof:
            fill()
        if pos >= len(buf) or buf[pos] != "[":
            raise ValueError(f"expected top-level JSON array: {path}")
        pos += 1
        first = True
        while True:
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                fill()
            if not first:
                if buf[pos] != ",":
                    if buf[pos] == "]":
                        return
                    raise ValueError(f"missing comma in {path}")
                pos += 1
                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if pos < len(buf):
                        break
                    if eof:
                        raise ValueError(f"unterminated JSON array: {path}")
                    fill()
                if buf[pos] == "]":
                    return
            first = False
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            yield value
            pos = end
