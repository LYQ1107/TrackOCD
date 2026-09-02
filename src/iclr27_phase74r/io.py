"""Small, deterministic and atomic IO primitives for Phase74R."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(Path(path), text.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream a top-level JSON array without buffering the Q0 stream."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer = ""
        cursor = 0
        eof = False

        def fill() -> None:
            nonlocal buffer, cursor, eof
            if eof:
                return
            part = handle.read(chunk_size)
            if part:
                if cursor:
                    buffer = buffer[cursor:]
                    cursor = 0
                buffer += part
            else:
                eof = True

        fill()
        while cursor < len(buffer) and buffer[cursor].isspace():
            cursor += 1
        if cursor >= len(buffer) or buffer[cursor] != "[":
            raise ValueError(f"expected top-level JSON array: {path}")
        cursor += 1
        first = True
        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    break
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                fill()
            if not first:
                if buffer[cursor] == "]":
                    return
                if buffer[cursor] != ",":
                    raise ValueError(f"missing comma: {path}")
                cursor += 1
                while True:
                    while cursor < len(buffer) and buffer[cursor].isspace():
                        cursor += 1
                    if cursor < len(buffer):
                        break
                    if eof:
                        raise ValueError(f"unterminated JSON array: {path}")
                    fill()
                if buffer[cursor] == "]":
                    return
            first = False
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            yield value
            cursor = end
