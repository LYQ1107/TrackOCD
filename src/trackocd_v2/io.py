"""Small I/O helpers shared by TrackOCD v2 scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TARGET = Path("/data2/usr_for_deadline/trackocd_v2/project_outputs")
OUTPUT_LINK = PROJECT_ROOT / "outputs/trackocd_v2"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_output_layout() -> Path:
    """Create the v2 output target and a safe project-local symlink.

    A pre-existing non-symlink is never replaced.  This is important because
    historical output directories are read-only evidence.
    """

    OUTPUT_TARGET.mkdir(parents=True, exist_ok=True)
    if OUTPUT_LINK.is_symlink():
        if OUTPUT_LINK.resolve() != OUTPUT_TARGET.resolve():
            raise RuntimeError(f"trackocd_v2 output link points elsewhere: {OUTPUT_LINK.resolve()}")
    elif OUTPUT_LINK.exists():
        raise RuntimeError(f"refusing to replace non-symlink output path: {OUTPUT_LINK}")
    else:
        OUTPUT_LINK.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_LINK.symlink_to(OUTPUT_TARGET, target_is_directory=True)
    for name in ("audit", "manifests", "features", "diagnostics", "tables", "logs", "paper", "checkpoints"):
        (OUTPUT_TARGET / name).mkdir(parents=True, exist_ok=True)
    return OUTPUT_TARGET
