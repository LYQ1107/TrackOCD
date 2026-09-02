"""Static dependency audit for the legacy OVTR Q0 reference path."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .io import sha256


def scan_files(paths: Iterable[Path]) -> dict[str, Any]:
    evidence = []
    hits = []
    needles = ("clip", "text", "category", "category_id", "class_embed", "language")
    for p in paths:
        if not p.exists() or not p.is_file(): continue
        text = p.read_text(encoding="utf-8", errors="replace")
        match = [{"line": i, "text": line.strip()[:300]} for i, line in enumerate(text.splitlines(), 1) if any(n in line.lower() for n in needles)]
        evidence.append({"path": str(p.resolve()), "sha256": sha256(p), "hit_count": len(match), "hits": match[:40]})
        hits.extend(match)
    # Static references alone do not establish that category affects output,
    # but they do prevent class-agnostic qualification without a runtime trace.
    return {"schema_version": "phase74.q0_dependency.v1", "classification": "TEXT_CATEGORY_DEPENDENCY_UNKNOWN" if hits else "NO_TEXT_CATEGORY_FORWARD_PATH",
            "qualified_for_semantic_stage": False if hits else None, "static_hits": len(hits), "evidence": evidence,
            "runtime_trace": {"status": "NOT_RUN_Q0_REPLAY_BLOCKED"}, "input_perturbation": {"status": "NOT_RUN_Q0_REPLAY_BLOCKED"},
            "notes": "Legacy OVTR config/source contains CLIP/text/category symbols; no claim of output dependence is made without replay."}
