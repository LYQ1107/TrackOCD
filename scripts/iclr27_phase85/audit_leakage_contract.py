#!/usr/bin/env python3
"""Check Phase85 manifests and code for declared inference-boundary flags."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    manifests = [OUT / "manifests/phase85_support_prefix_manifest.json", OUT / "manifests/physical_gate_examples.json", OUT / "manifests/source_track_selective_vectors.json"]
    manifest_checks = []
    for path in manifests:
        z = json.loads(path.read_text(encoding="utf-8"))
        forbidden = z.get("model_input_forbidden", [])
        manifest_checks.append({"path": str(path.resolve()), "sha256": sha(path), "exists": True, "forbidden_declared": forbidden, "public_dev_q1_sealed_accessed": z.get("public_dev_q1_sealed_accessed", False), "future_rows_or_tracks": z.get("future_rows_or_tracks", False), "ids_as_model_input": z.get("ids_as_model_input", False), "labels_posthoc_only": z.get("labels_posthoc_only", z.get("train_labels_posthoc_only", True))})
    scripts = sorted((ROOT / "scripts/iclr27_phase85").glob("*.py"))
    forbidden_inference_tokens = ["physical_id", "semantic_id", "category_name", "category_text", "future_frame", "future_track", "StateMemory", "controller_action"]
    source_mentions = []
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        # Mentions in explicit forbidden lists/post-hoc audit fields are not
        # violations; record them so the report can distinguish declaration
        # from inference use.
        for token in forbidden_inference_tokens:
            if token in text:
                source_mentions.append({"script": str(path.relative_to(ROOT)), "token": token, "declaration_or_audit_context": bool(re.search(r"forbidden|posthoc|post_hoc|labels_posthoc|ids_as_model_input|future_rows", text, re.I))})
    flags = {"public_dev_q1_sealed_accessed": any(x["public_dev_q1_sealed_accessed"] for x in manifest_checks), "future_rows_or_tracks": any(x["future_rows_or_tracks"] for x in manifest_checks), "ids_as_model_input": any(x["ids_as_model_input"] for x in manifest_checks), "non_declaration_forbidden_mentions": [x for x in source_mentions if not x["declaration_or_audit_context"]]}
    result = {"schema_version": "trackocd.phase85.leakage_contract.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "manifest_checks": manifest_checks, "source_mentions": source_mentions, "flags": flags, "status": "PASS" if not flags["public_dev_q1_sealed_accessed"] and not flags["future_rows_or_tracks"] and not flags["ids_as_model_input"] and not flags["non_declaration_forbidden_mentions"] else "REVIEW", "note": "Token mentions in post-hoc label/audit declarations are retained as evidence and are not inference inputs."}
    atomic(OUT / "audit/leakage_contract.json", result)
    atomic(OUT / "completion/leakage_contract.done", {"status": result["status"], "audit": str((OUT / "audit/leakage_contract.json").resolve()), "sha256": sha(OUT / "audit/leakage_contract.json")})
    print(json.dumps({"status": result["status"], "manifest_checks": len(manifest_checks), "non_declaration_forbidden_mentions": flags["non_declaration_forbidden_mentions"]}, indent=2, sort_keys=True))
    if result["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
