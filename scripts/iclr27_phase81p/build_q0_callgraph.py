#!/usr/bin/env python3
"""Emit a source/hash callgraph for the frozen Q0 physical path."""
from __future__ import annotations
import ast, datetime, hashlib, json, os, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase81p/audit/q0_physical_callgraph.json"
Q0 = ROOT / "third_party/research_refs_phase4n/OVTR"
FILES = [
    Q0 / "ovtr/eval.py", Q0 / "ovtr/models/ovtr.py", Q0 / "ovtr/models/deformable_detr.py",
    ROOT / "scripts/iclr27_phase75a/ovtr_native_eval.py", ROOT / "scripts/iclr27_phase75b/run_event_replay.py",
    ROOT / "scripts/iclr27_phase75b/run_observability.py",
]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def symbols(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append({"name": node.name, "kind": type(node).__name__, "line": node.lineno})
    return sorted(out, key=lambda x: x["line"])

def main():
    rows = []
    for path in FILES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({
            "path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size,
            "symbols": symbols(path),
            "frozen_fields_referenced": sorted(set(re.findall(r"(?:score_mode|score_thresh|filter_score_thresh|ious_thresh|miss_tolerance|maximum_quantity|frame_id|proposal_local_id|physical_track_id)", text))),
        })
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema_version": "phase81p.q0_physical_callgraph.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_head": commit,
        "ovtr_commit": "500e72c",
        "q0_checkpoint": "outputs/iclr27_phase4q/q0_long/checkpoint.pth",
        "q0_checkpoint_sha256": "809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738",
        "q0_stream": "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json",
        "q0_stream_sha256": "112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac",
        "pipeline": ["RGB/frame loader", "OVTR detector/query decoder", "TrackerPostProcess score_mode=base", "RuntimeTrackerBase filtering", "native lineage capture", "event observability evaluator"],
        "frozen_contract": {"proposal_source": "Q0 OVTR output", "boxes": "xyxy absolute after target-size scaling", "base_score_only": True, "physical_id": "bookkeeping only", "semantic_heads": "excluded", "future_or_held_labels_before_inference": False},
        "files": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name("." + OUT.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
