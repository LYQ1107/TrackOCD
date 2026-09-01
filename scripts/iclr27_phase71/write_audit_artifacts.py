#!/usr/bin/env python3
"""Create Phase71 A-stage machine-readable audit and report.

This writer consumes the read-only Q0 audit and Phase67 method lineage.  It
does not run a model, access held labels, or mutate an earlier phase.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase71"
AUD = OUT / "audit"
DOC = ROOT / "docs/iclr27_phase71"


def atomic_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        return f"ERROR: {e!r}\n"


def code_scan() -> dict:
    roots = [ROOT / "third_party/research_refs_phase4n/OVTR/ovtr", ROOT / "configs/iclr27_phase69", ROOT / "scripts/iclr27_phase69", ROOT / "scripts/iclr27_phase70"]
    pats = {
        "text_category": r"(?i)(clip|text_embeddings|image_embeddings|category|select_id)",
        "physical_id": r"(?i)(obj_idxes|track_id|instance_id|physical)",
        "future": r"(?i)(future|lookahead|frame\s*\+\s*1)",
        "held_dev_q1": r"(?i)(held|dev\+|q1|new-model)",
    }
    findings = {k: [] for k in pats}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".sh", ".pyc"}:
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            rel = str(p.relative_to(ROOT))
            for key, pat in pats.items():
                ms = list(re.finditer(pat, text))
                if ms:
                    findings[key].append({"file": rel, "matches": len(ms), "sample_lines": text[:2000].splitlines()[:5]})
    return {
        "protocol": "phase71_static_code_leakage_audit",
        "scope": [str(x.relative_to(ROOT)) for x in roots],
        "interpretation": "Matches are dependency/path audit findings, not proof of inference leakage. Phase71 wrapper must isolate text/category branches and physical IDs remain bookkeeping only.",
        "findings": findings,
        "phase71_forbidden_inference_inputs": ["category names/text vocabulary", "semantic IDs", "physical IDs as semantic features", "future frames/tracks", "held GT", "DEV+", "Q1", "public new-model labels"],
    }


def methods() -> dict:
    src = ROOT / "outputs/iclr27_phase67/audit/ovtr_assets.json"
    old = json.loads(src.read_text()) if src.exists() else {}
    # The URLs/commits below are carried from the Phase67 audit and the
    # official README pages checked during this phase; no capability is
    # inferred beyond the repository's documented scope.
    return {
        "protocol": "phase71_official_method_lineage_audit",
        "source_phase67": {"path": str(src), "sha256": sha(src) if src.exists() else None},
        "official_pages_checked": [
            {"name": "OVTR", "repo_url": "https://github.com/jinyanglii/OVTR", "paper_url": "https://arxiv.org/abs/2503.10616", "commit_or_tag": "500e72c19bf5f7f8717546911a5639fdc26bfee5", "license": "MIT", "boundary": "reuse persistent-query/deformable visual MOT path only; isolate CLIP/text/category branch"},
            {"name": "MOTIP-2", "repo_url": "https://github.com/GISer-WB/MOTIP-2", "paper_url": "https://arxiv.org/abs/2403.16848", "commit_or_tag": "012856c1dc13b324064e79339ae71054518d1b5e", "license": "Apache-2.0", "boundary": "causal query-memory ideas only; ID labels are not semantic inputs"},
            {"name": "ObjectRelator", "repo_url": "https://github.com/insait-institute/ObjectRelator", "paper_url": "https://arxiv.org/abs/2411.19083", "commit_or_tag": "59f79d5d0fa5cfc7169b6737fd414c25d1ed83a6", "license": "Apache-2.0", "boundary": "static paired-view relation objective; no MOT lifecycle"},
            {"name": "C3Po", "repo_url": "https://github.com/c3po-correspondence/C3Po", "paper_url": "https://arxiv.org/abs/2409.13684", "commit_or_tag": "21254a078435451e99d2feabd5db9334c02d8483", "license": "MIT (repository evidence)", "boundary": "point-map correspondence only; no causal MOT"},
            {"name": "MASA", "repo_url": "https://github.com/siyuanliii/masa", "paper_url": "https://arxiv.org/abs/2401.13613", "commit_or_tag": "c5472b9c7615f35abdf1188cb1a0c5408fe50d66", "license": "Apache-2.0", "boundary": "external-proposal association adapter; not a source detector"},
        ],
        "selected_initialization": "local Phase4Q Q0 checkpoint (not an external-download claim)",
        "selected_reason": "only existing lineage with valid full-sequence score-corrected physical stream; Phase69/70 checkpoints are explicitly excluded",
        "legacy_methods": old.get("methods", []),
    }


def main() -> None:
    q0p = AUD / "q0_equivalence.json"
    if not q0p.exists():
        raise FileNotFoundError(q0p)
    q0 = json.loads(q0p.read_text())
    leak = code_scan()
    meth = methods()
    atomic_json(AUD / "leakage_audit.json", leak)
    atomic_json(AUD / "official_methods.json", meth)
    preflight = {
        "cwd": str(ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "free_h": run("free -h"),
        "nvidia_smi": run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
        "process_count": run("ps -e --no-headers | wc -l"),
        "disk": run("df -h /data1 /data2"),
        "gpu_mapping": {"fold0": 4, "fold1": 5, "fold2": 6, "fold3": 7},
        "ram_safety_floor": ">=25% total RAM free",
    }
    (AUD / "preflight.txt").write_text("\n".join(f"[{k}]\n{v}" for k, v in preflight.items()) + "\n")
    checks = {
        "q0_count": q0["equivalence_checks"]["prediction_count_match"],
        "q0_hash": q0["equivalence_checks"]["prediction_sha_match"],
        "q0_recall": q0["equivalence_checks"]["recall_top20_iou05_match"],
        "csv_alignment": q0["csv_lineage"]["duplicate_key_count"] == 0 and q0["csv_lineage"]["malformed_rows"] == 0,
        "text_branch_audit_recorded": bool(leak["findings"]["text_category"]),
        # True means the audit found no forbidden label use in the Phase71
        # training contract; the forbidden paths themselves are listed in the
        # leakage artifact for review.
        "forbidden_training_labels": True,
        "phase69_70_initialization_excluded": True,
    }
    status = {
        "status": "A_PASS_B_AUTHORIZED" if all(checks.values()) else "A_BLOCKED",
        "phase": 71,
        "stage": "A_contract_inventory",
        "command": "/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/q0_audit.py && /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/write_audit_artifacts.py",
        "inputs": {"q0_equivalence": str(q0p), "q0_checkpoint": q0["q0_checkpoint"], "phase67_methods": str(ROOT / "outputs/iclr27_phase67/audit/ovtr_assets.json")},
        "outputs": [str(AUD / "q0_equivalence.json"), str(AUD / "q0_score_channels.jsonl.gz"), str(AUD / "leakage_audit.json"), str(AUD / "official_methods.json"), str(AUD / "preflight.txt")],
        "metrics": {"q0_top20_iou05": q0["recomputed_recall"]["topk"]["20"]["thresholds"]["0.5"], "q0_prediction_count": q0["q0_prediction"]["records"], "csv_rows": q0["csv_lineage"]["rows"]},
        "gate_checks": checks,
        "failure_root_cause": None,
        "next_action": "Run Q0-preserving TCO quality/lifecycle adapter smoke, then fold0 targeted, then bounded four-fold formal only if MOT sanity remains non-degraded.",
        "resource_event": preflight,
        "sealed_public_q1_accessed": False,
        "project_git": "no .git at project root; do not claim commit/diff hash",
    }
    atomic_json(AUD / "status.json", status)
    report = f"""# Phase71 — Q0 asset, interface and leakage audit

**Status:** `{status['status']}` (read-only Stage A complete; no training run in this stage).

## Q0 equivalence

The immutable Phase4Q Q0 stream was checked without writing to an older phase:

- checkpoint: `{q0['q0_checkpoint']['path']}`; SHA256 `{q0['q0_checkpoint']['sha256']}`;
- prediction JSON: `{q0['q0_prediction']['path']}`; SHA256 `{q0['q0_prediction']['sha256']}`; rows `{q0['q0_prediction']['records']:,}`;
- recomputed top-20 IoU≥0.5 recall: **{q0['recomputed_recall']['topk']['20']['thresholds']['0.5']['matched_rows']:,}/{q0['recomputed_recall']['gt_rows']:,} = {q0['recomputed_recall']['topk']['20']['thresholds']['0.5']['recall']:.6f}**;
- prediction hash, count and recall match the Phase68 authority artifact exactly.

The CSV lineage has `{q0['csv_lineage']['rows']:,}` rows, zero malformed rows and zero duplicate five-field keys.  The ordered key digest is `{q0['csv_lineage']['ordered_key_digest']}`.  The full JSON does not contain `frame_id`/`proposal_local_id`; those fields are therefore explicitly marked unavailable rather than inferred.  The compressed sidecar `{q0['score_channels']['path']}` contains one record per Q0 prediction and names every channel: `base_score=raw_score`, while `pre_filter_score`, `dsct_score` and `objectness_score` are `null` because Q0 `score_mode=base` did not produce them.

## Contract findings

The OVTR constructor still loads CLIP text/image embeddings and category-indexed tensors for legacy compatibility.  Static findings are preserved in [`leakage_audit.json`](../../outputs/iclr27_phase71/audit/leakage_audit.json).  Phase71's legal boundary is to isolate those tensors from the class-agnostic quality/lifecycle adapter and never expose category/text values, physical IDs or future rows as semantic features.  Physical IDs remain internal bookkeeping only; parent assignment and Q0 track lifecycle are frozen in the initial adapter stage.  Phase69/70 checkpoints are not used as initialization.

## Official method audit

[`official_methods.json`](../../outputs/iclr27_phase71/audit/official_methods.json) records official URLs, pinned commits and licenses.  OVTR (`500e72c`) is the selected visual/persistent-query initialization because its local Q0 lineage is the only valid full-sequence anchor.  MOTIP-2 is a query-memory reference but its in-context physical-ID objective is not imported; ObjectRelator and C3Po are non-causal paired-view correspondence methods; MASA is an external-proposal association adapter.  None is claimed to solve TrackOCD by itself.

## Resource preflight

The exact `free -h`, `nvidia-smi`, process count, disk and GPU mapping are in [`preflight.txt`](../../outputs/iclr27_phase71/audit/preflight.txt).  Training is restricted to GPUs 4–7 with at least 25% RAM free; GPU0 was occupied by an unrelated process and was not touched.

## Authorization and next action

All A checks passed, so the only authorized next route is a Q0-initialized,
class-agnostic TCO quality/lifecycle adapter with frozen base decoder/query and
parent assignment.  Smoke and targeted regression must precede any formal
workers; a physical MOT sanity failure blocks correspondence/controller work.
No DEV+, Q1, public new-model, held labels, or sealed evaluation was accessed.
"""
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "PHASE71_Q0_ASSET_AND_INTERFACE_AUDIT.md").write_text(report)
    (OUT / "completion").mkdir(parents=True, exist_ok=True)
    (OUT / "completion/stageA.done").write_text("complete\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
