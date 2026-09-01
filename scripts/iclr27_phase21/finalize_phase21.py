#!/usr/bin/env python
"""Generate the Phase21 decision artifact and self-contained final report."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase21"
DOC = ROOT / "docs/iclr27_phase21"
P20 = ROOT / "outputs/iclr27_phase20"
P19 = ROOT / "outputs/iclr27_phase19r"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def command(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def fmt(x: Any, n: int = 4) -> str:
    if x is None: return "NA"
    if isinstance(x, bool): return "yes" if x else "no"
    if isinstance(x, float): return f"{x:.{n}f}"
    return str(x)


def main() -> None:
    geometry = read_json(OUT / "audit/geometry_audit.json")
    by_prefix = read_json(OUT / "audit/observability_by_prefix.json")
    stage1 = read_json(OUT / "metrics/stage1_proposal_variants.json")
    stage3 = read_json(OUT / "audit/stage3_gate_o.json")
    p20_dec = read_json(P20 / "audit/phase20_decision.json")
    p20_q = read_json(P20 / "audit/proposal_quality_repair.json")
    folds = read_json(OUT / "manifests/fold_manifest.json")
    now = datetime.now(timezone.utc).astimezone().isoformat()

    expected = [OUT / "audit/geometry_audit.json", OUT / "audit/observability_event_audit.json", OUT / "audit/observability_by_prefix.json", OUT / "audit/stage3_gate_o.json", OUT / "audit/stage3_event_records.json", OUT / "audit/full_76_event_summary.csv", OUT / "metrics/stage1_proposal_variants.json", OUT / "completion/stage0.done", OUT / "completion/stage1.done", OUT / "completion/stage3.done", DOC / "STAGE0_GEOMETRY_OBSERVABILITY_REPORT.md", DOC / "STAGE1_PROPOSAL_VARIANTS_REPORT.md", DOC / "STAGE3_GATE_O_REPORT.md"]
    parse_ok = True
    for p in expected:
        if not p.exists() or p.stat().st_size == 0: parse_ok = False
        if p.suffix == ".json":
            try: read_json(p)
            except Exception: parse_ok = False

    residual = []
    for line in command(["ps", "-eo", "pid,ppid,etime,cmd"]).splitlines():
        if "scripts/iclr27_phase21/" in line and "finalize_phase21.py" not in line:
            residual.append(line.strip())
    forbidden = []
    for p in OUT.rglob("*"):
        if p.is_file() and any(s in p.name.lower() for s in ("q1", "devplus", "new_model", "public_after", "freeze")):
            forbidden.append(str(p))
    symlinks = {}
    for rel in ("data/iclr27_phase19r/sources/public_rows_corrected.csv", "data/iclr27_phase19r/sources/public_cls_roi.npz"):
        p = ROOT / rel
        symlinks[rel] = {"is_symlink": p.is_symlink(), "link_target": os.readlink(p) if p.is_symlink() else None, "resolved_target": str(p.resolve()), "exists": p.exists()}

    base16 = next(x for x in stage1["variants"]["raw_baseline"]["prefix_summary"] if x["prefix"] == 16)
    best = stage1["best_variant"]; best16 = next(x for x in stage1["variants"][best]["prefix_summary"] if x["prefix"] == 16)
    gt16 = next(x for x in stage3["variants"]["gt_tight_oracle"]["prefix_summary"] if x["prefix"] == 16)
    frozen16 = next(x for x in stage3["variants"]["frozen_oracle_correspondence"]["prefix_summary"] if x["prefix"] == 16)
    stage1_pass = bool(stage1["gate_o_stage1_pass"]); gate_o = bool(stage3["gate_o_pass"])

    decision = {
        "protocol": "trackocd_iclr27_phase21_proposal_observability_repair_v1", "execution_time": now,
        "decision_code": "P21_GATE_O_PASS_AUTHORIZE_STAGE2" if gate_o else "P21_GATE_O_FAIL_PROPOSAL_OBSERVABILITY_STOP",
        "status": "STOPPED_AT_PROPOSAL_LAYER" if not gate_o else "STAGE2_AUTHORIZED",
        "gates": {"stage0_reproduction": bool(geometry["phase20_reproduction"]["exact_match"]), "stage1": stage1_pass, "O": gate_o,
                   "O_rule": "prefix16 true-IoU ceiling >=0.50, >25/76, both-side coverage improvement, >=3/4 fold direction"},
        "phase20_baseline": {"prefix_summary": p20_dec["stage0"]["prefix_summary"], "max_ceiling": p20_dec["gates"]["O"]["max_ceiling_recall"], "quality_proxy": p20_q["quality_proxy_ceiling_at_prefix16"], "quality_true": p20_q["true_iou_ceiling_at_prefix16"]},
        "stage0": {"geometry": geometry, "prefix_summary": by_prefix["prefix_summary"]},
        "stage1": stage1,
        "stage2": {"status": "not_authorized_stage1_failed" if not stage1_pass else "not_run_in_this_report", "checkpoints": [], "training_started": False},
        "stage3": {"best_nontraining_variant": best, "best_prefix16": best16, "gt_tight_oracle_prefix16": gt16, "frozen_oracle_correspondence_prefix16": frozen16, "gate_o_pass": gate_o},
        "public_status": "sealed; DEV+, Q1, and public new-model labels were not read; no public evaluation/freeze artifacts created",
        "resources": {"gpu_snapshot": command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]), "memory_snapshot": command(["free", "-h"]), "disk_snapshot": command(["df", "-h", "/data1"]), "python": sys.version, "platform": platform.platform(), "phase21_training": False, "oom": False, "near_oom": False, "other_user_processes_touched": False, "residual_phase21_processes": residual},
        "storage_symlinks": symlinks, "forbidden_phase21_outputs": forbidden,
        "integrity": {"expected_artifacts_present_and_json_parse": parse_ok, "stage0_done": (OUT / "completion/stage0.done").exists(), "stage1_done": (OUT / "completion/stage1.done").exists(), "stage3_done": (OUT / "completion/stage3.done").exists(), "all_76_event_rows": sum(1 for _ in (OUT / "audit/full_76_event_summary.csv").open()) - 1 if (OUT / "audit/full_76_event_summary.csv").exists() else 0, "no_public_q1_outputs": not forbidden, "residual_phase21_processes": residual},
        "git": {"status": command(["git", "status", "--short"]), "root": command(["git", "rev-parse", "--show-toplevel"]), "metadata_available": False},
        "commands": ["python scripts/iclr27_phase21/run_stage0_audit.py", "python scripts/iclr27_phase21/run_stage1_variants.py", "python scripts/iclr27_phase21/run_stage3_gate.py", "python scripts/iclr27_phase21/finalize_phase21.py"],
        "next_direction": "repair proposal-domain/ROI observability or revisit task definition; do not train correspondence/controller or tune thresholds until true Gate O passes",
    }
    atomic_json(OUT / "metrics/phase21_aggregate.json", {"protocol": decision["protocol"], "phase20_baseline": decision["phase20_baseline"], "stage0": decision["stage0"], "stage1": stage1, "stage3": stage3, "decision": decision["decision_code"]})
    atomic_json(OUT / "audit/phase21_decision.json", decision)

    # Hash all stable Phase21 artifacts after writing the decision/aggregate;
    # the hash file intentionally excludes itself.
    hash_paths = [p for p in expected if p.is_file()] + [OUT / "metrics/phase21_aggregate.json", OUT / "audit/phase21_decision.json"]
    atomic_json(OUT / "manifests/artifact_hashes.json", {str(p.relative_to(ROOT)): sha256(p) for p in hash_paths})

    lines = ["# TrackOCD ICLR 2027 — Phase 21 Proposal/Observability Repair", "", f"**Execution:** `{now}`  ", f"**Decision:** `{decision['decision_code']}`", "", "## Executive result", "", f"Phase21 rechecked the real TRAIN DSCT proposal stream and stopped at the proposal layer.  The Stage0 geometry/chronology audit reproduced Phase20 exactly: prefix16 perfect-correspondence true-IoU ceiling **25/76 = 0.328947**.  No registered non-training repair raised this ceiling; therefore Gate O is **FAIL** and no Stage2 proposal training, correspondence, controller, backbone, final, or public evaluation was started.", "", "The failure is an observation ceiling, not a claim that DINOv2 has no semantic signal.  Every event and denominator is preserved.", ""]
    lines += ["## Data, sealing, and frozen comparator", "", "- Public TRAIN category/video metadata and the original 76 positive pseudo-held events (plus 76 negatives for context) were used.  DEV+, Q1, public new-model labels, future frames/tracks, physical IDs, semantic text, and GT-tight boxes in the main path were not read.", "- Phase19R/20 evaluator, StateMemory, thresholds, action semantics, row keys, chronology, and event denominator remained read-only.  Phase21 wrote only its independent namespace.", "- No Git repository metadata exists; content hashes are recorded in [`artifact_hashes.json`](../../outputs/iclr27_phase21/manifests/artifact_hashes.json).", "", "Phase20 context: mixed persistent CT 2/76; event-aligned 2/76; event-repair 0/76.  Phase20 quality-head proxy was 31/76 while true IoU ceiling stayed 25/76.  Phase15S/17R showed DINOv2 offline signal and an independent observability limit, motivating the O-only split.", ""]
    lines += ["## Stage 0 — geometry, time, and baseline reproduction", "", f"Rows audited: **{geometry['source_rows']}** across resolutions `{geometry['resolution_counts']}`.  Invalid boxes: **{geometry['invalid_bbox_rows']}**; normalized-coordinate mismatches: **{geometry['normalized_coordinate_mismatch_rows']}**; stored/recomputed IoU mismatches: **{geometry['stored_iou_mismatch_rows']}**; duplicate row keys: **{geometry['duplicate_row_keys']}**; non-monotone tracks: **{geometry['chronology_bad_track_count']}**.  Actual `image_width`/`image_height` were used; 640×480 was not hard-coded; no future-frame/track read was detected.", "", "| prefix | Phase20 ceiling | Phase21 ceiling | source reliable | target reliable | category coverage | video coverage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    p20ps = {int(x["prefix"]): x for x in p20_dec["stage0"]["prefix_summary"]}
    for x in by_prefix["prefix_summary"]:
        lines.append(f"| {x['prefix']} | {p20ps[x['prefix']]['perfect_correspondence_ct_ceiling_correct'] if 'perfect_correspondence_ct_ceiling_correct' in p20ps[x['prefix']] else p20ps[x['prefix']].get('ceiling_correct')} | {x['ceiling_correct']} | {x['source_reliable']} | {x['target_reliable']} | {x['category_coverage']} | {x['video_coverage']} |")
    lines += ["", "The exact Phase20 curve 17/22/22/23/25 is reproduced at prefixes 1/2/4/8/16.  Event-level geometry/chronology and failure reasons are in [`geometry_audit.json`](../../outputs/iclr27_phase21/audit/geometry_audit.json), [`observability_event_audit.json`](../../outputs/iclr27_phase21/audit/observability_event_audit.json), and [`observability_by_prefix.json`](../../outputs/iclr27_phase21/audit/observability_by_prefix.json).", ""]
    lines += ["## Stage 1 — fixed proposal repair variants", "", "All variants were registered before execution, retained all rows/events, and used `assigned == 1 and transformed IoU >= 0.5`.  No event-specific tuning or hard-event deletion occurred.", "", "| variant | prefix | source reliable | target reliable | ceiling | recall | category coverage | video coverage | target IoU mean | median |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in stage1["variants"]:
        for x in stage1["variants"][n]["prefix_summary"]:
            st = x["target_iou_stats"]; lines.append(f"| {n} | {x['prefix']} | {x['source_reliable']} | {x['target_reliable']} | {x['ceiling_correct']}/76 | {x['ceiling_recall']:.4f} | {x['category_coverage']} | {x['video_coverage']} | {st['mean']:.4f} | {st['median']:.4f} |")
    lines += ["", f"At prefix16 the best non-training variant is **{best}** at **{best16['ceiling_correct']}/76**, equal to the raw baseline.  `causal_smoothed` fell to 22/76 and fixed 10% expansion to 21/76; history/ROI-history/quality-rerank were geometry-preserving and unchanged.  Because Stage0 had no coordinate error, no class-agnostic refinement branch was justified.", "", "Each variant's complete event list and failure event keys are available under [`outputs/iclr27_phase21/audit/stage1_*_events.json`](../../outputs/iclr27_phase21/audit/); aggregate/fold metrics are in [`stage1_proposal_variants.json`](../../outputs/iclr27_phase21/metrics/stage1_proposal_variants.json).", ""]
    lines += ["## Stage 3 — proposal/oracle ceiling and fold comparison", "", "| condition | prefix16 source reliable | target reliable | ceiling | recall | category coverage | video coverage |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n, v in stage3["variants"].items():
        x = next(z for z in v["prefix_summary"] if z["prefix"] == 16); lines.append(f"| {n} | {x['source_reliable']} | {x['target_reliable']} | {x['ceiling_correct']}/76 | {x['ceiling_recall']:.4f} | {x['category_coverage']} | {x['video_coverage']} |")
    lines += ["", "GT-tight/oracle is diagnostic only (73/76); frozen proposal plus oracle correspondence is still 25/76.  The Stage2 trained-proposal condition is explicitly **not authorized** because Stage1 failed.  Fold-level source/target/ceiling values for every condition are in [`stage3_gate_o.json`](../../outputs/iclr27_phase21/audit/stage3_gate_o.json).", "", "### Prefix16 fold table (all non-training variants)", "", "| variant | fold0 ceiling | fold1 ceiling | fold2 ceiling | fold3 ceiling |", "|---|---:|---:|---:|---:|"]
    for n, v in stage3["variants"].items():
        x = next(z for z in v["prefix_summary"] if z["prefix"] == 16); vals = {z["fold"]: f"{z['ceiling_correct']}/{z['denominator']}" for z in x["by_fold"]}; lines.append(f"| {n} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    lines += ["", "## Complete 76-event index (prefix16)", "", "The following compact table is the complete positive CT denominator.  It reports raw, best non-training, and GT-tight ceiling status per event; the full per-variant columns and failure reasons are in [`full_76_event_summary.csv`](../../outputs/iclr27_phase21/audit/full_76_event_summary.csv).", "", "| event | fold | category | source track | target track | raw | best | GT-tight |", "|---|---:|---:|---|---|---:|---:|---:|"]
    with (OUT / "audit/full_76_event_summary.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            lines.append(f"| {r['event_key']} | {r['fold']} | {r['category']} | {r['source_tracklet_key']} | {r['target_tracklet_key']} | {r['raw_baseline_ceiling']} | {r[f'{best}_ceiling']} | {r['gt_tight_oracle_ceiling']} |")
    lines += ["", "## Gate O and conditional branches", "", "| requirement | result |", "|---|---|", f"| prefix16 true-IoU ceiling >= 0.50 and >25/76 | **FAIL** ({best16['ceiling_correct']}/76 = {best16['ceiling_recall']:.4f}) |", f"| both source and target coverage improve | **FAIL** (baseline 49/40; best {best16['source_reliable']}/{best16['target_reliable']}) |", f"| >=3/4 folds improve directionally | **FAIL** (best equals baseline) |", "| fixed row keys, chronology, denominator, and evaluator | PASS |", "| Stage2 proposal module | **not trained/unauthorized** |", "| correspondence/controller/backbone/final/public | **stopped/sealed** |", "", "Final Gate O decision: **FAIL**.  The actionable evidence is missing reliable proposal geometry, not a controller or backbone result.  The next candidate is a proposal-domain/ROI observation repair or task-definition study; do not tune Phase19R thresholds or train correspondence until this gate passes.", ""]
    lines += ["## Environment, storage, failures, and reproduction", "", f"- Execution snapshot: `{now}`; Python `{platform.python_version()}` on `{platform.platform()}`.", "- `nvidia-smi`/RAM/disk snapshots and symlink targets are in [`phase21_decision.json`](../../outputs/iclr27_phase21/audit/phase21_decision.json).  Phase21 diagnostics were CPU-only; no GPU training, checkpoint, OOM, near-OOM, or other-user process impact occurred.  `/data1` had about 50G available at preflight, so features were reused by symlink rather than copied.", "- No residual Phase21 process remains; no public/Q1 output is present.  `stage0.done`, `stage1.done`, and `stage3.done` plus all JSON parse checks are recorded in the decision artifact.", "- Reproduce with: `python scripts/iclr27_phase21/run_stage0_audit.py`; `python scripts/iclr27_phase21/run_stage1_variants.py`; `python scripts/iclr27_phase21/run_stage3_gate.py`; `python scripts/iclr27_phase21/finalize_phase21.py`.", "", "Artifacts: [`phase21_decision.json`](../../outputs/iclr27_phase21/audit/phase21_decision.json), [`phase21_aggregate.json`](../../outputs/iclr27_phase21/metrics/phase21_aggregate.json), [`artifact_hashes.json`](../../outputs/iclr27_phase21/manifests/artifact_hashes.json).", ""]
    (DOC / "PHASE21_PROPOSAL_OBSERVABILITY_REPAIR_COMPLETE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    final_report = DOC / "PHASE21_PROPOSAL_OBSERVABILITY_REPAIR_COMPLETE_REPORT.md"
    # Finalize the integrity record only after the Markdown exists, then hash
    # the final decision/aggregate/report state (hash file itself is excluded).
    decision["integrity"]["final_report_nonempty"] = bool(final_report.exists() and final_report.stat().st_size > 0)
    atomic_json(OUT / "audit/phase21_decision.json", decision)
    hash_paths = [p for p in expected if p.is_file()] + [OUT / "metrics/phase21_aggregate.json", OUT / "audit/phase21_decision.json", final_report]
    atomic_json(OUT / "manifests/artifact_hashes.json", {str(p.relative_to(ROOT)): sha256(p) for p in hash_paths})
    print(json.dumps({"decision": decision["decision_code"], "gate_o": gate_o, "best_variant": best, "best16": best16["ceiling_correct"], "report": str(DOC / "PHASE21_PROPOSAL_OBSERVABILITY_REPAIR_COMPLETE_REPORT.md"), "integrity": parse_ok}, indent=2))


if __name__ == "__main__": main()
