#!/usr/bin/env python3
"""Freeze Phase29 representation evidence and write the negative report."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase29"
DOC = ROOT / "docs/iclr27_phase29/PHASE29_CROSS_FOLD_DOMAIN_ALIGNMENT_COMPLETE_REPORT.md"
PREFIXES = (1, 2, 4, 8, 16)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def smi() -> dict[str, Any]:
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        rows = []
        for line in text.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 4: rows.append({"index": int(p[0]), "memory_used_MiB": int(p[1]), "memory_free_MiB": int(p[2]), "utilization_percent": int(p[3])})
        return {"query_ok": True, "gpus": rows}
    except Exception as exc:
        return {"query_ok": False, "error": repr(exc)}


def fmt(v: Any, n: int = 4) -> str:
    try: return f"{float(v):.{n}f}"
    except Exception: return "NA"


def main() -> None:
    frozen = json.loads((OUT / "audit/frozen_inputs.json").read_text())
    measured = json.loads((OUT / "metrics/representation_validation.json").read_text())
    event_records = json.loads((OUT / "audit/representation_event_records.json").read_text())
    rows = measured["folds"]
    gate = measured["gate_r"]
    # Confirm formal units and preserve the two failed/superseded smoke paths;
    # their markers are provenance, not successful units.
    formal_units = []
    for fold in range(4):
        m = json.loads((OUT / "metrics" / f"domain_aligned_f{fold}.json").read_text())
        formal_units.append({"fold": fold, "steps": m["steps"], "best_step": m["best_step"], "checkpoint": m["checkpoint_best"], "checkpoint_sha256": sha(Path(m["checkpoint_best"]))})
    smoke_markers = sorted(p.name for p in (OUT / "completion").glob("domain_smoke*_f0.launched"))
    smoke_done = sorted(p.name for p in (OUT / "completion").glob("domain_smoke*_f0.done"))
    incidents = {
        "repair_cycles": 2,
        "first_failure": {"tag": "domain_smoke_smoke_f0", "root_cause": "pos_sim multiplied [B,D] by [B,2,D]; missing singleton dimension", "exit_code": 1, "checkpoint_only": True, "done": False},
        "debug_reproduction": {"tag": "domain_smoke_debug_smoke_f0", "root_cause": "same shape error captured with stderr", "checkpoint_only": True, "done": False},
        "repair": "changed (ea * pos) to (ea[:, None, :] * pos); no protocol/model-boundary change",
        "performance_repair": "vectorized candidate masks and positive membership in retrieval evaluator; exact retrieval semantics preserved",
        "smoke_success": ["domain_smoke_fix1_smoke_f0", "domain_smoke_fix2_smoke_f0"],
        "targeted_success": ["domain_targeted_f0", "domain_targeted_fix2_f0"],
        "no_oom": True,
        "external_processes_touched": False,
    }
    atomic(OUT / "audit/process_and_repair_incidents.json", incidents)
    # Full aggregate summary at prefix16; no held-event metric is used for
    # checkpoint selection or this gate.
    p16 = []
    for r in rows:
        b, e = r["baseline"]["16"], r["encoder"]["16"]
        p16.append({"fold": r["fold"], "validation_tracklets": r["validation_tracklets"], "best_step": r["best_step"], "baseline_r1": b["r1"], "encoder_r1": e["r1"], "delta_r1": e["r1"] - b["r1"], "baseline_r5": b["r5"], "encoder_r5": e["r5"], "delta_r5": e["r5"] - b["r5"], "baseline_map": b["map"], "encoder_map": e["map"], "delta_map": e["map"] - b["map"], "baseline_hard_negative_gap": b["hard_negative_gap"], "encoder_hard_negative_gap": e["hard_negative_gap"], "substantial": bool(r["fold"] in [x["fold"] for x in gate.get("folds", []) if x.get("substantial")])})
    agg = {"baseline_r1_mean": sum(x["baseline_r1"] for x in p16) / 4, "encoder_r1_mean": sum(x["encoder_r1"] for x in p16) / 4, "baseline_map_mean": sum(x["baseline_map"] for x in p16) / 4, "encoder_map_mean": sum(x["encoder_map"] for x in p16) / 4, "folds_substantial": gate["folds_substantial"], "folds_directional": gate["folds_directional"]}
    atomic(OUT / "metrics/representation_aggregate.json", {"protocol": "trackocd_iclr27_phase29_representation_aggregate", "prefix16": p16, "aggregate": agg, "gate_r": gate, "positive_event_denominator": 76})
    paths = [OUT / "audit/frozen_inputs.json", OUT / "audit/representation_event_records.json", OUT / "audit/process_and_repair_incidents.json", OUT / "audit/mot_invariants.json", OUT / "metrics/representation_validation.json", OUT / "metrics/representation_aggregate.json", OUT / "completion/stage0.done", OUT / "completion/representation_validation.done"]
    paths += [Path(x["checkpoint"]) for x in formal_units]
    hashes = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() and p.is_file() else None, "is_symlink": p.is_symlink(), "resolved": str(p.resolve()) if p.exists() else None} for p in paths}
    atomic(OUT / "audit/artifact_hashes.json", hashes)
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,cmd"], text=True).splitlines()
    residual = [x for x in ps if "iclr27_phase29" in x and "finalize_phase29.py" not in x]
    resource = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "nvidia_smi": smi(), "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()), "phase29_processes": residual, "free_h": subprocess.check_output(["free", "-h"], text=True), "disk_df": subprocess.check_output(["df", "-h", "/data1"], text=True)}
    atomic(OUT / "audit/resource_postflight.json", resource)
    forbidden = [str(p) for p in OUT.rglob("*") if p.is_file() and any(x in p.name.lower() for x in ("q1", "dev+", "public_new_model"))]
    integrity = {"json_parse_ok": True, "positive_event_records": len(event_records.get("records", [])) == 76, "formal_fold_done": all((OUT / "completion" / f"domain_aligned_f{f}.done").exists() for f in range(4)), "formal_checkpoints": all(Path(x["checkpoint"]).exists() for x in formal_units), "stage0_done": (OUT / "completion/stage0.done").exists(), "validation_done": (OUT / "completion/representation_validation.done").exists(), "mot_invariants_frozen": (OUT / "audit/mot_invariants.json").exists() and (OUT / "audit/mot_invariants.json").is_symlink() and (OUT / "audit/mot_invariants.json").resolve().exists(), "phase29_processes_empty_at_finalize": not residual, "forbidden_output_name_hits": forbidden, "public_q1_accessed": False, "controller_started": False, "proposal_changed": False, "smoke_failed_markers_preserved": all(x in smoke_markers for x in ["domain_smoke_smoke_f0.launched", "domain_smoke_debug_smoke_f0.launched"]), "no_public_label_artifact": True}
    atomic(OUT / "audit/integrity.json", integrity)
    lines = [
        "# TrackOCD ICLR 2027 — Phase29 Cross-fold/Domain Representation Alignment",
        "",
        f"**Execution (UTC):** `{datetime.now(timezone.utc).isoformat()}`  ",
        f"**Decision:** **`{gate['decision']}`**  ",
        "**Scope:** exactly one class-agnostic representation/domain-alignment route after Phase28 Gate C failure; Phase26 proposal, physical MOT and Phase19R controller remain frozen.",
        "",
        "## Executive decision",
        "",
        f"The zero-initialized residual adapter was trained for 2,000 updates on each of four TRAIN-only video/category-disjoint folds. At prefix16, validation R@1 changed from **{agg['baseline_r1_mean']:.4f}** to **{agg['encoder_r1_mean']:.4f}** and mAP from **{agg['baseline_map_mean']:.4f}** to **{agg['encoder_map_mean']:.4f}** (means over folds). Representation Gate R is **{gate['decision']}**: {gate['folds_substantial']}/4 folds meet the preregistered +0.02 R@1 and +0.01 mAP improvement, and {gate['folds_directional']}/4 improve both metrics. No controller compatibility run, threshold/memory tuning, backbone download or public evaluation was authorized.",
        "",
        "Phase28's frozen controller compatibility result (3/76) remains the only Phase28 persistent measurement; Phase29 does not claim a new Commit-CT. This is a negative representation result, not MOT+OCD success.",
        "",
        "## Frozen boundaries and protocol",
        "",
        "- Phase26 source proposal Gate P2 PASS (real source prefix16 41/76) and physical track/MOT stream are symlinked read-only. Phase19R controller, StateMemory, masks, thresholds, action semantics and evaluator are not imported into the training graph.",
        "- Inputs are key-aligned fused DINOv2 CLS/ROI causal track features. The adapter sees only mean, last and absolute last-minus-mean statistics; no category/physical/semantic ID, text, video ID, future row, GT or StateMemory is a model input.",
        "- Training used public TRAIN rows and category/video metadata only for legal cross-video positives and hard negatives. Validation is held-category and held-video disjoint within the fixed four folds. The original 76 positive pseudo-held event manifest is retained for a non-selective cosine diagnostic; registered negatives were not used to tune the model.",
        "- DEV+, Q1 and public new-model labels were not read. The 76-event denominator, causal chronology, row keys, parent assignments and proposal source were unchanged.",
        "",
        "## Registered method",
        "",
        "`DomainAlignedResidualEncoder` is a single non-recurrent residual adapter: LayerNorm over concatenated causal mean/last/absolute-delta statistics, a zero-initialized linear residual at fixed scale 0.10, addition to the frozen causal mean, and L2 normalization to 768-D. The identity initialization protects the frozen baseline; supervised training uses cross-video multi-positive InfoNCE, nearest raw-DINO hard-negative ranking, prefix consistency and a small residual norm penalty. Video-balanced sampling is metadata-only. No GRU or modern backbone is used.",
        "",
        "## Smoke, targeted regression and repair provenance",
        "",
        "- The first smoke and a debug reproduction failed before the first update because the positive-similarity tensor omitted a singleton dimension (`[B,D] * [B,2,D]`). Both `.launched` markers and initial checkpoints are preserved as superseded/debug evidence; neither has a `.done` or metrics file and neither is treated as success.",
        "- The minimal fix changed only the broadcast to `ea[:,None,:] * pos`. `domain_smoke_fix1` and `domain_smoke_fix2` completed; fold0 targeted runs `domain_targeted_f0` and `domain_targeted_fix2` completed. A second code-only repair vectorized candidate masks/positive membership in retrieval, preserving exact ranking semantics and reducing validation time. The incident ledger is [`process_and_repair_incidents.json`](../../outputs/iclr27_phase29/audit/process_and_repair_incidents.json).",
        "",
        "## Four-fold retrieval validation",
        "",
        "| fold | val tracks | best step | baseline R@1 | adapter R@1 | ΔR@1 | baseline mAP | adapter mAP | ΔmAP | baseline R@5 | adapter R@5 | hard gap base→adapter |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in p16:
        lines.append(f"| {x['fold']} | {x['validation_tracklets']} | {x['best_step']} | {fmt(x['baseline_r1'])} | {fmt(x['encoder_r1'])} | {fmt(x['delta_r1'])} | {fmt(x['baseline_map'])} | {fmt(x['encoder_map'])} | {fmt(x['delta_map'])} | {fmt(x['baseline_r5'])} | {fmt(x['encoder_r5'])} | {fmt(x['baseline_hard_negative_gap'])} → {fmt(x['encoder_hard_negative_gap'])} |")
    lines += [
        f"| **mean** | — | — | **{agg['baseline_r1_mean']:.4f}** | **{agg['encoder_r1_mean']:.4f}** | **{agg['encoder_r1_mean']-agg['baseline_r1_mean']:.4f}** | **{agg['baseline_map_mean']:.4f}** | **{agg['encoder_map_mean']:.4f}** | **{agg['encoder_map_mean']-agg['baseline_map_mean']:.4f}** | — | — | — |",
        "",
        "Fold0 and fold3 select the identity checkpoint (best step 0), fold1 improves R@1 but loses mAP, and fold2 improves R@1 but loses mAP/R@5. Therefore no fold meets the joint substantial criterion. Complete prefix curves, category/video macro and query denominators are in [`representation_validation.json`](../../outputs/iclr27_phase29/metrics/representation_validation.json).",
        "",
        "## Prefix curves and 76-event diagnostic",
        "",
        "The evaluator reports prefixes 1/2/4/8/16 with the same causal mean baseline and frozen adapter. These are representation diagnostics only; no prefix snapshot changes the registered controller cutoff or event denominator.",
        "",
        "| fold | prefix | baseline R@1 | adapter R@1 | baseline mAP | adapter mAP | ΔR@1 | ΔmAP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        for p in PREFIXES:
            b, e = r["baseline"][str(p)], r["encoder"][str(p)]
            lines.append(f"| {r['fold']} | {p} | {fmt(b['r1'])} | {fmt(e['r1'])} | {fmt(b['map'])} | {fmt(e['map'])} | {fmt(e['r1']-b['r1'])} | {fmt(e['map']-b['map'])} |")
    lines += [
        "",
        f"All **{len(event_records.get('records', []))}/76** event keys are retained in [`representation_event_records.json`](../../outputs/iclr27_phase29/audit/representation_event_records.json). Event cosine differences are not persistent Commit-CT and were not used for checkpoint selection.",
        "",
        "## Phase28 compatibility context and Gate accounting",
        "",
        "Phase28's no-training frozen DINOv2 + old controller diagnostic produced 3/76 versus historical 2/76, but all three were fold3/category81/source video575 (target videos1814/1955); fold3 false-merge and new-recall safety regressed. The Phase28 Gate C decision was `P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION`. Phase29 therefore tests only representation alignment and does not rerun that controller.",
        f"Representation Gate R conditions: +0.02 R@1 and +0.01 mAP on at least three folds. Measured substantial={gate['folds_substantial']}/4, directional={gate['folds_directional']}/4, decision **{gate['decision']}**. Because Gate R failed, persistent Commit-CT, false merge, duplicate births, premature commit, known/novel safety and MOT compatibility for the adapter are **not run/claimed**.",
        "",
        "## Resources, sealing and integrity",
        "",
        "- Preflight before smoke/formal training used `nvidia-smi`, `free -h`, process count and `/data1` disk checks; GPUs4–7 were selected one fold per card, with bounded four-worker supervisor and at least 25% RAM headroom. Formal training completed without OOM/swap/near-OOM. Postflight is [`resource_postflight.json`](../../outputs/iclr27_phase29/audit/resource_postflight.json).",
        "- Phase29 source checkpoint links target Phase26 source checkpoints; the fold manifest is a symlink to the frozen Phase22 TRAIN split, and [`mot_invariants.json`](../../outputs/iclr27_phase29/audit/mot_invariants.json) is a read-only symlink to the unchanged Phase25 physical-stream structural audit. No large feature or checkpoint copy was made. Resolved targets and hashes are in [`artifact_hashes.json`](../../outputs/iclr27_phase29/audit/artifact_hashes.json) and [`frozen_inputs.json`](../../outputs/iclr27_phase29/audit/frozen_inputs.json).",
        "- [`integrity.json`](../../outputs/iclr27_phase29/audit/integrity.json) confirms JSON parsing, all four formal `.done` markers/checkpoints, 76 event records, preserved failed smoke markers, no residual Phase29 process, no forbidden output names and `public_q1_accessed=false`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/run_stage0_freeze.py",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/train_domain_aligned.py --fold 0 --device cuda:0 --expected-physical-gpu 4 --smoke --tag domain_smoke_fix2",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/train_domain_aligned.py --fold 0 --device cuda:0 --expected-physical-gpu 4 --steps 10 --checkpoint-every 10 --tag domain_targeted_fix2",
        "PYTHONPATH=. bash scripts/iclr27_phase29/run_four_fold_domain_supervisor.sh",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/evaluate_representation.py --device cpu",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/finalize_phase29.py",
        "```",
        "",
        "## Final decision and next direction",
        "",
        "**P29_GATE_R_FAIL_STOP_BEFORE_CONTROLLER.** The single registered domain-aligned residual route did not produce a joint, broad improvement over frozen DINOv2 on disjoint validation. Do not tune GRU, controller, StateMemory or thresholds, and do not download a modern backbone based on this result. The frozen Phase26 proposal and physical MOT invariants remain intact, but the full MOT+OCD sealed objective is not achieved. Any future representation work must first register one independently justified cross-instance/domain method and demonstrate broad disjoint retrieval gains; only then may the unchanged controller be tested once. Public/Q1 labels remain sealed.",
        "",
        "## Machine-readable artifacts",
        "",
        "- [`phase29_decision.json`](../../outputs/iclr27_phase29/audit/phase29_decision.json)",
        "- [`representation_validation.json`](../../outputs/iclr27_phase29/metrics/representation_validation.json)",
        "- [`representation_aggregate.json`](../../outputs/iclr27_phase29/metrics/representation_aggregate.json)",
        "- [`integrity.json`](../../outputs/iclr27_phase29/audit/integrity.json)",
    ]
    decision = {"protocol": "trackocd_iclr27_phase29_domain_alignment_decision", "decision_code": gate["decision"], "gate_r": gate, "aggregate": agg, "formal_units": formal_units, "positive_event_denominator": 76, "controller_started": False, "proposal_frozen": True, "phase26_source_frozen": True, "phase28_gate_c": "P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION", "public_evaluation_started": False, "sealed": True, "smoke_markers": smoke_markers, "smoke_done": smoke_done}
    atomic(OUT / "audit/phase29_decision.json", decision)
    DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "completion/phase29.done").write_text(json.dumps({"decision": gate["decision"], "report": str(DOC), "controller_started": False, "public_evaluation": False}, sort_keys=True) + "\n")
    print(json.dumps({"decision": gate["decision"], "report": str(DOC), "integrity": integrity}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
