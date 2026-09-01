#!/usr/bin/env python
"""Assemble the Phase20 machine decision and self-contained final report."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase20"
DOC = ROOT / "docs/iclr27_phase20"
P19 = ROOT / "outputs/iclr27_phase19r"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fmt(x: Any, n: int = 4) -> str:
    if x is None: return "NA"
    if isinstance(x, bool): return "yes" if x else "no"
    if isinstance(x, float): return f"{x:.{n}f}"
    return str(x)


def main() -> None:
    stage0 = read_json(OUT / "audit/observability_ceiling.json")
    by_prefix = read_json(OUT / "audit/observability_by_prefix.json")
    stage1 = read_json(OUT / "metrics/frozen_correspondence_baseline.json")
    quality = read_json(OUT / "audit/proposal_quality_repair.json")
    shutdown = read_json(OUT / "audit/stage1_duplicate_process_shutdown.json")
    p19 = read_json(P19 / "audit/phase19r_corrective_decision.json")
    expected = [OUT / "audit/observability_ceiling.json", OUT / "audit/observability_by_prefix.json",
                OUT / "audit/observability_events.json", OUT / "audit/observability_events.csv",
                OUT / "audit/proposal_quality_repair.json", OUT / "metrics/frozen_correspondence_baseline.json",
                OUT / "metrics/frozen_correspondence_queries.json", OUT / "manifests/fold_manifest.json",
                OUT / "completion/stage0.done", OUT / "completion/stage1.done", OUT / "completion/proposal_quality_repair.done",
                OUT / "audit/stage1_duplicate_process_shutdown.json"]
    parse_ok = True
    for path in expected:
        if not path.exists() or path.stat().st_size == 0:
            parse_ok = False
        if path.suffix == ".json":
            try: read_json(path)
            except Exception: parse_ok = False

    comparison = {}
    for name in ("mixed_baseline", "event_aligned", "event_repair"):
        x = p19["authoritative_comparison"][name]
        comparison[name] = {
            "commit_ct": x["commit_ct"], "category_coverage_sum": x.get("category_coverage_sum", 0),
            "video_coverage_sum": x.get("video_coverage_sum", 0),
            "existing_precision_mean": x.get("existing_precision_mean"), "existing_recall_mean": x.get("existing_recall_mean"),
            "negative_false_merge_mean": x.get("negative_false_merge_mean"), "known_micro_mean": x.get("known_micro_mean"),
            "known_macro_mean": x.get("known_macro_mean"), "unresolved_mean": x.get("unresolved_mean"),
            "duplicate_births": x.get("duplicate_births"), "folds": x.get("folds", []),
        }

    symlinks = {}
    for rel in ("data/iclr27_phase19r/sources/public_rows_corrected.csv", "data/iclr27_phase19r/sources/public_cls_roi.npz"):
        p = ROOT / rel
        symlinks[rel] = {"is_symlink": p.is_symlink(), "link_target": os.readlink(p) if p.is_symlink() else None,
                         "resolved_target": str(p.resolve()), "exists": p.exists()}

    decision = {
        "protocol": "trackocd_iclr27_phase20_proposal_aware_correspondence_v1",
        "decision_code": "P20_GATE_O_FAIL_PROPOSAL_OBSERVABILITY_STOP_BEFORE_CORRESPONDENCE",
        "status": "STOPPED_AFTER_STAGE0_AND_STAGE1_DIAGNOSTICS",
        "gates": {
            "O": {"pass": bool(stage0["gate_o_pass"]), "max_ceiling_recall": stage0["max_positive_ceiling_recall"], "rule": stage0["gate_o_rule"]},
            "R": {"pass": False, "status": "not_opened_because_O_failed"},
            "C": {"pass": False, "status": "not_run_because_O_failed", "persistent_comparator": "Phase19R frozen 2/76"},
        },
        "stage0": {"positive_events": stage0["positive_events"], "negative_events": stage0["negative_events"], "prefix_summary": stage0["prefix_summary"]},
        "stage1": {"training_performed": False, "methods": stage1["methods"], "prefixes": stage1["prefixes"], "metric_artifact": str(OUT / "metrics/frozen_correspondence_baseline.json")},
        "proposal_quality_repair": {"true_ceiling": quality["true_iou_ceiling_at_prefix16"], "proxy_ceiling": quality["quality_proxy_ceiling_at_prefix16"], "repair_gate_pass": quality["repair_gate_pass"]},
        "prior_phase19r": comparison,
        "speed": p19.get("speed", {}),
        "public_status": "sealed; no DEV+, Q1, or public new-model labels read; no public freeze/evaluation artifacts created",
        "training_processes": "none launched in Phase20; no residual Phase19R/Phase20 training processes",
        "resource_preflight": {"nvidia_smi": "10 x A100-SXM4-40GB, 0 MiB used, 0% utilization at preflight", "memory": "120G available / 125G total, swap 0", "oom_or_termination": False},
        "symlink_ledger": symlinks,
        "integrity": {"expected_artifacts_present_and_json_parse": parse_ok, "stage0_done": (OUT / "completion/stage0.done").exists(), "stage1_done": (OUT / "completion/stage1.done").exists(), "quality_repair_done": (OUT / "completion/proposal_quality_repair.done").exists(), "phase19r_public_freeze_artifacts_created": False,
                      "duplicate_stage1_pid_19198_closed": True, "duplicate_stage1_shutdown_artifact": str(OUT / "audit/stage1_duplicate_process_shutdown.json"),
                      "duplicate_stage1_artifacts_changed": bool(shutdown.get("artifacts_changed", True)), "post_shutdown_phase20_processes": shutdown.get("post_shutdown_phase20_processes", [])},
        "next_direction": "repair proposal domain/observability and establish a verifiable cross-instance semantic correspondence baseline before any new online controller; do not tune Phase19R thresholds or memory",
    }
    atomic_json(OUT / "metrics/phase20_aggregate.json", {"protocol": decision["protocol"], "stage0": stage0, "stage1": stage1, "proposal_quality_repair": quality, "phase19r_comparison": comparison})
    atomic_json(OUT / "audit/phase20_decision.json", decision)
    atomic_json(OUT / "manifests/artifact_hashes.json", {str(p.relative_to(ROOT)): sha256(p) for p in expected if p.is_file()})

    ps = stage0["prefix_summary"]
    lines: list[str] = []
    lines += ["# TrackOCD ICLR 2027 — Phase 20 complete report", "", "**Decision:** `P20_GATE_O_FAIL_PROPOSAL_OBSERVABILITY_STOP_BEFORE_CORRESPONDENCE`", "", "## Executive result", "", "Phase20 stopped after the proposal observability audit and frozen correspondence diagnostics.  The real DSCT proposal stream exposes too few legal reliable observations for the persistent CT event denominator: the best perfect-correspondence ceiling is **25/76 = 0.3289** at causal prefix 16.  Because a majority of events remain unobservable, no correspondence encoder, modern-backbone download, controller reconnect, final training, or public evaluation was authorized.", "", "This is an observability result, not a claim that DINOv2 has no semantic information.  Stage1 remains a descriptive representation audit on the same proposal rows.", ""]
    lines += ["## Frozen scope and prior evidence", "", "The Phase19R RC-MS-OCD, StateMemory, thresholds, DEFER/COMMIT semantics, known masks, physical stream, and causal evaluator were read-only comparators.  Phase20 wrote only its own namespace.  Public TRAIN category/video metadata was used to inherit four category-disjoint and video-disjoint pseudo-held folds; DEV+, Q1, and public new-model labels were not read.", "", "Prior frozen facts:", "", "- Phase19R mixed persistent Commit-CT: **2/76**.", "- Phase19R first event-aligned run: **2/76**; second event-repair run: **0/76**.", "- Synthetic episodes showed existing precision near 1 and recall about 0.35–0.42, while persistent existing precision/recall were approximately zero.", "- Phase15S/17R found real DINOv2 offline signal (matched-ROI R@1 about 0.82; complete paired-novel R@1 about 0.45), while Phase17R also found a separate observability ceiling under perfect semantic labels.", "", "The causal question was therefore split into O (proposal evidence), R (cross-instance representation), and C (online controller use).  O must pass before R/C training.", ""]
    lines += ["## Stage 0 — real proposal observability ceiling", "", "Main inputs were the frozen DSCT proposal rows and their existing ROI/CLS cache.  GT-tight boxes were not mixed into the main path.  A row is reliable only under the existing rule `assigned == 1 and row_iou >= 0.5`; every positive and negative event was retained at every causal prefix.", "", "| causal prefix | positive denominator | source reliable | target reliable | perfect-correspondence CT ceiling | ceiling recall | category coverage | video coverage |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in ps:
        lines.append(f"| {x['prefix']} | {x['positive_denominator']} | {x['source_reliable_materialized']} | {x['target_reliably_visible']} | {x['perfect_correspondence_ct_ceiling_correct']} | {x['perfect_correspondence_ct_ceiling_recall']:.4f} | {x['ceiling_category_coverage']} | {x['ceiling_video_coverage']} |")
    lines += ["", "At prefix16, 27/76 events have no source reliable observation and 24/76 still have no target reliable observation in the causal prefix.  The exact event-by-prefix failure reasons, IoUs, track lengths, and area fractions are in [`observability_events.csv`](../../outputs/iclr27_phase20/audit/observability_events.csv) and [`observability_events.json`](../../outputs/iclr27_phase20/audit/observability_events.json).  Mean positive-event source/target track lengths are 12.66/11.75 rows; mean source area fraction is 0.3205 and prefix16 target area fraction is 0.3188.", "", "**Gate O: FAIL.** The maximum ceiling is below the registered majority threshold of 0.50.  The exact denominator is preserved; no event was removed to improve the ceiling.", ""]
    lines += ["## Stage 1 — frozen DINOv2 correspondence baseline", "", "No parameters were fitted and no online threshold was changed.  Each query uses a target track prefix; candidates are from a different physical track and different video in the same held TRAIN-category fold.  CLS/ROI mean, last, and max causal aggregations are all reported; O-observable and non-observable strata are retained in the machine artifact.", "", "| method | prefix | queries | R@1 | R@5 | mAP | pair ROC-AUC | pair PR-AUC | hard-negative gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in stage1["methods"]:
        for p in stage1["prefixes"]:
            x = stage1["metrics"][method][str(p)]
            lines.append(f"| {method} | {p} | {x['query_count']} | {x['r_at_1']:.4f} | {x['r_at_5']:.4f} | {x['mAP']:.4f} | {fmt(x['pair_roc_auc'])} | {fmt(x['pair_pr_auc'])} | {x['hard_negative_positive_minus_negative_gap']:.4f} |")
    lines += ["", "ROI improves the prefix-1 frozen R@1 over CLS (0.3026 vs 0.2500), but neither representation is a Gate-R result: the protocol does not use these diagnostics to tune the controller, and O already fails.  These values show some measurable correspondence signal on visible proposals while the dominant event-level limitation is missing reliable evidence.", ""]
    lines += ["## Stage H — one TRAIN-only proposal-quality repair", "", "A fold-local logistic proposal-quality head was trained on public TRAIN fit-video rows using only causal score, geometry, prefix-count, stability, and temporal-IoU fields.  It cannot create a proposal or change the stored IoU.  With a fixed 0.5 proxy quality threshold, the proxy ceiling was 31/76, but the true IoU ceiling stayed **25/76**; therefore the repair gate failed and no further correspondence training is justified.", "", "| fold | fit rows | validation rows | validation ROC-AUC | validation PR-AUC | true ceiling events | proxy ceiling events |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for k, x in sorted(quality["folds"].items(), key=lambda z: int(z[0])):
        lines.append(f"| {k} | {x['fit_rows']} | {x['validation_rows']} | {fmt(x['validation_roc_auc'])} | {fmt(x['validation_pr_auc'])} | {x['event_true_ceiling']} | {x['event_proxy_ceiling']} |")
    lines += ["", "The proxy/true distinction is explicit: reprioritizing existing proposals is not evidence that legal observations became available.  A future proposal-domain study must improve the actual reliable-observation ceiling under the same evaluator.", ""]
    lines += ["## Stage 2/3 branch decisions", "", "| gate/branch | status | evidence or reason |", "|---|---|---|", "| O | **FAIL** | maximum perfect-correspondence ceiling 25/76=0.3289 |", "| R | not opened | Stage0 failed; Stage1 is descriptive only |", "| correspondence encoder | not trained | no checkpoints, no 4-GPU training launch |", "| C | not run | no frozen encoder to reconnect; Phase19R comparator remains 2/76 |", "| modern backbone | not downloaded | O failed and temporal ablation did not authorize a backbone change |", "| final/public | sealed | no final 50k and no public/Q1 label access |", ""]
    lines += ["## Phase19R comparator and speed context", "", "| comparator | Commit-CT | category coverage | video coverage | existing precision mean | existing recall mean | negative false-merge mean | known micro mean | known macro mean | unresolved mean | duplicate births |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("mixed_baseline", "event_aligned", "event_repair"):
        x = comparison[name]; ct = x["commit_ct"]
        lines.append(f"| {name} | {ct['correct']}/{ct['eligible']} | {x['category_coverage_sum']} | {x['video_coverage_sum']} | {fmt(x['existing_precision_mean'])} | {fmt(x['existing_recall_mean'])} | {fmt(x['negative_false_merge_mean'])} | {fmt(x['known_micro_mean'])} | {fmt(x['known_macro_mean'])} | {fmt(x['unresolved_mean'])} | {x['duplicate_births']} |")
    lines += ["", "Per-fold Phase19R values are retained in [`phase20_decision.json`](../../outputs/iclr27_phase20/audit/phase20_decision.json); the event-repair 0/76 result is not hidden by an average.", "", "Phase19R acceleration context is recorded without overstating it: old steady throughput was about 0.88–0.91 updates/s; first event-aligned about 1.74–1.81; event-repair about 1.54–1.70.  Hard-pair cache and feature-free episode indices removed generation overhead, and fold-parallel scheduling reduced wall time, but rollout/state-machine CPU work remained.  The event commit-margin loss added overhead.  The strict 2× target was not met by every fold/variant.", ""]
    lines += ["## Reproducibility, resources, and storage", "", "- Fresh preflight: 10 A100-SXM4-40GB GPUs, each 0 MiB used and 0% utilization; memory 120G available of 125G, swap disabled.  Phase20 ran CPU-only diagnostics; no GPU training was launched.", "- No OOM, near-OOM, broad kill, or other-user impact occurred.  One task-owned duplicate Stage1 process was explicitly terminated after completion; it is recorded below and no Phase19R/Phase20 process remains.", "- Integrity follow-up: task-owned duplicate Stage1 PID **19198** (`python scripts/iclr27_phase20/run_stage1_baseline.py`) was observed after `stage1.done` already existed and was stopped with explicit SIGTERM; the second task-owned duplicate PID 19969 (and its shell 19968) was also stopped.  No child or other-user process was touched.  Stage1 artifact hashes, mtimes, and sizes were identical before/after; details are in [`stage1_duplicate_process_shutdown.json`](../../outputs/iclr27_phase20/audit/stage1_duplicate_process_shutdown.json).", "- No Git metadata exists in the project; content hashes are the revision record in [`artifact_hashes.json`](../../outputs/iclr27_phase20/manifests/artifact_hashes.json).", "- Large feature storage was reused by symlink: `data/iclr27_phase19r/sources/public_rows_corrected.csv` resolves to the existing Phase19 source, and `data/iclr27_phase19r/sources/public_cls_roi.npz` resolves to the existing Phase15S feature cache.  Phase20 copied neither features nor checkpoints.", "- Completion markers: [`stage0.done`](../../outputs/iclr27_phase20/completion/stage0.done), [`stage1.done`](../../outputs/iclr27_phase20/completion/stage1.done), and [`proposal_quality_repair.done`](../../outputs/iclr27_phase20/completion/proposal_quality_repair.done).", ""]
    lines += ["## Final decision and limitations", "", "Phase20 establishes an event-level O bottleneck on the registered real proposal stream.  DINOv2 retains offline signal on the subset with visible proposals, so the result does not prove that semantic correspondence is impossible; it proves that representation training cannot be fairly credited for CT when 51/76 prefix16 events lack the joint reliable source/target evidence required by the current evaluator.  The quality-head repair did not improve the true ceiling.", "", "The branch therefore stops before Stage2/3, before any final training, and before public evaluation.  The next phase should first repair proposal-domain/ROI observability and publish a verifiable cross-instance correspondence baseline under the same causal prefixes, then revisit an online state controller.  Continuing Phase19R thresholds, memory, lifecycle, or controller tuning is not authorized by this result.", "", "Machine decision: [`phase20_decision.json`](../../outputs/iclr27_phase20/audit/phase20_decision.json).  Aggregate artifact: [`phase20_aggregate.json`](../../outputs/iclr27_phase20/metrics/phase20_aggregate.json).", ""]
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "PHASE20_PROPOSAL_AWARE_CORRESPONDENCE_COMPLETE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision_code": decision["decision_code"], "gate_o_pass": decision["gates"]["O"]["pass"], "report": str(DOC / "PHASE20_PROPOSAL_AWARE_CORRESPONDENCE_COMPLETE_REPORT.md"), "integrity": parse_ok}, indent=2))


if __name__ == "__main__": main()
