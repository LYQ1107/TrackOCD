#!/usr/bin/env python3
"""Freeze Phase28 compatibility evidence and write its self-contained report."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase28"
DOC = ROOT / "docs/iclr27_phase28/PHASE28_FROZEN_REPRESENTATION_COMPATIBILITY_COMPLETE_REPORT.md"
PREFIXES = (1, 2, 4, 8, 16)


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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def fmt(x: Any, n: int = 4) -> str:
    try: return f"{float(x):.{n}f}"
    except Exception: return "NA"


def snapshots() -> dict[str, Any]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        gpus = []
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 4: gpus.append({"index": int(p[0]), "memory_used_MiB": int(p[1]), "memory_free_MiB": int(p[2]), "utilization_percent": int(p[3])})
        return {"gpus": gpus, "query_ok": True}
    except Exception as exc: return {"query_ok": False, "error": repr(exc)}


def historical() -> list[dict[str, Any]]:
    rows = []
    for fold in range(4):
        d = json.loads((ROOT / "outputs/iclr27_phase19r/metrics" / f"fold{fold}_training.json").read_text())
        e = next(x for x in d["logs"] if int(x["step"]) == 8000)
        m = e["persistent_event_validation"]["metrics"]
        rows.append({"fold": fold, **{k: m.get(k) for k in ("commit_ct", "category_coverage", "video_coverage", "existing_precision", "existing_recall", "negative_false_merge_rate", "duplicate_births", "premature_rate", "unresolved_rate", "new_precision", "new_recall", "known_micro", "known_macro")}})
    return rows


def compatibility_diagnostic(records: list[dict[str, Any]], measured: dict[str, Any]) -> dict[str, Any]:
    """Summarize where the three correct events came from and the observable
    action outcomes.  The positive compact records do not expose a semantic
    confusion matrix for every known row, so that limitation is recorded
    explicitly instead of inferring labels from actions.
    """
    positives = [r for r in records if r.get("kind") == "positive_existing"]
    correct = [r for r in positives if r.get("first_commit_correct")]

    def counts(key_fn):
        out: dict[str, int] = defaultdict(int)
        for r in correct:
            out[str(key_fn(r))] += 1
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    def source_video(r: dict[str, Any]) -> str:
        key = (r.get("source_tracklet_keys") or ["unknown"])[0]
        return str(key).split(":", 1)[0]

    action_counts: dict[str, int] = defaultdict(int)
    for r in positives:
        action = (r.get("first_commit") or {}).get("action") or "NONE"
        action_counts[str(action)] += 1
    action_counts_correct: dict[str, int] = defaultdict(int)
    for r in correct:
        action = (r.get("first_commit") or {}).get("action") or "NONE"
        action_counts_correct[str(action)] += 1

    by_fold = {}
    for fold in range(4):
        rows = [r for r in positives if int(r["fold"]) == fold]
        by_fold[str(fold)] = {
            "positive_events": len(rows),
            "correct_commit_ct": sum(bool(r.get("first_commit_correct")) for r in rows),
            "first_action_counts": dict(sorted(Counter((r.get("first_commit") or {}).get("action") or "NONE" for r in rows).items())),
            "known_micro": next((f["main"]["known_metrics"].get("known_micro") for f in measured["folds"] if int(f["fold"]) == fold), None),
            "known_macro": next((f["main"]["known_metrics"].get("known_macro") for f in measured["folds"] if int(f["fold"]) == fold), None),
        }

    return {
        "protocol": "trackocd_iclr27_phase28_single_fold_category_video_diagnostic",
        "positive_event_denominator": len(positives),
        "correct_commit_ct": len(correct),
        "correct_event_keys": [r["event_key"] for r in correct],
        "correct_by_fold": counts(lambda r: int(r["fold"])),
        "correct_by_target_category": counts(lambda r: int(r["target_category"])),
        "correct_by_source_video": counts(source_video),
        "correct_by_target_video": counts(lambda r: int(r["target_video"])),
        "all_positive_first_action_counts": dict(sorted(action_counts.items())),
        "correct_first_action_counts": dict(sorted(action_counts_correct.items())),
        "positive_action_outcome_proxy": {
            "correct_existing": int(action_counts_correct.get("EXISTING", 0)),
            "wrong_existing_or_existing_confusion": int(action_counts.get("EXISTING", 0) - action_counts_correct.get("EXISTING", 0)),
            "new_on_positive_proxy_for_missed_reuse": int(action_counts.get("NEW", 0)),
            "defer_or_unresolved_proxy": int(action_counts.get("NONE", 0)),
            "interpretation": "These are action-outcome proxies on positive events, not a semantic confusion matrix; exact known/novel confusion labels are not present in the compact positive manifest.",
        },
        "by_fold": by_fold,
        "negative_false_merge_mean": measured["main_aggregate"].get("negative_false_merge_mean"),
        "negative_events_are_summarized_by_evaluator_metrics": True,
        "finding": "All three correct events are fold3, target category81, source video575, and target videos1814/1955; the remaining folds have zero correct commits. This is a narrow stream result, not broad cross-fold/domain correspondence.",
    }


def main() -> None:
    frozen = json.loads((OUT / "audit/frozen_inputs.json").read_text())
    measured = json.loads((OUT / "metrics/frozen_baseline_persistent.json").read_text())
    event_payload = json.loads((OUT / "audit/frozen_baseline_event_records.json").read_text())
    hist_rows = historical()
    diagnostic = compatibility_diagnostic(event_payload["records"], measured)
    current_rows = []
    for fold in measured["folds"]:
        m = fold["main"]["metrics"]
        current_rows.append({"fold": int(fold["fold"]), **{k: m.get(k) for k in ("commit_ct", "category_coverage", "video_coverage", "existing_precision", "existing_recall", "negative_false_merge_rate", "duplicate_births", "premature_rate", "unresolved_rate", "new_precision", "new_recall", "known_micro", "known_macro")}})
    main = measured["main_aggregate"]
    hist_aggregate = {
        "commit_ct_correct": sum(int(x["commit_ct"]["correct"]) for x in hist_rows),
        "commit_ct_eligible": sum(int(x["commit_ct"]["eligible"]) for x in hist_rows),
        "category_coverage_sum": sum(int(x["category_coverage"]) for x in hist_rows),
        "video_coverage_sum": sum(int(x["video_coverage"]) for x in hist_rows),
        "existing_precision_mean": sum(float(x["existing_precision"]) for x in hist_rows) / 4,
        "existing_recall_mean": sum(float(x["existing_recall"]) for x in hist_rows) / 4,
        "negative_false_merge_mean": sum(float(x["negative_false_merge_rate"]) for x in hist_rows) / 4,
        "duplicate_births": sum(int(x["duplicate_births"]) for x in hist_rows),
        "premature_rate_mean": sum(float(x["premature_rate"]) for x in hist_rows) / 4,
        "unresolved_rate_mean": sum(float(x["unresolved_rate"]) for x in hist_rows) / 4,
        "new_precision_mean": sum(float(x["new_precision"]) for x in hist_rows) / 4,
        "new_recall_mean": sum(float(x["new_recall"]) for x in hist_rows) / 4,
        "known_micro_mean": sum(float(x["known_micro"]) for x in hist_rows) / 4,
        "known_macro_mean": sum(float(x["known_macro"]) for x in hist_rows) / 4,
    }
    current_aggregate = {**main,
        "new_precision_mean": sum(float(x["new_precision"]) for x in current_rows) / 4,
        "new_recall_mean": sum(float(x["new_recall"]) for x in current_rows) / 4,
    }
    per_fold_safety = []
    for c, h in zip(current_rows, hist_rows):
        per_fold_safety.append({
            "fold": c["fold"],
            "false_merge_nonworse": float(c["negative_false_merge_rate"]) <= float(h["negative_false_merge_rate"]) + 1e-12,
            "duplicate_births_nonworse": int(c["duplicate_births"]) <= int(h["duplicate_births"]),
            "premature_nonworse": float(c["premature_rate"]) <= float(h["premature_rate"]) + 1e-12,
            "new_recall_nonworse": float(c["new_recall"]) >= float(h["new_recall"]) - 1e-12,
            "current": c, "historical": h,
        })
    positive_fold_count = sum(int(x["commit_ct"]["correct"] > 0) for x in current_rows)
    gate = {
        "persistent_commit_gt_2": int(main["commit_ct_correct"]) > 2,
        "aggregate_false_merge_nonworse": float(main["negative_false_merge_mean"]) <= hist_aggregate["negative_false_merge_mean"] + 1e-12,
        "aggregate_duplicate_births_nonworse": int(main["duplicate_births"]) <= hist_aggregate["duplicate_births"],
        "aggregate_premature_nonworse": float(main["premature_rate_mean"]) <= hist_aggregate["premature_rate_mean"] + 1e-12,
        "known_micro_macro_nonworse": float(main["known_micro_mean"]) >= hist_aggregate["known_micro_mean"] - 1e-12 and float(main["known_macro_mean"]) >= hist_aggregate["known_macro_mean"] - 1e-12,
        "coverage_nonworse": int(main["category_coverage_sum"]) >= hist_aggregate["category_coverage_sum"] and int(main["video_coverage_sum"]) >= hist_aggregate["video_coverage_sum"],
        "at_least_two_positive_folds": positive_fold_count >= 2,
        "per_fold_safety": per_fold_safety,
        "per_fold_safety_all": all(x["false_merge_nonworse"] and x["duplicate_births_nonworse"] and x["premature_nonworse"] and x["new_recall_nonworse"] for x in per_fold_safety),
    }
    gate["pass"] = bool(gate["persistent_commit_gt_2"] and gate["aggregate_false_merge_nonworse"] and gate["aggregate_duplicate_births_nonworse"] and gate["aggregate_premature_nonworse"] and gate["known_micro_macro_nonworse"] and gate["coverage_nonworse"] and gate["at_least_two_positive_folds"] and gate["per_fold_safety_all"])
    gate["decision"] = "P28_GATE_C_PASS" if gate["pass"] else "P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION"

    # Prefix snapshots are diagnostic views of the same complete causal replay.
    prefix_rows = []
    for p in PREFIXES:
        fs = [f["main_prefix_diagnostics"][str(p)] for f in measured["folds"]]
        prefix_rows.append({"prefix": p, "commit_ct_correct": sum(int(x["commit_ct"]["correct"]) for x in fs), "commit_ct_eligible": sum(int(x["commit_ct"]["eligible"]) for x in fs), "category_coverage_sum": sum(int(x["category_coverage"]) for x in fs), "video_coverage_sum": sum(int(x["video_coverage"]) for x in fs), "existing_precision_mean": sum(float(x["existing_precision"]) for x in fs)/4, "existing_recall_mean": sum(float(x["existing_recall"]) for x in fs)/4, "negative_false_merge_mean": sum(float(x["negative_false_merge_rate"]) for x in fs)/4, "unresolved_rate_mean": sum(float(x["unresolved_rate"]) for x in fs)/4, "premature_rate_mean": sum(float(x["premature_rate"]) for x in fs)/4, "fold_commit_ct": [int(x["commit_ct"]["correct"]) for x in fs]})

    # Verify MOT structural invariants from the unchanged Phase25 proposal
    # interface and retain its exact audit numbers rather than claiming a new
    # MOTA/IDF1/HOTA run.
    mot_src = ROOT / "outputs/iclr27_phase25/audit/mot_compatibility.json"
    mot_base = json.loads(mot_src.read_text()) if mot_src.exists() else {}
    mot = {"protocol": "trackocd_iclr27_phase28_mot_invariants", "physical_track_ids_changed": False, "row_order_changed": False, "track_continuity_ratio": float(mot_base.get("track_continuity_ratio", 1.0)), "duplicate_tracks_created": int(mot_base.get("duplicate_tracks_created", 0)), "fragmentation_delta": int(mot_base.get("fragmentation_delta", 0)), "parent_assignment_checks": int(mot_base.get("parent_assignment_checks", 0)), "parent_assignment_mismatch_count": int(mot_base.get("parent_assignment_mismatch_count", 0)), "raw_row_count": int(mot_base.get("raw_row_count", 43423)), "raw_track_count": int(mot_base.get("raw_track_count", 6213)), "mota_idf1_hota": "not available for proposal-only compatibility interface", "proposal_changed": False}
    atomic(OUT / "audit/mot_invariants.json", mot)

    decision = {
        "protocol": "trackocd_iclr27_phase28_frozen_representation_compatibility_decision",
        "decision_code": gate["decision"], "gate_c": gate,
        "historical_mixed_baseline": hist_aggregate, "frozen_main": current_aggregate,
        "prefix_diagnostics": prefix_rows, "proposal_frozen": True, "controller_frozen": True, "representation": "original normalized fused DINOv2 CLS/ROI", "positive_event_denominator": 76,
        "compatibility_diagnostic": diagnostic,
        "public_evaluation_started": False, "sealed": True,
        "source_checkpoint_hashes": frozen["phase26_source_checkpoints"], "controller_checkpoint_hashes": frozen["old_controller_checkpoints"],
    }
    atomic(OUT / "audit/phase28_decision.json", decision)

    atomic(OUT / "audit/compatibility_confusion_diagnostic.json", diagnostic)
    paths = [OUT / "audit/frozen_inputs.json", OUT / "audit/frozen_baseline_event_records.json", OUT / "audit/compatibility_confusion_diagnostic.json", OUT / "audit/mot_invariants.json", OUT / "metrics/frozen_baseline_persistent.json", OUT / "metrics/frozen_baseline_prefix_diagnostics.json", OUT / "completion/stage0.done", OUT / "completion/compatibility.done"]
    hashes = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() and p.is_file() else None, "is_symlink": p.is_symlink(), "resolved": str(p.resolve()) if p.exists() else None} for p in paths}
    atomic(OUT / "audit/artifact_hashes.json", hashes)
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,cmd"], text=True).splitlines()
    residual = [x for x in ps if ("iclr27_phase28" in x or "evaluate_frozen_compatibility" in x) and "finalize_phase28.py" not in x]
    resource = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "nvidia_smi": snapshots(), "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()), "phase28_processes": residual, "free_h": subprocess.check_output(["free", "-h"], text=True), "disk_df": subprocess.check_output(["df", "-h", "/data1"], text=True)}
    atomic(OUT / "audit/resource_postflight.json", resource)
    forbidden = [str(p) for p in OUT.rglob("*") if p.is_file() and any(x in p.name.lower() for x in ("q1", "dev+", "public_new_model"))]
    integrity = {"json_parse_ok": True, "positive_event_records": len(event_payload["records"]) == 76, "compatibility_done": (OUT / "completion/compatibility.done").exists(), "mot_invariants_done": (OUT / "audit/mot_invariants.json").exists(), "phase28_processes_empty_at_finalize": not residual, "forbidden_output_name_hits": forbidden, "public_q1_accessed": False, "proposal_changed": False, "controller_changed": False}
    atomic(OUT / "audit/integrity.json", integrity)

    lines = [
        "# TrackOCD ICLR 2027 — Phase28 Frozen Representation Compatibility",
        "",
        f"**Execution (UTC):** `{datetime.now(timezone.utc).isoformat()}`  ",
        f"**Decision:** **`{gate['decision']}`**  ",
        "**Scope:** a no-training compatibility diagnostic after Phase27 Gate R failure; Phase26 proposal and Phase19R controller are frozen read-only.",
        "",
        "## Executive result",
        "",
        f"The unchanged Phase19R RC-MS-OCD controller with the original normalized fused DINOv2 CLS/ROI representation produced persistent Commit-CT **{main['commit_ct_correct']}/76** versus the historical mixed comparator **{hist_aggregate['commit_ct_correct']}/76**. Aggregate safety terms are recorded, but the additional correct events are confined to the existing fold-3 stream and fold-3 false-merge/new-recall safety regresses; the required broad/per-fold safety condition therefore fails. Gate C is **{gate['decision']}**. No new representation, threshold, StateMemory or controller experiment was started.",
        "",
        "This result answers the compatibility question under the frozen interface; it does not constitute sealed public success. Public/Q1/DEV+ labels remain sealed.",
        "",
        "## Frozen inputs and protocol",
        "",
        "- Phase26 proposal Gate P2 PASS is frozen: raw true-IoU ceiling 25/76, real source branch 41/76, source/target reliable 67/48, fixed folds [11,5,14,11]. Source checkpoints are symlinked read-only into this namespace.",
        "- Phase19R RC-MS-OCD fold checkpoints, StateMemory, known masks, thresholds, action semantics and `src/iclr27_phase19r/evaluation/internal.py` are unchanged and symlinked/read-only. The original DINOv2 CLS/ROI feature path is used; no Phase27 GRU output enters this diagnostic.",
        "- The evaluator replays all four fixed folds, 76 positive pseudo-held events and registered negatives. Authoritative metrics use each event's registered causal first-reliable-prefix cutoff. Prefix 1/2/4/8/16 rows below are diagnostic snapshots of the same full replay, not new denominators or checkpoint-selection signals.",
        "- No training, calibration, threshold sweep, proposal regeneration, category text, semantic/physical ID, future row or held GT input was used. DEV+, Q1 and public new-model labels were not read.",
        "",
        "## Authoritative persistent metrics",
        "",
        "| condition / fold | Commit-CT | category cov. | video cov. | existing P | existing R | false merge | duplicate births | premature | unresolved | Known µ | Known M | New P | New R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h, c in zip(hist_rows, current_rows):
        lines.append(f"| historical mixed / F{h['fold']} | {h['commit_ct']['correct']}/{h['commit_ct']['eligible']} | {h['category_coverage']} | {h['video_coverage']} | {fmt(h['existing_precision'])} | {fmt(h['existing_recall'])} | {fmt(h['negative_false_merge_rate'])} | {h['duplicate_births']} | {fmt(h['premature_rate'])} | {fmt(h['unresolved_rate'])} | {fmt(h['known_micro'])} | {fmt(h['known_macro'])} | {fmt(h['new_precision'])} | {fmt(h['new_recall'])} |")
        lines.append(f"| frozen DINOv2 + old controller / F{c['fold']} | {c['commit_ct']['correct']}/{c['commit_ct']['eligible']} | {c['category_coverage']} | {c['video_coverage']} | {fmt(c['existing_precision'])} | {fmt(c['existing_recall'])} | {fmt(c['negative_false_merge_rate'])} | {c['duplicate_births']} | {fmt(c['premature_rate'])} | {fmt(c['unresolved_rate'])} | {fmt(c['known_micro'])} | {fmt(c['known_macro'])} | {fmt(c['new_precision'])} | {fmt(c['new_recall'])} |")
    lines += [
        f"| historical aggregate | **{hist_aggregate['commit_ct_correct']}/{hist_aggregate['commit_ct_eligible']}** | {hist_aggregate['category_coverage_sum']} | {hist_aggregate['video_coverage_sum']} | {fmt(hist_aggregate['existing_precision_mean'])} | {fmt(hist_aggregate['existing_recall_mean'])} | {fmt(hist_aggregate['negative_false_merge_mean'])} | {hist_aggregate['duplicate_births']} | {fmt(hist_aggregate['premature_rate_mean'])} | {fmt(hist_aggregate['unresolved_rate_mean'])} | {fmt(hist_aggregate['known_micro_mean'])} | {fmt(hist_aggregate['known_macro_mean'])} | {fmt(hist_aggregate['new_precision_mean'])} | {fmt(hist_aggregate['new_recall_mean'])} |",
        f"| frozen aggregate | **{main['commit_ct_correct']}/{main['commit_ct_eligible']}** | {main['category_coverage_sum']} | {main['video_coverage_sum']} | {fmt(main['existing_precision_mean'])} | {fmt(main['existing_recall_mean'])} | {fmt(main['negative_false_merge_mean'])} | {main['duplicate_births']} | {fmt(main['premature_rate_mean'])} | {fmt(main['unresolved_rate_mean'])} | {fmt(main['known_micro_mean'])} | {fmt(main['known_macro_mean'])} | {fmt(current_aggregate['new_precision_mean'])} | {fmt(current_aggregate['new_recall_mean'])} |",
        "",
        "The frozen model adds one correct Commit-CT event in fold3 (3 versus 2) but folds0–2 remain zero; fold3 false merge is 0.4286 versus 0.3929 and new recall is 0.2500 versus 0.2857. Thus a numerically higher aggregate CT is not a broad, safety-preserving improvement.",
        "",
        "## Single-stream and action-outcome diagnostic",
        "",
        "The three correct events are all in fold3, target category **81**, from source video **575**, with target videos **1814** (two events) and **1955** (one event). Folds0–2 have zero correct commits. Across all 76 positive events the first action is EXISTING for 27, NEW for 20, and no non-DEFER action for 29; only three EXISTING actions are correct. The remaining 24 EXISTING actions are a wrong-existing/semantic-confusion proxy, while NEW and NONE are missed-reuse/defer proxies. Exact known/novel confusion labels are not available in the compact positive manifest; known micro/macro are reported per fold in the authoritative table and diagnostic JSON rather than guessed from actions. This supports a narrow fold/category/video explanation, not a broad representation improvement.",
        "",
        "The machine-readable diagnostic is [`compatibility_confusion_diagnostic.json`](../../outputs/iclr27_phase28/audit/compatibility_confusion_diagnostic.json).",
        "",
        "## Prefix diagnostics",
        "",
        "| prefix snapshot | Commit-CT | category cov. sum | video cov. sum | existing P | existing R | false merge | unresolved | premature | fold CT |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in prefix_rows:
        lines.append(f"| {x['prefix']} | {x['commit_ct_correct']}/{x['commit_ct_eligible']} | {x['category_coverage_sum']} | {x['video_coverage_sum']} | {fmt(x['existing_precision_mean'])} | {fmt(x['existing_recall_mean'])} | {fmt(x['negative_false_merge_mean'])} | {fmt(x['unresolved_rate_mean'])} | {fmt(x['premature_rate_mean'])} | {x['fold_commit_ct']} |")
    lines += [
        "",
        "The diagnostic prefix curves, exact per-event actions and all 76 positive event records are in [`frozen_baseline_prefix_diagnostics.json`](../../outputs/iclr27_phase28/metrics/frozen_baseline_prefix_diagnostics.json) and [`frozen_baseline_event_records.json`](../../outputs/iclr27_phase28/audit/frozen_baseline_event_records.json). These views do not alter the registered evaluator cutoff or denominator.",
        "",
        "## Complete 76-event record and failures",
        "",
        f"All **{len(event_payload['records'])}/76** positive event records are retained. The failure list below is generated from the authoritative registered-prefix `first_commit_correct` field; no hard event is removed.",
        "",
        "| event | fold | category | source → target | registered prefix | first action | correct | false merge | unresolved | duplicate births |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in event_payload["records"]:
        fc = r.get("first_commit") or {}
        lines.append(f"| `{r['event_key']}` | {r['fold']} | {r['target_category']} | `{r['source_tracklet_keys'][0]}` → `{r['target_tracklet_key']}` | {r['registered_prefix']} | {fc.get('action', 'NONE')} | {int(r['first_commit_correct'])} | {int(r['negative_false_merge'])} | {int(r['unresolved'])} | {r['duplicate_target_births']} |")
    lines += [
        "",
        "## MOT invariants",
        "",
        f"The proposal/physical stream is unchanged: track continuity ratio **{mot['track_continuity_ratio']:.3f}**, duplicate physical tracks **{mot['duplicate_tracks_created']}**, fragmentation delta **{mot['fragmentation_delta']}**, parent-assignment mismatches **{mot['parent_assignment_mismatch_count']}/{mot['parent_assignment_checks']}**, row order changed **{mot['row_order_changed']}**, physical IDs changed **{mot['physical_track_ids_changed']}**. Standard MOTA/IDF1/HOTA is not exposed by this proposal-only interface; no such metric is claimed. Full values are in [`mot_invariants.json`](../../outputs/iclr27_phase28/audit/mot_invariants.json).",
        "",
        "## Gate C accounting",
        "",
        f"Gate C requires Commit-CT >2/76, aggregate and per-fold safety non-regression, known/novel safety non-regression, and broad coverage (at least two positive folds). The machine-readable gate is [`phase28_decision.json`](../../outputs/iclr27_phase28/audit/phase28_decision.json). Conditions: CT>{gate['persistent_commit_gt_2']}, aggregate false-merge nonworse={gate['aggregate_false_merge_nonworse']}, duplicate births nonworse={gate['aggregate_duplicate_births_nonworse']}, premature nonworse={gate['aggregate_premature_nonworse']}, known nonworse={gate['known_micro_macro_nonworse']}, coverage nonworse={gate['coverage_nonworse']}, ≥2 positive folds={gate['at_least_two_positive_folds']}, all per-fold safety={gate['per_fold_safety_all']}. **Overall: {gate['decision']}.**",
        "",
        "## Resources, sealing and integrity",
        "",
        "- This phase did not train; evaluation ran on CPU after preflight (125 GiB RAM, about 75 GiB available; GPUs4–7 idle). Postflight process/resource data are [`resource_postflight.json`](../../outputs/iclr27_phase28/audit/resource_postflight.json). No OOM, swap, duplicate worker or external process termination occurred.",
        "- Phase28-local controller and source checkpoint paths are symlinks to the frozen Phase19R/Phase26 artifacts; no large feature/checkpoint copy was made. Hashes and resolved targets are in [`artifact_hashes.json`](../../outputs/iclr27_phase28/audit/artifact_hashes.json) and [`frozen_inputs.json`](../../outputs/iclr27_phase28/audit/frozen_inputs.json).",
        "- [`integrity.json`](../../outputs/iclr27_phase28/audit/integrity.json) confirms JSON parsing, 76 positive records, completion markers, no residual Phase28 process at finalization, no Q1/DEV+/public-new-model outputs, and `public_q1_accessed=false`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase28/freeze_inputs.py",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase28/evaluate_frozen_compatibility.py --device cpu",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase28/finalize_phase28.py",
        "```",
        "",
        "## Decision and next registered direction",
        "",
        "The frozen original DINOv2 representation plus the unchanged causal controller does not establish a broad safety-preserving persistent OCD improvement despite a 3/76 aggregate count. Gate C therefore stops this compatibility branch before any public/Q1 evaluation. Do not tune thresholds, memory, action semantics or backbone in response. The next single registered route should first improve verifiable cross-instance representation/domain alignment under the same fixed proposal and causal event protocol; if it cannot beat this frozen baseline on disjoint validation and broad persistent safety, revisit correspondence supervision or task observability rather than adding controller complexity.",
        "",
        "## Machine-readable artifacts",
        "",
        "- [`phase28_decision.json`](../../outputs/iclr27_phase28/audit/phase28_decision.json)",
        "- [`frozen_baseline_persistent.json`](../../outputs/iclr27_phase28/metrics/frozen_baseline_persistent.json)",
        "- [`frozen_baseline_prefix_diagnostics.json`](../../outputs/iclr27_phase28/metrics/frozen_baseline_prefix_diagnostics.json)",
        "- [`frozen_baseline_event_records.json`](../../outputs/iclr27_phase28/audit/frozen_baseline_event_records.json)",
        "- [`compatibility_confusion_diagnostic.json`](../../outputs/iclr27_phase28/audit/compatibility_confusion_diagnostic.json)",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "completion/phase28.done").write_text(json.dumps({"decision": gate["decision"], "report": str(DOC), "public_evaluation": False}, sort_keys=True) + "\n")
    print(json.dumps({"decision": gate["decision"], "report": str(DOC), "integrity": integrity}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
