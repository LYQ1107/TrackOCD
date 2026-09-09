#!/usr/bin/env python3
"""Generate the self-contained Phase88R report and decision artifact."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
REPORT = ROOT / "docs/iclr27_phase88/PHASE88R_EQUAL_BUDGET_H2_REPORT.md"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load(p: Path): return json.loads(p.read_text())


def atomic_json(p: Path, value: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, p)


def main() -> None:
    selection = load(OUT / "audit/h2_equal_train_selection.json")
    frozen = load(OUT / "audit/h2_equal_frozen_selection.json")
    held = load(OUT / "diagnostic/h2_equal_76plus76_diagnostic/metrics.json")
    gcd = load(OUT / "audit/standard_gcd_stream_metrics.json")
    prereg = load(OUT / "audit/hypothesis_2_equal_budget_preregistration.json")
    train_rows = []
    for fold in range(4):
        d = load(OUT / "metrics" / f"h2_equal_f{fold}.json")
        v = load(OUT / "validation" / f"h2_equal_f{fold}_val/final_metrics.json")["metrics"]
        train_rows.append((d, v))
    held_m = held["aggregate_metrics"]
    lines = [
        "# TrackOCD Phase88R — Equal-Budget H2 Report", "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}", "",
        "## Decision", "",
        "**H2_EQUAL_BUDGET TRAIN selection: SELECTED. Held diagnostic: completed but not a broad final success.**",
        "The registered H2 route won the exact TRAIN-disjoint selection score, so one frozen 76-positive + 76-negative diagnostic was allowed. It produced 4/76 Commit-CT, but all four correct events are from fold0 (category coverage 2, video coverage 3); no public/Q1/sealed evaluation was run.", "",
        "Machine-readable artifacts:",
        "- `outputs/iclr27_phase88/audit/hypothesis_2_equal_budget_preregistration.json`",
        "- `outputs/iclr27_phase88/audit/resource_metric_correction.json`",
        "- `outputs/iclr27_phase88/audit/h2_equal_train_selection.json`",
        "- `outputs/iclr27_phase88/audit/h2_equal_frozen_selection.json`",
        "- `outputs/iclr27_phase88/diagnostic/h2_equal_76plus76_diagnostic/metrics.json`",
        "- `outputs/iclr27_phase88/audit/standard_gcd_stream_metrics.json`", "",
        "## Frozen protocol and scope", "",
        "- Four fixed TRAIN video/category-disjoint folds; event_tag `fix2`; seed 88002; support_mode false.",
        "- Each H2 fold resumed its same-fold C0v2 fix2 checkpoint at step 20,000 and ended at step 30,000 (+10,000 updates). Only the known-suppression loss term (weight 1.5) changed.",
        "- Optimizer, model tensors, causal event manifest, row order, evaluator, denominator and action semantics were retained. No threshold, StateMemory, physical tracker, detector or backbone change was made.",
        "- TRAIN validation was used for selection. The held replay was run once after freezing and was not used to tune or select anything.",
        "- Public DEV+, Q1, public new-model labels and sealed labels were not accessed; future rows/tracks, category text, semantic IDs and physical IDs were not model inputs.", "",
        "## Historical frozen comparators (not H2 selection inputs)", "",
        "The earlier same-protocol TRAIN-only comparators are retained for context: C0 continuation mean selection **0.213472**, C1 support mean **0.200465**, and H1 false-merge/reset mean **0.210993**. The earlier C0 held diagnostic was **2/76** with existing precision **1.0**; those held outcomes were not used to select H2. The old extended-budget H2 artifacts remain historical/diagnostic evidence and were not used to initialize the equal-budget route.", "",
        "## Resource-protocol repair", "",
        "The old Phase88 H2 stop artifacts used `MemFree` as the hard trigger. Under shared NumPy memmaps this is not a valid safety estimate because reclaimable page cache can reduce MemFree while MemAvailable remains high. Phase88R records both fields but gates only on `MemAvailable < 31.25 GiB` for three consecutive 30-second samples.", "",
        "The equal-budget supervisor discovered four usable GPUs and ran one bounded worker per fold (GPU0–3 at launch). No external PID was touched. MemFree reached below 1 GiB during the run while MemAvailable stayed above 70 GiB; no OOM or resource pause occurred. A two-step resume smoke passed the expected start-step check and produced atomic checkpoint/metrics.", "",
        "The f0 C0 base checkpoint predates sampler/rollout-state persistence. This is explicitly marked `resume_state_repair_required=true` in the f0 H2 metrics and preregistration; it is not presented as an exact RNG continuation. f1–f3 restored sampler and RNG state. Historical extended-budget H2 checkpoints remain diagnostic-only and did not initialize this route.", "",
        "## Equal-budget training", "",
        "| fold | start→final | loss first→last | checkpoint SHA256 | resume-state repair |",
        "|---:|---:|---:|---|---|",
    ]
    for d, _v in train_rows:
        lines.append(f"| {d['fold']} | {d['start_step']}→{d['updates']} | {d['loss_first']:.6f}→{d['loss_last']:.6f} | `{d['checkpoint_sha256']}` | `{d.get('resume_state_repair_required', False)}` |")
    lines += ["", "## TRAIN-disjoint validation and selection", "", "| fold | C0 selection | H2 selection | Δ H2−C0 | H2 CT recall | H2 neg false merge | category/video coverage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for i, (c, h) in enumerate(zip(selection["c0"]["folds"], selection["h2_equal"]["folds"])):
        lines.append(f"| {i} | {c['selection_score']:.6f} | {h['selection_score']:.6f} | {h['selection_score']-c['selection_score']:.6f} | {h['metrics']['commit_ct']['recall']:.4f} | {h['metrics']['negative_false_merge_rate']:.4f} | {h['metrics']['category_coverage']}/{h['metrics']['video_coverage']} |")
    lines += ["", f"C0 mean selection score: **{selection['c0']['mean_selection_score']:.9f}**; H2 mean: **{selection['h2_equal']['mean_selection_score']:.9f}**; H2 wins {selection['h2_folds_won']}/4 folds. This is the preregistered selection result.", "", "## One held 76+76 diagnostic", "", "| metric | aggregate |", "|---|---:|"]
    for key in ("commit_ct", "existing_precision", "existing_recall", "existing_f1", "negative_false_merge_rate", "anonymous_false_merge_rate", "known_capture_error_rate", "open_world_false_assignment_rate", "premature_rate", "unresolved_rate", "duplicate_births", "category_coverage", "video_coverage", "known_micro", "known_macro"):
        val = held_m.get(key)
        if isinstance(val, dict): val = f"{val.get('correct')}/{val.get('eligible')}"
        elif isinstance(val, float): val = f"{val:.6f}"
        lines.append(f"| {key} | {val} |")
    lines += ["", "| fold | Commit-CT | category coverage | video coverage | negative false assignment |", "|---:|---:|---:|---:|---:|"]
    for f in held["folds"]:
        m = f["metrics"]; lines.append(f"| {f['fold']} | {m['commit_ct']['correct']}/{m['commit_ct']['eligible']} | {m['category_coverage']} | {m['video_coverage']} | {m['open_world_false_assignment_rate']:.6f} |")
    lines += ["", "Interpretation: H2 improves the historical 3/76 count to 4/76 and sharply reduces aggregate false assignment relative to the old C0 diagnostic, but it is concentrated in fold0 and does not establish broad cross-fold OCD. Duplicate births and premature commits remain material. This result is diagnostic-only and does not authorize public/sealed evaluation.", "", "## Auxiliary standard stream metrics", "", "The deterministic `standard_gcd_stream_v1` manifest keeps one StateMemory per fold stream, places known anchors before novel tracks, and never uses category/performance ordering. Anonymous SID tokens are fold-stream-local; global Hungarian is applied only after stream collection. These metrics are reporting-only and were not used for H2 selection.", "", "| metric | aggregate |", "|---|---:|"]
    for key in ("all_acc", "old_acc", "new_acc", "h_score", "nmi", "ari", "rows", "old_rows", "new_rows"):
        val = gcd["aggregate"].get(key); val = f"{val:.6f}" if isinstance(val, float) else val
        lines.append(f"| {key} | {val} |")
    lines += ["", "## Route status and next authorization", "", "- `H2_EQUAL_BUDGET`: TRAIN selection and one held diagnostic complete.",
              "- H3 hierarchical router was not opened because the registered condition for H3 was H2 TRAIN failure.",
              "- No new controller/backbone/threshold experiment, public evaluation or sealed evaluation was run in Phase88R.",
              "- The full TrackOCD research objective is not claimed complete: broad persistent OCD, physical MOT invariants and sealed validation remain outside this route.",
              "- Existing Phase88 historical artifacts, old extended-budget H2 checkpoints, failed supervisor evidence and storage symlink lineage are retained.", "", "## Reproduction", "", "```bash", "python scripts/iclr27_phase88/register_h2_equal_budget.py", "python scripts/iclr27_phase88/build_standard_stream_manifest.py", "python scripts/iclr27_phase88/run_h2_equal_multigpu.py", "python scripts/iclr27_phase88/run_h2_equal_validation.py", "python scripts/iclr27_phase88/freeze_h2_equal_selection.py", "python scripts/iclr27_phase88/evaluate_frozen_76_diagnostic.py --selection outputs/iclr27_phase88/audit/h2_equal_frozen_selection.json --tag h2_equal_76plus76_diagnostic --device cuda:0 --memmap-root /data2/usr_for_deadline/trackocd_phase88/shared_features", "python scripts/iclr27_phase88/evaluate_standard_gcd_stream.py cuda:0", "```", "", "All commands preserve the TRAIN/held boundary and write atomic outputs. Report generated from the machine-readable artifacts above.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(f".{REPORT.name}.tmp.{os.getpid()}"); tmp.write_text("\n".join(lines)); os.replace(tmp, REPORT)
    decision = {
        "schema_version": "trackocd.phase88r.decision.v1", "phase": 88, "route": "H2_EQUAL_BUDGET",
        "status": "H2_TRAIN_SELECTED_HELD_DIAGNOSTIC_COMPLETE", "decision_code": "P88R_H2_TRAIN_WIN_HELD_DIAGNOSTIC_BOUNDED",
        "train_selection": {"c0_mean": selection["c0"]["mean_selection_score"], "h2_mean": selection["h2_equal"]["mean_selection_score"], "h2_folds_won": selection["h2_folds_won"]},
        "held_aggregate_metrics": held_m, "held_used_for_selection": False,
        "gcd_auxiliary": gcd["aggregate"], "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False, "ids_or_text_as_model_input": False,
        "report": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT), "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/phase88r_decision.json", decision)
    atomic_json(OUT / "audit/continuous_state.json", {"phase": 88, "task_status": "IN_PROGRESS", "current_stage": "H2_EQUAL_BUDGET_COMPLETE", "resource_blocker": False, "resource_metric": "MemAvailable", "completed": ["H2_EQUAL_TRAIN", "H2_TRAIN_VALIDATION", "H2_TRAIN_SELECTION", "H2_HELD_DIAGNOSTIC", "STANDARD_GCD_AUXILIARY"], "pending": ["NEXT_AUTHORIZED_RESEARCH_STAGE"], "public_dev_q1_sealed_accessed": False, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(REPORT)


if __name__ == "__main__": main()
