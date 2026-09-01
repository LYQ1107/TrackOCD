#!/usr/bin/env python3
"""Generate the self-contained Phase72 audit/status artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase72"
DOC = ROOT / "docs/iclr27_phase72/PHASE72_OCD_METRIC_AUDIT_AND_FROZEN_TEST_REPORT.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def metric_value(summary: dict[str, Any], key: str) -> str:
    x = summary.get(key, {})
    return fmt(x.get("value") if isinstance(x, dict) else x)


def event_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| event_key | fold | kind | category | target_video | first_action | first_pos | CT-correct | false_merge | premature | unresolved | duplicate_births |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        first = r.get("first_commit") or {}
        pos = first.get("position", first.get("tracklet_position"))
        lines.append(
            f"| `{r.get('event_key')}` | {r.get('fold')} | {r.get('kind')} | {r.get('target_category')} | {r.get('target_video')} | {first.get('action', 'UNRESOLVED')} | {pos if pos is not None else '—'} | {int(bool(r.get('first_commit_correct')))} | {int(bool(r.get('negative_false_merge')))} | {int(bool(r.get('premature')))} | {int(bool(r.get('unresolved')))} | {r.get('duplicate_target_births', 0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    audit = json.loads((OUT / "audit/q0_p71_interface_audit.json").read_text())
    ocd_audit_path = OUT / "audit/ocd_metric_audit.json"
    ocd_audit = json.loads(ocd_audit_path.read_text())
    summary_path = OUT / "metrics/phase19r_raw_metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    baseline = json.loads((OUT / "metrics/phase19r_raw_baseline.json").read_text())
    records = [r for f in baseline["folds"] for r in f["records"]]
    smoke = json.loads((OUT / "audit/causal_replay_smoke.json").read_text())
    targeted = json.loads((OUT / "audit/causal_replay_targeted_f1.json").read_text())
    p71_decision = json.loads((ROOT / "outputs/iclr27_phase71/audit/phase71_decision.json").read_text())
    q0_checkpoint_path = Path(p71_decision["inputs"]["q0_checkpoint"])
    q0_checkpoint_sha = sha256(q0_checkpoint_path) if q0_checkpoint_path.exists() else "recorded_in_phase71_decision"
    event_hashes = audit["event_manifest_audit"]["manifest_sha256"]

    # Stage C is now complete; update only the new Phase72 audit artifact.
    ocd_audit["phase19r_native_baseline_status"] = "RUN_COMPLETE"
    ocd_audit["phase19r_native_metrics_summary"] = str(summary_path)
    ocd_audit["phase19r_native_event_records"] = str(OUT / "metrics/phase19r_raw_event_records.jsonl")
    ocd_audit["phase19r_native_event_count"] = len(records)
    write_atomic(ocd_audit_path, json.dumps(ocd_audit, indent=2, sort_keys=True) + "\n")

    aggregate = summary["aggregate"]
    by_fold = summary["by_fold"]
    stage_status = {
        "stage_A_schema_key_audit": "PASS",
        "stage_B_pytest_collection": "NO_TESTS_COLLECTED_EXIT5",
        "stage_B_direct_9_case_regression": "PASS",
        "stage_B_causal_smoke": "PASS",
        "stage_B_causal_targeted_fold1": "PASS",
        "stage_C_phase19r_native_frozen_replay": "PASS",
        "stage_D_q0_p71_ocd": "NOT_RUN_INTERFACE_MISMATCH",
        "sealed_evaluation": "NOT_RUN_BLOCKED",
    }
    metrics = {
        "aggregate": aggregate,
        "by_fold": by_fold,
        "optional_trackocd_evaluator": summary["optional_trackocd_evaluator"],
    }
    outputs = {
        "ocd_metric_audit": str(OUT / "audit/ocd_metric_audit.json"),
        "q0_p71_interface_audit": str(OUT / "audit/q0_p71_interface_audit.json"),
        "causal_smoke": str(OUT / "audit/causal_replay_smoke.json"),
        "causal_targeted": str(OUT / "audit/causal_replay_targeted_f1.json"),
        "native_baseline": str(OUT / "metrics/phase19r_raw_baseline.json"),
        "native_summary": str(summary_path),
        "native_event_records": str(OUT / "metrics/phase19r_raw_event_records.jsonl"),
        "report": str(DOC),
    }
    status = {
        "phase": 72,
        "task": "OCD_METRIC_AUDIT_AND_FROZEN_BASELINE_TEST",
        "status": "PHASE72_TEST_COMPLETE_WAITING_FOR_NEXT_STAGE",
        "stage_status": stage_status,
        "cwd": str(ROOT),
        "session": "01a01fb6-96f7-7132-a318-0833180c88d8",
        "inputs": {
            "q0_checkpoint": str(ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"),
            "p71_decision": str(ROOT / "outputs/iclr27_phase71/audit/phase71_decision.json"),
            "event_positive_manifest": str(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"),
            "event_negative_manifest": str(ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"),
            "event_manifest_sha256": event_hashes,
            "q0_checkpoint_sha256": q0_checkpoint_sha,
        },
        "outputs": outputs,
        "metrics": metrics,
        "gate_checks": {
            "schema_and_key_audit": True,
            "positive_events_76": summary["event_contract"]["positive_manifest_count"] == 76,
            "negative_events_76": summary["event_contract"]["negative_manifest_count"] == 76,
            "records_152": summary["event_contract"]["record_count"] == 152,
            "manifest_record_key_sets_equal": summary["event_contract"]["manifest_record_key_sets_equal"],
            "causal_chronology_monotonic": summary["event_contract"]["chronology_all_monotonic"],
            "direct_evaluator_9_cases": True,
            "smoke_contract": bool(smoke["contract_passed"]),
            "targeted_contract": bool(targeted["contract_passed"]),
            "q0_p71_legal_ocd_exporter": False,
            "sealed_public_q1_accessed": False,
        },
        "failure_root_cause": [
            {"attempt": "streaming_audit_v1", "cause": "per-record lstrip scanned the full buffer; interrupted with SIGINT exit 130", "repair": "cursor-based streaming decoder with buffer compaction"},
            {"attempt": "pytest", "cause": "tests/test_trackocd_evaluator.py exposes main() but no pytest test functions; pytest exit 5", "repair": "ran the same 9-case script directly; all 9 passed; old report hash unchanged"},
            {"attempt": "causal_smoke_v1", "cause": "RawPersistentController uses tracklet_position rather than model position", "repair": "checker accepts both legacy causal field names"},
            {"attempt": "causal_smoke_v2", "cause": "set-valued source keys were not JSON serializable", "repair": "sorted list before atomic JSON write"},
            {"attempt": "causal_smoke_v3", "cause": "source and target each start position at zero; checker incorrectly required concatenated global monotonicity", "repair": "check source and target streams independently"},
            {"attempt": "native_summary_v1", "cause": "tuple keys in Counter could not be JSON encoded", "repair": "stringified kind/action keys"},
        ],
        "resource_event": {
            "training_started": False,
            "gpu_used": False,
            "supervisor": "single CPU process for audit/replay; no parallel workers",
            "oom": False,
            "external_processes_touched": False,
            "preflight_file": str(OUT / "audit/stageC_resource_preflight.txt"),
        },
        "next_action": "Engineer and audit a strict per-track causal semantic/action exporter that maps Q0/P71 physical tracks to the TrackOCDEvaluator schema without category_id leakage; then register the next route. Do not reinterpret this diagnostic as P71 OCD or final success.",
    }
    write_atomic(OUT / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")
    for marker in ("stageA_interface_audit.done", "stageB_smoke_regression.done", "stageC_native_replay.done"):
        write_atomic(OUT / "completion" / marker, "done\n")

    q0_summary = next(s for s in audit["tao_stream_summaries"] if "phase4q/q0_long" in s["path"])
    p71_summaries = [s for s in audit["tao_stream_summaries"] if "phase71" in s["path"]]
    q0m = p71_decision["metrics"]["q0_trackeval_macro"]
    p71m = p71_decision["metrics"]["learned_trackeval_macro"]
    report: list[str] = []
    report += [
        "# Phase72 — OCD Metric Audit and Frozen Baseline Test",
        "",
        "**Status:** `PHASE72_TEST_COMPLETE_WAITING_FOR_NEXT_STAGE`  ",
        "**Scope:** read-only interface audit plus frozen Phase19R native diagnostic; no training, threshold search, protocol change, or sealed evaluation.",
        "",
        "## Scope and data boundary",
        "",
        "The fixed Luna/project identity was retained. The only new files are under `outputs/iclr27_phase72/` and `docs/iclr27_phase72/`; Phase19R, Q0 and P71 artifacts were read-only inputs. No DEV+, Q1, public-new or sealed labels were accessed. Category values in the Phase19R event manifests were used only for evaluator denominator/scoring metadata; they were not model features. No physical ID was used as a semantic feature and no future row/track was read.",
        "",
        "Phase71 remains a physical-MOT result, not an OCD result. Its registered decision is `P71_GATE_PHYSICAL_SANITY_FAIL_STOP_BEFORE_SEMANTIC`; this Phase72 test does not alter that gate.",
        "",
        "## Input lineage and hashes",
        "",
        "| input | role | SHA256 / evidence |",
        "|---|---|---|",
        f"| Q0 checkpoint `{q0_checkpoint_path}` | frozen physical anchor | `{q0_checkpoint_sha}` |",
        f"| Q0 TAO stream | physical-only baseline | `{q0_summary['sha256']}`; {q0_summary['records']} records |",
        f"| positive event manifest | evaluator denominator | `{summary['event_contract']['positive_manifest_count']}` events; `{event_hashes['positive']}` |",
        f"| negative event manifest | evaluator safety denominator | `{summary['event_contract']['negative_manifest_count']}` events; `{event_hashes['negative']}` |",
        f"| native replay JSON | frozen Phase19R output | `{summary['input_baseline_sha256']}` |",
        "",
        "The event-manifest SHA256 values are preserved in `outputs/iclr27_phase72/audit/q0_p71_interface_audit.json` and the exact replay JSON hash in `phase19r_raw_metrics_summary.json`.",
        "",
        "## P71 MOT versus Q0 (separate from OCD)",
        "",
        "| metric | Q0 `score_mode=base` | P71 TCO `score_mode=tco` |",
        "|---|---:|---:|",
        f"| top20 IoU≥0.5 recall | {p71_decision['metrics']['q0_top20_iou05']:.6f} | {p71_decision['metrics']['learned_top20_iou05']:.6f} |",
        f"| macro HOTA | {q0m['HOTA']:.6f} | {p71m['HOTA']:.6f} |",
        f"| macro DetA | {q0m['DetA']:.6f} | {p71m['DetA']:.6f} |",
        f"| macro AssA | {q0m['AssA']:.6f} | {p71m['AssA']:.6f} |",
        f"| macro IDF1 | {q0m['IDF1']:.6f} | {p71m['IDF1']:.6f} |",
        f"| IDSW | {q0m['IDSW']:.6f} | {p71m['IDSW']:.6f} |",
        f"| Frag | {q0m['Frag']:.6f} | {p71m['Frag']:.6f} |",
        "",
        "These are the already-frozen P71 validation metrics. They do not contain a semantic/action prediction and are not fed to the OCD evaluator.",
        "",
        "## Stage A — schema, key and leakage audit",
        "",
        "Q0 and each of the four serial P71 TAO streams were parsed with a cursor-based streaming JSON decoder. TrackEval symlink duplicates were deduplicated by resolved path. Every stream has only `bbox`, `category_id`, `image_id`, `score`, `track_id`, `video_id`; none has `prediction_type`, `semantic_category_id`, `virtual_category_id`, `action`, `commit`, or `causal_representation`.",
        "",
        "| stream | records | unique videos | unique physical keys | OCD fields present |",
        "|---|---:|---:|---:|---|",
        f"| Q0 | {q0_summary['records']} | {q0_summary['unique_video_count']} | {q0_summary['unique_track_count']} | none |",
    ]
    for s in p71_summaries:
        fold_name = Path(s["path"]).parents[1].name
        report.append(f"| P71 {fold_name} | {s['records']} | {s['unique_video_count']} | {s['unique_track_count']} | none |")
    report += [
        "",
        "The 76 positive and 76 negative manifests have unique `p19r-pos:`/`p19r-neg:` event keys with fold counts 12/12/24/28 per kind. Their `(video_id, track_id)` keys have zero source or target intersection with Q0 or any P71 TAO stream. This is a real key/lineage mismatch, not a zero OCD score.",
        "",
        "| physical stream | events with any intersection | source intersection | target intersection |",
        "|---|---:|---:|---:|",
    ]
    for s in audit["tao_stream_summaries"]:
        inter = audit["event_manifest_audit"]["tao_intersections"][s["path"]]
        label = "Q0" if "phase4q/q0_long" in s["path"] else "P71 " + Path(s["path"]).parents[1].name
        report.append(f"| {label} | {inter['events_with_any_track_intersection']} | {inter['events_with_source_intersection']} | {inter['events_with_target_intersection']} |")
    report += [
        "",
        "### Evaluator contract evidence",
        "",
        "`src/trackocd_v1/evaluation/trackocd_evaluator.py` requires per-sample/per-track `prediction_type=known` plus `semantic_category_id`, `prediction_type=novel` plus anonymous `virtual_category_id`, or `prediction_type=unresolved`. TAO rows cannot satisfy that contract. Consequently:",
        "",
        "- `q0_p71_ocd_status = NOT_RUN_INTERFACE_MISMATCH`;",
        "- `p71_learned_ocd_status = NOT_RUN_BLOCKED`;",
        "- TrackOCDEvaluator-only metrics are `NOT_APPLICABLE_INTERFACE_MISMATCH`, with null metric values rather than fabricated zeros.",
        "",
        "The Phase19R code audit found `oracle_category` passed from `internal.py:29,36–37` into `StateMemory.apply_action` only as evaluator/state metadata. `runner.py:35–40` calls the model with raw/geometry/quality/known mask/candidate bundle and no category argument; `runner.py:47–58` passes the category only after forward, to state metadata. `state.py:156–164` candidate tensors contain raw/z and count/dispersion/age/anchor/track-video bookkeeping, not category. `state.py:184–200` stores `oracle_birth_category` for scoring/impurity accounting. Thus the fields do not enter forward logits, candidate features, or semantic state tensors. The native evaluator still uses them for correctness denominators, as intended.",
        "",
        "## Stage B — evaluator smoke and causal regression",
        "",
        "The prescribed pytest command was executed and returned exit 5 (`no tests ran`) because `tests/test_trackocd_evaluator.py` is a `main()` script without pytest-collected test functions. The same script was then run directly as the minimal repair: all 9 protocol cases passed, and the pre-existing test report SHA256 remained unchanged.",
        "",
        "A one-positive/one-negative fold0 causal replay and a fold1 targeted replay passed the repaired contract checker. Each preserves source-before-target processing, per-stream monotonic causal positions, target prefix truncation, action/state fields, negative false-merge flag, denominator, and exact raw controller behavior. The smoke/targeted machine-readable outputs are `causal_replay_smoke.json` and `causal_replay_targeted_f1.json`.",
        "",
        "The audit checker required three small, local repairs (legacy `tracklet_position` naming, set serialization, and independent source/target position checks). A separate summary repair stringified tuple Counter keys. All failures and repairs are recorded in `outputs/iclr27_phase72/status.json`; no old-stage file was changed.",
        "",
        "## Stage C — frozen Phase19R native baseline diagnostic",
        "",
        "Command (CPU, one serial process):",
        "",
        "```bash",
        "PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lwr/anaconda3/envs/locatemot/bin/python \\",
        "  scripts/iclr27_phase19r/run_internal_evaluation.py \\",
        "  --candidate raw --device cpu \\",
        "  --out outputs/iclr27_phase72/metrics/phase19r_raw_baseline.json",
        "```",
        "",
        "This result is explicitly named `phase19r_native_frozen_baseline_diagnostic`; it is not P71 learned OCD, Q0 OCD, or final MOT+OCD. All four folds and all 152 events were retained. Manifest/replay key sets match exactly, there are no duplicate or missing event records, and source/target causal positions are monotonic.",
        "",
        "### Aggregate event metrics (numerator/denominator)",
        "",
        "| metric | numerator | denominator | value |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        ("correct persistent Commit-CT", "commit_ct"), ("post-prefix CT", "post_prefix_ct"),
        ("existing precision", "existing_precision"), ("existing recall", "existing_recall"),
        ("existing F1", "existing_f1"), ("new precision", "new_precision"), ("new recall", "new_recall"),
        ("new F1", "new_f1"), ("negative false merge", "negative_false_merge"),
        ("negative false commit (any non-DEFER first action)", "negative_false_commit"),
        ("premature rate", "premature_rate"), ("unresolved rate", "unresolved_rate"),
        ("defer rate", "defer_rate"), ("pre-prefix defer rate", "pre_prefix_defer_rate"),
        ("first action position", "first_action_position"), ("assignment delay after prefix", "assignment_delay_after_prefix"),
        ("duplicate births", "duplicate_births"), ("fragmentation", "fragmentation"),
        ("merge error", "merge_error"), ("NMI", "nmi"), ("ARI", "ari"),
        ("category coverage", "category_coverage"), ("video coverage", "video_coverage"),
    ]
    for label, key in metric_rows:
        x = aggregate[key]
        report.append(f"| {label} | {fmt(x.get('numerator'))} | {fmt(x.get('denominator'))} | {fmt(x.get('value'))} |")
    report += [
        "",
        "The aggregate correct persistent Commit-CT is **1/76 (0.013158)**. This is a frozen Phase19R diagnostic, not a new claim of success. Negative first-action counts are 26 EXISTING, 1 KNOWN, 20 NEW and 29 unresolved; positive counts are 25 EXISTING, 1 KNOWN, 21 NEW and 29 unresolved.",
        "",
        "### Four-fold decomposition",
        "",
        "| fold | positive/negative | Commit-CT | post-prefix CT | false merge | false commit | premature | unresolved | duplicate births | category coverage | video coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in ("0", "1", "2", "3"):
        x = by_fold[f]
        report.append(
            f"| {f} | {x['event_counts']['positive']}/{x['event_counts']['negative']} | {x['commit_ct']['numerator']}/{x['commit_ct']['denominator']} ({x['commit_ct']['value']:.6f}) | {x['post_prefix_ct']['numerator']}/{x['post_prefix_ct']['denominator']} ({x['post_prefix_ct']['value']:.6f}) | {x['negative_false_merge']['numerator']}/{x['negative_false_merge']['denominator']} ({x['negative_false_merge']['value']:.6f}) | {x['negative_false_commit']['numerator']}/{x['negative_false_commit']['denominator']} ({x['negative_false_commit']['value']:.6f}) | {x['premature_rate']['numerator']}/{x['premature_rate']['denominator']} ({x['premature_rate']['value']:.6f}) | {x['unresolved_rate']['numerator']}/{x['unresolved_rate']['denominator']} ({x['unresolved_rate']['value']:.6f}) | {x['duplicate_births']['numerator']} | {x['category_coverage']['numerator']}/{x['category_coverage']['denominator']} | {x['video_coverage']['numerator']}/{x['video_coverage']['denominator']} |")
    report += [
        "",
        "### Complete event-level result",
        "",
        "The following table is generated from the 152 replay records. The full lossless records (including source/target decisions and state snapshots) are in [`phase19r_raw_event_records.jsonl`](../../outputs/iclr27_phase72/metrics/phase19r_raw_event_records.jsonl), with SHA256 recorded in `phase19r_raw_metrics_summary.json`.",
        "",
        event_table(records),
        "",
        "## Stage D status and metric availability",
        "",
        "| item | status | reason |",
        "|---|---|---|",
        "| Q0/P71 OCD replay | `NOT_RUN_INTERFACE_MISMATCH` | TAO physical rows have no legal semantic/action representation exporter and no event-key intersection |",
        "| P71 learned OCD | `NOT_RUN_BLOCKED` | P71 physical gate failed; TCO checkpoint cannot be relabeled as OCD |",
        "| Phase19R native frozen replay | `RUN_COMPLETE` | four folds, 76 positive + 76 negative events, 152 records |",
        "| TrackOCDEvaluator optional per-track metrics | `NOT_APPLICABLE_INTERFACE_MISMATCH` | no strict prediction exporter; values are null |",
        "| sealed/public/Q1 evaluation | `NOT_RUN_BLOCKED` | explicitly out of scope and labels remain sealed |",
        "",
        "The optional TrackOCDEvaluator metric names and null values are in the summary JSON. No unrun metric is represented as zero.",
        "",
        "## Resources, processes and reproducibility",
        "",
        "- No training or GPU job was launched. The Stage C replay used one CPU process (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). The project directory is not a Git worktree (`git status` reported no repository), so no commit hash is claimed.",
        "- Preflight and postflight recorded `free -h`, `nvidia-smi`, process count and `/data1`/`/data2` disk state in [`stageC_resource_preflight.txt`](../../outputs/iclr27_phase72/audit/stageC_resource_preflight.txt) and [`stageC_resource_postflight.txt`](../../outputs/iclr27_phase72/audit/stageC_resource_postflight.txt). No OOM, swap, external-process termination or duplicate Phase72 worker occurred.",
        "- Q0/P71 TAO files were streamed; TrackEval symlink duplicates were deduplicated by real path. No large file was copied or moved.",
        "- Atomic outputs: audit JSON, native metrics summary, event JSONL, status and completion markers are written by temporary-file replacement.",
        "- Machine-readable status: [`status.json`](../../outputs/iclr27_phase72/status.json). Interface audit: [`q0_p71_interface_audit.json`](../../outputs/iclr27_phase72/audit/q0_p71_interface_audit.json). OCD metric contract: [`ocd_metric_audit.json`](../../outputs/iclr27_phase72/audit/ocd_metric_audit.json).",
        "",
        "## Protocol boundary and next action",
        "",
        "Phase56's 4/76 controller result and P71's physical-MOT metrics remain historical, protocol-specific evidence. This Phase72 native replay is the first complete OCD metric pipeline test on the native Phase19R stream, but it is not a P71/Q0 OCD result and does not change any P71 gate. It does not run sealed evaluation and does not represent final MOT+OCD completion.",
        "",
        "The single legal next engineering action is to build and audit a strict per-track causal semantic/action exporter that maps the Q0/P71 physical track lineage to the TrackOCDEvaluator schema while preserving five-field keys, causal chronology, physical/semantic separation and no-category/no-ID model inputs. Only after that contract is proven should a new registered route be considered; no threshold, controller, StateMemory, backbone or candidate lottery is authorized by this Phase72 test.",
        "",
        "**Phase72 completion state:** `PHASE72_TEST_COMPLETE_WAITING_FOR_NEXT_STAGE`. The long-term Luna research session remains open.",
        "",
    ]
    write_atomic(DOC, "\n".join(report))
    print(json.dumps({"status": str(OUT / "status.json"), "report": str(DOC), "event_records": str(OUT / "metrics/phase19r_raw_event_records.jsonl"), "aggregate_commit_ct": aggregate["commit_ct"]}, indent=2))


if __name__ == "__main__":
    main()
