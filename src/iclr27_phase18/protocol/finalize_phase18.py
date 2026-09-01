"""Assemble machine-readable Phase18 summaries and terminal decision."""
from __future__ import annotations
import itertools, json, math, os, subprocess
from pathlib import Path
from statistics import mean, median
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase18"

def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / "eval" / name).read_text())

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)

def summary(d: dict[str, Any]) -> dict[str, Any]:
    m, k, r = d["metrics"], d.get("known", {}), d.get("reliability", {})
    return {
        "candidate": d.get("candidate", d.get("protocol")), "variant": d.get("variant", "baseline"), "seed": d.get("seed"),
        "commit_ct": m["commit_ct"], "post_prefix_ct": m["post_prefix_ct"],
        "existing_precision": m["existing_precision"], "negative_false_merge_rate": m["negative_false_merge_rate"],
        "correct_categories": m["correct_categories"], "correct_target_videos": m["correct_target_videos"],
        "mean_time_to_correct_commit": m["mean_time_to_correct_commit"], "median_time_to_correct_commit": m["median_time_to_correct_commit"],
        "pre_prefix_defer_rate": m["pre_prefix_defer_rate"], "premature_commit_event_rate": m["premature_commit_event_rate"],
        "unresolved_event_rate": m["unresolved_event_rate"], "mean_state_count": m["mean_state_count"],
        "duplicate_target_births": m["duplicate_target_births"], "merge_count": m["merge_count"],
        "known_category_macro": k.get("category_macro_accuracy_mean", k.get("category_macro_accuracy")),
        "known_micro": k.get("micro_accuracy_mean", k.get("micro_accuracy")),
        "reliability_auroc": r.get("auroc"), "reliability_auprc": r.get("auprc"),
        "reliability_precision": r.get("precision", r.get("precision_at_fold_threshold")),
        "reliability_recall": r.get("recall", r.get("recall_at_fold_threshold")),
    }

def aggregate_repairs(repairs: list[dict[str, Any]], b1: dict[str, Any]) -> dict[str, Any]:
    cats = sorted(b1["metrics"]["by_category"], key=int)
    per_seed = [summary(x) for x in repairs]
    pool = {"correct": 0, "eligible": 0, "post_correct": 0, "post_rows": 0,
            "first_correct": 0, "first_existing": 0, "false_merges": 0,
            "pre_rows": 0, "pre_defer": 0, "premature": 0, "unresolved": 0,
            "latencies": [], "states": [], "duplicates": 0, "merges": 0}
    recovery = []
    for d in repairs:
        m = d["metrics"]
        pool["correct"] += m["commit_ct"]["correct"]; pool["eligible"] += m["commit_ct"]["eligible"]
        pool["post_correct"] += m["post_prefix_ct"]["correct_rows"]; pool["post_rows"] += m["post_prefix_ct"]["rows"]
        pool["false_merges"] += m["negative_false_merges"]; pool["duplicates"] += m["duplicate_target_births"]; pool["merges"] += m["merge_count"]
        events = [e for f in d["fold_results"] for e in f["event_records"]]
        pool["first_correct"] += sum(e["first_commit_correct_existing"] for e in events if e["kind"] == "positive_existing")
        pool["first_existing"] += sum(e["first_commit_after_prefix"] is not None and e["first_commit_after_prefix"]["action"] == "EXISTING_NOVEL" for e in events if e["kind"] == "positive_existing")
        pool["pre_rows"] += sum(e["pre_prefix_rows"] for e in events); pool["pre_defer"] += sum(e["pre_prefix_defer_rows"] for e in events)
        pool["premature"] += sum(e["premature_commit"] for e in events); pool["unresolved"] += sum(e["unresolved_after_prefix"] for e in events)
        pool["latencies"] += [e["time_to_correct_commit"] for e in events if e["kind"] == "positive_existing" and e["time_to_correct_commit"] is not None]
        pool["states"] += [e["state_count"] for e in events]
        hard = [e for e in events if e["kind"] == "positive_existing" and e["pre_prefix_rows"] > 0]
        recovery.append({"eligible": len(hard), "correct_commit": sum(e["first_commit_correct_existing"] for e in hard), "post_prefix_recovered": sum(e["post_prefix_correct_existing_rows"] > 0 for e in hard)})
    base = {c: b1["metrics"]["by_category"][c]["recall"] for c in cats}
    vals = {c: [d["metrics"]["by_category"][c]["recall"] for d in repairs] for c in cats}
    deltas = np.asarray([[vals[c][i] - base[c] for c in cats] for i in range(3)], np.float64)
    md = deltas.mean(axis=0); rng = np.random.default_rng(1818)
    boot = np.asarray([md[rng.integers(0, len(cats), len(cats))].mean() for _ in range(10000)])
    flips = np.asarray([sum(s * md) / len(cats) for s in itertools.product([-1.0, 1.0], repeat=len(cats))])
    point = float(md.mean())
    uncertainty = {"category_mean_delta_vs_B1": {c: float(md[i]) for i, c in enumerate(cats)}, "category_mean_delta_point": point,
                   "positive_category_count": int((md > 0).sum()), "category_clustered_bootstrap_95": {"low": float(np.quantile(boot, .025)), "high": float(np.quantile(boot, .975)), "clusters": len(cats), "resamples": 10000},
                   "exact_category_sign_flip_p_two_sided": float(np.mean(np.abs(flips) >= abs(point))), "sign_flip_assignments": len(flips), "descriptive_not_external_inference": True}
    pooled = {"commit_ct": {"correct": pool["correct"], "eligible": pool["eligible"], "recall": pool["correct"] / pool["eligible"]},
              "post_prefix_ct": {"correct_rows": pool["post_correct"], "rows": pool["post_rows"], "recall": pool["post_correct"] / pool["post_rows"]},
              "existing_precision": pool["first_correct"] / max(pool["first_existing"], 1), "negative_false_merge_rate": pool["false_merges"] / 123,
              "negative_false_merges": pool["false_merges"], "first_commit_existing_count": pool["first_existing"], "pre_prefix_defer_rate": pool["pre_defer"] / max(pool["pre_rows"], 1),
              "premature_commit_event_rate": pool["premature"] / 123, "unresolved_event_rate": pool["unresolved"] / 123,
              "mean_time_to_correct_commit": mean(pool["latencies"]), "median_time_to_correct_commit": median(pool["latencies"]),
              "mean_state_count": mean(pool["states"]), "duplicate_target_births": pool["duplicates"], "merge_count": pool["merges"],
              "correct_categories_per_seed": [x["correct_categories"] for x in per_seed], "correct_target_videos_per_seed": [x["correct_target_videos"] for x in per_seed]}
    mean_keys = ["known_category_macro", "known_micro", "reliability_auroc", "reliability_auprc", "reliability_precision", "reliability_recall"]
    means = {k: float(np.mean([x[k] for x in per_seed])) for k in mean_keys}
    return {"per_seed": per_seed, "pooled_event_counts": pooled, "mean_metrics": means, "recovery_over_15_unreliable_prefix_events": recovery,
            "uncertainty_vs_B1": uncertainty, "category_recall_by_seed": vals}

def main() -> None:
    b0, b1, b2 = load("b0_historical_replay.json"), load("b1_prereg_baseline.json"), load("b2_prereg_baseline.json")
    main = load("dstm_seed1801_crossfit.json"); b3 = load("b3_seed1801_crossfit.json"); nm = load("no_merge_seed1801_crossfit.json"); nh = load("no_history_seed1801_crossfit.json")
    repairs = [load(f"repair_r1_seed{s}_crossfit.json") for s in (1801, 1802, 1803)]
    baseline = {"protocol": "trackocd_iclr27_phase18_baseline_results", "denominator_sha256": "a37f366368ad534278e164f2d406caf6cf631df3f8f6c2b1a431aad12a57bdf6",
                "B0_historical_controller_replay": summary(b0), "B1_dinov2_causal_tracklet_prototype": summary(b1), "B2_dinov2_trained_pair_scorer": summary(b2), "B2_fit_summary": b2["fit_summary"], "B3_no_defer_no_merge": summary(b3),
                "B3_failed_cycle1": {"root_cause": "empty global anchor reached state projection during calibration", "minimal_repair": "B3 immediate commits write the current observation as an intentionally contaminated anchor", "smoke_passed": True}}
    atomic_json(OUT / "eval/baseline_results.json", baseline)
    agg = aggregate_repairs(repairs, b1)
    public = {"protocol": "trackocd_iclr27_phase18_public_crossfit_result", "public_is_blind_external_test": False,
              "population": {"rows": 43423, "eligible_categories": 11, "positive_events": 41, "negative_events": 41, "reliable_rows": 221},
              "denominator_sha256": "a37f366368ad534278e164f2d406caf6cf631df3f8f6c2b1a431aad12a57bdf6", "fold_sha256": "8601a12b9e95017f2c5a98986160dda01154309808cc3646976057e0ac8bbcfd",
              "strongest_baseline": summary(b1), "main_DSTM_seed1801": summary(main), "selected_repair": "R1", "repair_R1_three_seed_aggregate": agg, "repair_R1_seed_results": [summary(x) for x in repairs],
              "ablations": {"B3": summary(b3), "without_merge": summary(nm), "without_history": summary(nh)}, "main_strata": main["strata"], "repair_R1_strata_by_seed": {str(x["seed"]): x["strata"] for x in repairs},
              "contract_summary": {"main": main["contracts"], "repair": [x["contracts"] for x in repairs]}}
    atomic_json(OUT / "eval/public_crossfit_result.json", public)
    oracle = json.loads((OUT / "eval/oracle_contracts.json").read_text())
    oracle["learned_crossfit_components"] = {"DSTM_seed1801": main["oracles_learned_components"], **{f"R1_seed{x['seed']}": x["oracles_learned_components"] for x in repairs},
                                              "R1_O2_pooled": {"correct": sum(x["oracles_learned_components"]["O2_LEGAL_SEMANTIC_ORACLE_LEARNED_RELIABILITY"]["correct"] for x in repairs), "eligible": 123},
                                              "R1_O3_pooled": {"correct": sum(x["oracles_learned_components"]["O3_LEARNED_SEMANTIC_EXACT_RELIABILITY"]["commit_ct"]["correct"] for x in repairs), "eligible": 123}}
    atomic_json(OUT / "eval/oracle_contracts.json", oracle)
    gate = {"strongest_baseline_commit_correct": 9, "repair_pooled_commit_correct": agg["pooled_event_counts"]["commit_ct"]["correct"], "repair_pooled_commit_eligible": 123,
            "repair_mean_commit_recall": float(np.mean([x["commit_ct"]["recall"] for x in agg["per_seed"]])), "categories_required": 4, "categories_observed_per_seed": agg["pooled_event_counts"]["correct_categories_per_seed"],
            "videos_required": 7, "videos_observed_per_seed": agg["pooled_event_counts"]["correct_target_videos_per_seed"], "existing_precision_mean": float(np.mean([x["existing_precision"] for x in agg["per_seed"]])), "existing_precision_required_min": 0.6723076923076923,
            "false_merge_mean": float(np.mean([x["negative_false_merge_rate"] for x in agg["per_seed"]])), "false_merge_required_max": 0.06878048780487805, "known_macro_mean": agg["mean_metrics"]["known_category_macro"], "known_macro_required_min": 0.08903743315508021,
            "recovery_correct_per_seed": [x["correct_commit"] for x in agg["recovery_over_15_unreliable_prefix_events"]], "all_contracts_valid": True, "passed": False}
    decision = {"protocol": "trackocd_iclr27_phase18_terminal_decision", "decision_code": "P18-T3_PROTOCOL_IDENTIFIABLE_MODEL_FAILS", "task_identifiable": True, "O1_commit_ct": "41/41", "O1_categories": 11,
                "model_learning_success": False, "main_public_gate": False, "selected_repair": "R1_DEFER_ACTION_COLLAPSE", "repair_public_gate": False, "method_gate": gate,
                "devplus_authorized": False, "devplus_accessed": False, "q1_authorized": False, "q1_accessed": False, "iclr_strength_claim_ready": False,
                "bottleneck": "learned semantic state discrimination is unstable across folds/seeds; R1 improves readiness/recovery but has low precision, false merges, sparse category/video coverage, and low known accuracy",
                "next_action": "stop Phase18 memory/lifecycle/decoder tuning; retain the protocol and pursue independent representation/supervision evidence"}
    atomic_json(OUT / "eval/phase18_decision.json", decision)
    atomic_json(OUT / "manifests/external_evaluation_lock.json", {"protocol": "trackocd_iclr27_phase18_external_evaluation_lock", "phase17r_external_boundary_preserved": True, "devplus_authorized": False, "devplus_accessed": False, "q1_authorized": False, "q1_accessed": False, "reason": "P18-T3 public candidate failed method and DEV+ gates; no DEV+ or Q1 labels accessed"})
    try:
        free = subprocess.check_output(["free", "-h"], text=True).strip(); disk = subprocess.check_output(["df", "-h", "/data1"], text=True).strip(); smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    except Exception as e:
        free = disk = smi = f"query failed: {e}"
    units=[]
    for p in sorted(OUT.glob("eval/*_fold*_training.json")):
        d=json.loads(p.read_text()); units.append({"file":p.name,"variant":d["variant"],"seed":d["seed"],"fold":d["fold"],"updates":d["updates"],"passes":d["complete_unique_fit_row_passes"],"elapsed_seconds":d["elapsed_seconds"],"finite_gradient_steps":d["finite_gradient_steps"]})
    resource={"protocol":"trackocd_iclr27_phase18_resource_summary","phase18_output_path":str(OUT),"phase18_output_bytes":sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),"phase18_du_observed":"950M","official_reference_du_observed":"372M","source_data_symlinks_only":True,"free_h":free,"df_data1":disk,"nvidia_smi_at_finalize":smi,"training_units":units,"max_simultaneous_phase18_workers":4,"incidents":["blocking terminal handle reclaimed while surviving children completed; exact markers prevented duplicate launch","B3 cycle1 empty-anchor shape failure; one minimal repair plus smoke and regression passed","no OOM, near-OOM, swap use, or other-user process termination"],"external_gpu_occupancy_noted":True}
    atomic_json(OUT / "eval/resource_summary.json", resource)
    print(json.dumps({"decision": decision["decision_code"], "public": str(OUT/"eval/public_crossfit_result.json"), "report_pending": True}, indent=2))

if __name__ == "__main__": main()
