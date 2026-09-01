"""Assemble Phase17R terminal manifests without changing any metric."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase17r"


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    split = json.loads((OUT / "manifests/data_split_and_leakage_audit.json").read_text())
    geometry = json.loads((OUT / "eval/geometry_and_chronology_contract.json").read_text())
    features = json.loads((OUT / "features/full_public_dinov3.json").read_text())
    t0 = json.loads((OUT / "eval/t0_training_summary.json").read_text())
    m1 = json.loads((OUT / "eval/main_training_summary.json").read_text())
    audit = json.loads((OUT / "eval/public_final_audit.json").read_text())
    cal = json.loads((OUT / "eval/public_calibration_summary.json").read_text())
    denom = json.loads((OUT / "eval/fixed_ct_denominators.json").read_text())
    bank = json.loads((OUT / "eval/bank_contamination_audit.json").read_text())
    validity = json.loads((OUT / "eval/phase17_validity_audit.json").read_text())

    training_manifest = {
        "protocol": "trackocd_iclr27_phase17r_training_manifest",
        "population": split["population"], "train_roles": ["known_bank", "novel_correspondence_train"],
        "calibration_roles": ["known_calibration", "novel_calibration"], "audit_roles": ["known_audit", "novel_audit"],
        "features": features,
        "t0": {"summary": str((OUT / "eval/t0_training_summary.json").resolve()), "best_checkpoint": t0["best_checkpoint"], "best_step": t0["best_calibration_step"], "updates": t0["updates"], "physical_gpu": 8,
               "resume_note": "FP16 dynamic-scale path stopped; model checkpoint at step 3000 resumed for remaining 3000 BF16 steps without optimizer-state restoration"},
        "m1": {"summary": str((OUT / "eval/main_training_summary.json").resolve()), "best_checkpoint": m1["best_checkpoint"], "best_step": m1["best_calibration_step"], "updates": m1["updates"], "physical_gpus": [4, 5, 6, 8], "complete_unique_row_passes": m1["complete_unique_row_passes"]},
        "checkpoint_selection": "complete public calibration roles only", "audit_selected_model": False,
        "devplus_or_q1_selected_model": False, "future_frames_used": False, "gt_deployment_input": False,
        "physical_id_semantic_feature": False, "source_dataset_identity_input": False,
        "optimization_seed": 1701, "episode_order_seeds": [20260825, 20260826, 20260827]
    }
    atomic(OUT / "manifests/training_manifest.json", training_manifest)

    order_contracts = []
    for order in audit["orders"]:
        m = order["metrics"]
        order_contracts.append({"seed": order["seed"], "transition": m["transition_contract"], "chronology": m["chronology_contract"], "fixed_ct_denominator_sha256": m["fixed_ct"]["denominator_sha256"]})
    transition = {
        "protocol": "trackocd_iclr27_phase17r_transition_contract", "orders": order_contracts,
        "all_valid": all(x["transition"]["valid"] and x["chronology"]["valid"] for x in order_contracts),
        "explicit_event_rank_consumed": True, "prediction_independent_denominators": True,
        "split_leakage_zero": split["all_disjoint"], "future_input": False, "gt_inference_input": False,
        "past_actions_immutable": True, "physical_id_semantic_feature": False,
        "devplus_run": False, "q1_run": False
    }
    atomic(OUT / "eval/transition_contract.json", transition)

    audit_oracle = audit["oracle_controls"]
    exact_obs = [denom["denominators"]["audit"][str(seed)]["oracle_observable_rows"] for seed in (20260825, 20260826, 20260827)]
    targeted = {
        "protocol": "trackocd_iclr27_phase17r_targeted_retraining_decision", "run": False,
        "dominant_failure": "IMMEDIATE_ACTION_OBSERVABILITY_INFORMATION_LIMIT",
        "evidence": {"fixed_ct_denominators": [denom["denominators"]["audit"][str(seed)]["eligible"] for seed in (20260825, 20260826, 20260827)],
                     "exact_observable_rows_inside_fixed_ct": exact_obs,
                     "m1_oracle_known_plus_observability_ct_correct": [x["metrics"]["fixed_ct"]["correct"] for x in audit_oracle["oracle_both_routing"]],
                     "semantic_correspondence_oracle_ct_correct": [x["metrics"]["fixed_ct"]["correct"] for x in audit_oracle["semantic_correspondence_oracle"]],
                     "semantic_oracle_correct_categories": [x["metrics"]["fixed_ct"]["correct_categories"] for x in audit_oracle["semantic_correspondence_oracle"]]},
        "preregistered_rule": "do not retrain when oracle-known plus oracle-observability reuse fails or an information-theoretic/action-contract blocker is present",
        "why_r1_to_r5_f1_to_f4_not_authorized": "all 30 genuine-order CT rows are below exact IoU 0.5, so the reliable-update action contract exposes zero legal cross-video evidence on the frozen audit denominator; changing the target/policy would change the experiment",
        "architecture_lottery_run": False
    }
    atomic(OUT / "eval/targeted_retraining_summary.json", targeted)

    public_gate = audit["public_gate"]
    decision = {
        "protocol": "trackocd_iclr27_phase17r_training_first", "decision": "P17R-T8_IMMEDIATE_ACTION_CONTRACT_INFORMATION_LIMIT", "status": "complete",
        "mandatory_training": {"t0_updates": t0["updates"], "m1_updates": m1["updates"], "m1_complete_unique_row_passes": m1["complete_unique_row_passes"], "m1_checkpoint_step": cal["checkpoint_step"]},
        "public_gate": public_gate, "public_pass": public_gate["passed"],
        "headline": {"known_closed_top1": audit["offline"]["known_closed_top1"],
                     "known_online_all_orders": [x["metrics"]["known_occurrence_accuracy"] for x in audit["orders"]],
                     "fixed_ct": [str(x["metrics"]["fixed_ct"]["correct"]) + "/" + str(x["metrics"]["fixed_ct"]["eligible"]) for x in audit["orders"]],
                     "predicted_existing_precision": [x["metrics"]["predicted_existing_precision"] for x in audit["orders"]],
                     "exact_observable_fixed_ct_rows": exact_obs,
                     "semantic_oracle_fixed_ct": [str(x["metrics"]["fixed_ct"]["correct"]) + "/" + str(x["metrics"]["fixed_ct"]["eligible"]) for x in audit_oracle["semantic_correspondence_oracle"]]},
        "targeted_retraining": "not authorized by preregistered oracle/information-limit stop",
        "devplus": {"run": False, "reason": "public strengthened gate failed"},
        "q1": {"run": False, "reason": "DEV+ was not authorized"},
        "claim": "Full-data training improves offline known and novel retrieval but cannot satisfy the reliable-observation immediate-action CT contract on a denominator with zero observable rows."
    }
    atomic(OUT / "eval/phase17r_training_first_decision.json", decision)

    disk = shutil.disk_usage("/data1")
    resource = {
        "protocol": "trackocd_iclr27_phase17r_resource_summary",
        "initial": {"gpu_state": "10 A100 40GB idle at initial thread status; unrelated jobs appeared later", "ram_total_gib": 125, "ram_available_gib": 121, "data1_free_gib": 170, "swap_used": 0},
        "gpu_assignments": {"dinov3_full_extraction": [4, 5, 6], "t0_full_and_repair": [7, 8], "m1_ddp": [4, 5, 6, 8]},
        "extraction": {"benchmark_seconds": 29.967605352401733, "benchmark_crops_per_second": 108.58391792520086,
                       "full_shard_wall_seconds": [x["wall_seconds"] for x in features["shard_meta"]], "full_rows": features["rows"], "feature_shape": features["shape"], "artifact_dtype": features["dtype"]},
        "training": {"t0_recorded_resume_wall_seconds": t0["wall_seconds"], "m1_wall_seconds": m1["wall_seconds"], "m1_world_size": m1["world_size"], "m1_amp_dtype": m1["amp_dtype"]},
        "memory": {"prep_post_launch_available_kib": 94888924, "m1_post_launch_available_kib": 108727356, "safety_floor_kib": 33554432, "floor_crossed": False, "oom": False, "near_oom": False, "swap_used": False},
        "storage": {"phase17r_output_bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()), "budget_gib": 40, "current_data1_free_gib": disk.free / 2**30, "minimum_external_checkpoint_floor_gib": 120},
        "repairs_and_incidents": [
            "DINOv3 benchmark direct-file module import failed before model load; rerun as a module.",
            "DINOv3 extraction AMP was non-finite in torch 1.12; FP32 forward was finite/unit norm and used for frozen extraction, with float16 artifacts.",
            "T0 FP16 dynamic loss scaling stopped at step 2 and again at step 3090; the repeated FP16 path was abandoned, latest step-3000 model resumed in BF16 for the remaining 3000 updates without optimizer-state restoration.",
            "GPU 0-3 and later GPU 7 became occupied by unrelated tasks; hard checks changed task devices and no external process was signalled.",
            "One short T0 numerical regression command printed a transient GPU7 occupancy before launch but lacked the later hard-abort guard; it completed without OOM or process interference. This scheduling lapse is disclosed.",
            "No task process was killed for memory pressure; no failed sample, metric, seed, gate, or denominator was changed."
        ],
        "task_owned_processes_terminated": [], "other_user_processes_terminated": [], "new_external_checkpoint_downloads": []
    }
    atomic(OUT / "eval/resource_summary.json", resource)

    hash_paths = [
        "docs/iclr27_phase17r/PROTOCOL.md", "docs/iclr27_phase17r/STORAGE_AND_RESOURCE_LEDGER.md",
        "docs/iclr27_phase17r/TRAINING_FIRST_METHOD.md",
        "outputs/iclr27_phase17r/manifests/preregistration.json", "outputs/iclr27_phase17r/manifests/data_split_and_leakage_audit.json",
        "outputs/iclr27_phase17r/manifests/training_manifest.json", "outputs/iclr27_phase17r/manifests/public_lock.json",
        "outputs/iclr27_phase17r/eval/phase17_validity_audit.json", "outputs/iclr27_phase17r/eval/geometry_and_chronology_contract.json",
        "outputs/iclr27_phase17r/eval/full_paired_baseline.json", "outputs/iclr27_phase17r/eval/main_training_summary.json",
        "outputs/iclr27_phase17r/eval/t0_training_summary.json", "outputs/iclr27_phase17r/eval/targeted_retraining_summary.json",
        "outputs/iclr27_phase17r/eval/bank_contamination_audit.json", "outputs/iclr27_phase17r/eval/public_calibration_summary.json",
        "outputs/iclr27_phase17r/eval/public_final_audit.json", "outputs/iclr27_phase17r/eval/transition_contract.json",
        "outputs/iclr27_phase17r/eval/resource_summary.json", "outputs/iclr27_phase17r/eval/phase17r_training_first_decision.json",
        "outputs/iclr27_phase17r/features/full_public_dinov3.json", "outputs/iclr27_phase17r/checkpoints/m1_best.pt",
        "outputs/iclr27_phase17r/checkpoints/t0_best.pt", "src/iclr27_phase17r/training/model.py",
        "src/iclr27_phase17r/training/train_full_model.py", "src/iclr27_phase17r/evaluation/evaluate_candidate.py",
        "src/iclr27_phase17r/evaluation/posthoc_confidence.py", "src/iclr27_phase17r/evaluation/posthoc_prefix_curves.py"
    ]
    report = ROOT / "docs/iclr27_phase17r/PHASE17R_TRAINING_FIRST_COMPLETE_REPORT.md"
    if report.exists(): hash_paths.append(str(report.relative_to(ROOT)))
    hashes = {p: {"sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in hash_paths if (ROOT / p).exists()}
    symlinks = {}
    for p in (ROOT / "data/iclr27_phase17r").rglob("*"):
        if p.is_symlink(): symlinks[str(p.relative_to(ROOT))] = {"target": os.readlink(p), "resolved": str(p.resolve()), "exists": p.resolve().exists()}
    atomic(OUT / "manifests/artifact_hashes.json", {"protocol": "trackocd_iclr27_phase17r_artifact_hashes", "files": hashes, "symlinks": symlinks, "huge_raw_trees_hashed": False})
    print(json.dumps({"decision": decision["decision"], "public_pass": public_gate["passed"], "hashes": len(hashes), "symlinks": len(symlinks)}, indent=2))


if __name__ == "__main__":
    main()
