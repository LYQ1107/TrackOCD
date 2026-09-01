"""Evaluate and aggregate a complete four-fold Phase18 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase18.evaluation.baseline_b1 import event_metrics
from src.iclr27_phase18.evaluation.dstm_runtime import evaluate_held
from src.iclr27_phase18.models.dstm import DSTM
from src.iclr27_phase18.training.data import FoldData, ROOT


OUT = ROOT / "outputs/iclr27_phase18"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def load_fold(path: Path, device: torch.device) -> tuple[DSTM, dict[str, Any], FoldData]:
    ckpt = torch.load(path, map_location="cpu"); cfg = ckpt["config"]
    data = FoldData(int(ckpt["fold"]), cfg)
    model = DSTM(data.input_dim, int(cfg["model"]["hidden_dim"]), int(cfg["model"]["row_projection_dim"]),
                 len(data.known_ids), int(cfg["model"]["max_training_state_candidates"]),
                 no_history=ckpt["variant"] == "no_history")
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    return model, ckpt, data


def coverage_risk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for r in records:
        d = r["first_commit_after_prefix"]
        if d is None:
            continue
        if r["kind"] == "positive_existing":
            correct = bool(r["first_commit_correct_existing"])
        else:
            correct = d["action"] == "NEW_NOVEL"
        items.append((float(d.get("confidence", 0.0)), correct))
    if not items:
        return []
    values = np.asarray([x[0] for x in items], np.float32)
    points = []
    for quantile in [0.0, .25, .5, .75, .9]:
        threshold = float(np.quantile(values, quantile))
        chosen = [ok for conf, ok in items if conf >= threshold]
        points.append({"confidence_quantile": quantile, "threshold": threshold,
                       "coverage_over_all_events": len(chosen) / len(records),
                       "covered_commits": len(chosen), "selective_risk": 1.0 - float(np.mean(chosen))})
    return points


def paired_uncertainty(records: list[dict[str, Any]], b1: dict[str, Any]) -> dict[str, Any]:
    main = {x["event_key"]: int(x["first_commit_correct_existing"])
            for x in records if x["kind"] == "positive_existing"}
    base = {x["event_key"]: int(x["first_commit_correct_existing"])
            for x in b1["event_records"] if x["kind"] == "positive_existing"}
    assert set(main) == set(base)
    cats: dict[int, list[float]] = defaultdict(list); videos: dict[int, list[float]] = defaultdict(list)
    for key in sorted(main):
        parts = key.split(":"); cat = int(next(x[1:] for x in parts if x.startswith("c")))
        tv = int(next(x[2:] for x in parts if x.startswith("tv")))
        delta = float(main[key] - base[key]); cats[cat].append(delta); videos[tv].append(delta)
    cat_delta = {c: float(np.mean(v)) for c, v in cats.items()}
    video_delta = {v: float(np.mean(x)) for v, x in videos.items()}
    rng = np.random.default_rng(1818001)
    def boot(values: dict[int, float]) -> dict[str, Any]:
        arr = np.asarray(list(values.values()), np.float64); n = len(arr)
        draws = arr[rng.integers(0, n, size=(10000, n))].mean(1)
        return {"clusters": n, "point_mean": float(arr.mean()),
                "bootstrap_low95": float(np.quantile(draws, .025)),
                "bootstrap_high95": float(np.quantile(draws, .975)), "resamples": 10000}
    arr = np.asarray(list(cat_delta.values()), np.float64); observed = abs(float(arr.mean()))
    perm = []
    for mask in range(1 << len(arr)):
        signs = np.asarray([1 if mask & (1 << i) else -1 for i in range(len(arr))])
        perm.append(abs(float(np.mean(arr * signs))))
    return {
        "paired_event_delta": (sum(main.values()) - sum(base.values())) / len(main),
        "by_category_delta": {str(k): v for k, v in sorted(cat_delta.items())},
        "by_target_video_delta": {str(k): v for k, v in sorted(video_delta.items())},
        "category_clustered_bootstrap": boot(cat_delta),
        "target_video_clustered_bootstrap": boot(video_delta),
        "exact_category_sign_flip_p_two_sided": sum(x >= observed - 1e-12 for x in perm) / len(perm),
        "exact_sign_flip_assignments": len(perm),
        "descriptive_not_external_inference": True,
    }


def strata(records: list[dict[str, Any]], fold_data: dict[int, FoldData]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    event_lookup = {}
    for f, data in fold_data.items():
        for path in [OUT / "episodes/identifiable_positive_events.jsonl"]:
            for line in path.read_text().splitlines():
                e = json.loads(line)
                if int(e["fold"]) == f:
                    event_lookup[e["event_key"]] = (e, data)
    for r in records:
        if r["kind"] != "positive_existing":
            continue
        e, data = event_lookup[r["event_key"]]
        t = data.track_manifest[e["target_tracklet_key"]]; indices = [int(i) for i in t["row_indices"]]
        prefix = int(e["target_first_reliable_prefix_index_gt_only"])
        area = float(np.mean([float(data.rows[i]["area_fraction"]) for i in indices]))
        quality = float(np.mean([float(data.rows[i]["row_iou"]) for i in indices[prefix:]]))
        length = len(indices); correct = int(r["first_commit_correct_existing"])
        groups["object_size"]["small_<.02" if area < .02 else "medium_.02-.10" if area < .10 else "large_>=.10"].append(correct)
        groups["post_prefix_quality"]["low_<.50" if quality < .50 else "medium_.50-.75" if quality < .75 else "high_>=.75"].append(correct)
        groups["tracklet_length"]["short_<=4" if length <= 4 else "medium_5-16" if length <= 16 else "long_>16"].append(correct)
        groups["prefix_length"]["reliable_first_row" if prefix == 0 else "preprefix_1-4" if prefix <= 4 else "preprefix_>4"].append(correct)
    return {name: {key: {"correct": sum(v), "eligible": len(v), "recall": float(np.mean(v))}
                   for key, v in sorted(values.items())} for name, values in groups.items()}


def legacy_stress() -> dict[str, Any]:
    old = json.loads((ROOT / "outputs/iclr27_phase17r/eval/public_final_audit.json").read_text())
    orders = []
    for x in old["orders"]:
        m = x["metrics"]["fixed_ct"]
        orders.append({"seed": x["seed"], "correct": m["correct"], "eligible": m["eligible"],
                       "recall": m["recall"], "denominator_sha256": m["denominator_sha256"]})
    result = {
        "protocol": "trackocd_iclr27_phase18_legacy_immediate_ct_stress",
        "source": str((ROOT / "outputs/iclr27_phase17r/eval/public_final_audit.json").resolve()),
        "source_sha256": sha(ROOT / "outputs/iclr27_phase17r/eval/public_final_audit.json"),
        "replayed_unchanged": True, "reinterpretation_as_phase18_primary": False,
        "orders": orders,
    }
    atomic_json(OUT / "eval/legacy_immediate_ct_stress.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda", args.device)
    fold_results = []; exact_results = []; fold_data = {}; checkpoints = []
    for fold in range(4):
        path = OUT / "checkpoints" / f"{args.prefix}_fold{fold}_best.pt"
        model, ckpt, data = load_fold(path, device); fold_data[fold] = data
        result = evaluate_held(model, data, device, args.variant)
        exact = evaluate_held(model, data, device, args.variant, exact_readiness=True)
        fold_results.append(result); exact_results.append(exact)
        checkpoints.append({"fold": fold, "path": str(path.resolve()), "sha256": sha(path),
                            "step": ckpt["step"], "calibration": ckpt["calibration"]})
        del model; torch.cuda.empty_cache()
    records = [x for f in fold_results for x in f["event_records"]]
    exact_records = [x for f in exact_results for x in f["event_records"]]
    metrics = event_metrics(records); exact_metrics = event_metrics(exact_records)
    # O2: perfect semantic correspondence combined with the actually learned
    # readiness gate. Once readiness fires, the oracle maps to the source state.
    positive = [x for x in records if x["kind"] == "positive_existing"]
    o2_success = 0
    for r in positive:
        prefix = r["pre_prefix_rows"]; target = r["target_decisions"][prefix:]
        if any(d["predicted_ready"] for d in target): o2_success += 1
    known = {
        "per_fold": [x["known"] for x in fold_results],
        "micro_accuracy_mean": float(np.mean([x["known"]["micro_accuracy"] for x in fold_results])),
        "category_macro_accuracy_mean": float(np.mean([x["known"]["category_macro_accuracy"] for x in fold_results])),
    }
    # Unique OOF target row reliability.
    row_values = {}
    for fr in fold_results:
        data = fold_data[fr["fold"]]
        row_map = {r["row_key"]: r for r in data.rows}
        for event in fr["event_records"]:
            if event["kind"] != "positive_existing": continue
            for d in event["target_decisions"]:
                row_values.setdefault(d["row_key"], (d["readiness_score"], bool(row_map[d["row_key"]]["reliable"]), fr["tau_ready"]))
    s = np.asarray([x[0] for x in row_values.values()], np.float32); y = np.asarray([x[1] for x in row_values.values()], bool)
    pred = np.asarray([x[0] >= x[2] for x in row_values.values()], bool)
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
    reliability = {"rows": len(y), "positive_rows": int(y.sum()), "auroc": float(roc_auc_score(y,s)),
                   "auprc": float(average_precision_score(y,s)), "precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1)}
    b1 = json.loads((OUT / "eval/b1_prereg_baseline.json").read_text())
    result = {
        "protocol": "trackocd_iclr27_phase18_public_crossfit_candidate",
        "candidate": args.prefix, "variant": args.variant, "seed": args.seed,
        "public_development_not_external_test": True,
        "checkpoints": checkpoints, "fold_results": fold_results,
        "metrics": metrics, "known": known, "reliability": reliability,
        "coverage_risk": coverage_risk(records), "strata": strata(records, fold_data),
        "uncertainty_vs_B1": paired_uncertainty(records, b1),
        "oracles_learned_components": {
            "O2_LEGAL_SEMANTIC_ORACLE_LEARNED_RELIABILITY": {"correct": o2_success, "eligible": len(positive), "commit_ct_recall": o2_success/len(positive)},
            "O3_LEARNED_SEMANTIC_EXACT_RELIABILITY": exact_metrics,
        },
        "contracts": {"transition_valid_all_folds": all(x["transition_valid"] for x in fold_results),
                      "held_gt_model_input": False, "physical_id_semantic_value": False,
                      "future_frames": False, "denominators_changed": False},
    }
    atomic_json(args.output, result)
    legacy_stress()
    training = [json.loads((OUT / "eval" / f"{args.prefix}_fold{f}_training.json").read_text()) for f in range(4)]
    combined_training = {
        "protocol": "trackocd_iclr27_phase18_main_training_summary", "candidate": args.prefix,
        "seed": args.seed, "variant": args.variant, "folds": training,
        "total_updates": sum(x["updates"] for x in training),
        "all_registered_minimum_complete": all(x["full_registered_run"] and x["complete_unique_fit_row_passes"] >= 10 for x in training),
        "all_gradients_finite": all(x["finite_gradient_steps"] == x["updates"] for x in training),
        "best_steps": [x["best_step"] for x in training],
        "wall_seconds_max_parallel": max(x["elapsed_seconds"] for x in training),
    }
    if args.variant == "dstm" and args.seed == 1801:
        atomic_json(OUT / "eval/main_training_summary.json", combined_training)
    print(json.dumps({"candidate": args.prefix, "metrics": metrics, "known": known,
                      "reliability": reliability, "oracles": result["oracles_learned_components"],
                      "uncertainty": result["uncertainty_vs_B1"]}, indent=2, sort_keys=True))
    return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--prefix",required=True);p.add_argument("--variant",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--device",type=int,default=0)
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())


if __name__ == "__main__": main()
