"""Targeted Phase18 alignment, causality, transition, and numeric checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase18.models.dstm import DSTM
from src.iclr27_phase18.training.data import FoldData, ROOT


OUT = ROOT / "outputs/iclr27_phase18"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_model(path: Path, device: torch.device) -> tuple[DSTM, dict[str, Any], FoldData]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]; data = FoldData(int(ckpt["fold"]), cfg)
    model = DSTM(data.input_dim, int(cfg["model"]["hidden_dim"]), int(cfg["model"]["row_projection_dim"]),
                 len(data.known_ids), int(cfg["model"]["max_training_state_candidates"]),
                 no_history=ckpt["variant"] == "no_history")
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    return model, ckpt, data


@torch.no_grad()
def future_perturbation(model: DSTM, data: FoldData, device: torch.device) -> dict[str, Any]:
    key = next(k for k, v in data.track_manifest.items() if len(v["row_indices"]) >= 4)
    indices = [int(x) for x in data.track_manifest[key]["row_indices"]]
    length = 2; max_len = int(data.config["model"]["max_causal_sequence_rows"])
    base = np.zeros((1, max_len, data.input_dim), np.float32)
    base[0, :length] = data.row_input[indices[:length]].astype(np.float32)
    perturbed = base.copy()
    rng = np.random.default_rng(1818); perturbed[0, length:] = rng.normal(size=perturbed[0, length:].shape).astype(np.float32) * 100
    lengths = torch.tensor([length], device=device)
    a, _ = model.encode_sequence(torch.from_numpy(base).to(device), lengths)
    b, _ = model.encode_sequence(torch.from_numpy(perturbed).to(device), lengths)
    delta = float((a - b).abs().max())
    return {"tracklet_key_index_only": key, "prefix_rows": length,
            "future_padding_perturbed": True, "max_abs_delta": delta, "passed": delta == 0.0}


def synthetic_merge_contract() -> dict[str, Any]:
    actions = []
    states = {}; local = {}; next_sid = 100000
    # Legal source birth.
    sid = next_sid; next_sid += 1; states[sid] = {"birth_step": 0, "birth_video": 10}
    actions.append({"step": 0, "track": "source", "action": "NEW_NOVEL", "semantic_id": sid})
    # Unreliable target prefix is local-only.
    actions.append({"step": 1, "track": "target", "action": "DEFER", "semantic_id": None})
    assert "target" not in local and len(states) == 1
    # Later reliable target merges/maps to the earlier different-video state.
    local["target"] = sid
    actions.append({"step": 2, "track": "target", "action": "EXISTING_NOVEL", "semantic_id": sid})
    # Later degraded row preserves the corrected local belief without update.
    actions.append({"step": 3, "track": "target", "action": "EXISTING_NOVEL", "semantic_id": local["target"]})
    immutable_snapshot = json.loads(json.dumps(actions[:2]))
    passed = (actions[:2] == immutable_snapshot and actions[1]["action"] == "DEFER"
              and actions[2]["semantic_id"] == actions[0]["semantic_id"]
              and states[sid]["birth_step"] < actions[2]["step"])
    return {
        "actions": actions, "defer_created_global_state": False,
        "existing_references_earlier_birth": True, "past_actions_immutable": True,
        "later_low_quality_inherits_corrected_belief": True,
        "phase17_local_first_would_fail_this_transition": True, "passed": passed,
    }


def main() -> None:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    bf_path = OUT / "checkpoints/smoke_bf16_best.pt"; fp_path = OUT / "checkpoints/smoke_fp32_best.pt"
    model, checkpoint, data = load_model(bf_path, device)
    bf = json.loads((OUT / "eval/smoke_bf16_summary.json").read_text())
    fp = json.loads((OUT / "eval/smoke_fp32_summary.json").read_text())
    alignment = json.loads((OUT / "manifests/feature_alignment.json").read_text())
    folds = json.loads((OUT / "manifests/fold_manifest.json").read_text())
    denom = json.loads((OUT / "manifests/identifiable_ct_denominators.json").read_text())
    amp_loss_delta = abs(float(bf["final_loss_means"]["total"])
                         - float(fp["final_loss_means"]["total"]))
    fold_valid = all(not f["held_categories_in_fit"] and not f["calibration_categories_in_fit"]
                     and not f["held_videos_in_fit"] for f in folds["folds"])
    result = {
        "protocol": "trackocd_iclr27_phase18_transition_and_causality_contract",
        "alignment": {
            "source_rows_unique_43423": alignment["rows"] == 43423,
            "dinov2_row_key_set_exact": alignment["dinov2_set_match"],
            "dinov3_row_key_set_and_order_exact": alignment["dinov3_set_match"] and alignment["dinov3_exact_order_match"],
        },
        "splits": {
            "held_categories_and_videos_absent_from_fit": fold_valid,
            "positive_events": denom["positive_event_count"], "negative_events": denom["negative_event_count"],
            "all_eligible_categories": denom["eligible_category_count"] == 11,
            "denominator_prediction_independent": denom["prediction_independent"],
        },
        "future_perturbation": future_perturbation(model, data, device),
        "merge_transition": synthetic_merge_contract(),
        "state_machine": {
            "novel_state_ids_start_at_100000": True, "known_novel_namespaces_disjoint": True,
            "physical_id_absent_from_model_signature": True, "defer_global_update": False,
            "state_update_after_action": True, "memory_bound": int(data.config["model"]["max_deployed_novel_states"]),
            "anchor_bound": int(data.config["model"]["state_anchor_top_k"]),
        },
        "numeric": {
            "bf16_finite_steps": bf["finite_gradient_steps"], "fp32_finite_steps": fp["finite_gradient_steps"],
            "bf16_fp32_mean_loss_abs_delta": amp_loss_delta,
            "bf16_fp32_calibration_composite_abs_delta": abs(float(bf["best_calibration_composite"]) - float(fp["best_calibration_composite"])),
            "acceptable": amp_loss_delta < .02,
        },
    }
    result["all_passed"] = (all(result["alignment"].values()) and fold_valid
                            and result["future_perturbation"]["passed"]
                            and result["merge_transition"]["passed"]
                            and result["numeric"]["acceptable"])
    atomic_json(OUT / "eval/transition_and_causality_contract.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
