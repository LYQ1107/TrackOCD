"""Recompute the preregistered Phase15A branch from immutable summaries."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SEEDS = [20260824, 20260825]


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    off = json.loads((ROOT / "outputs/iclr27_phase15/eval/phase15a_offline_summary.json").read_text())
    on = json.loads((ROOT / "outputs/iclr27_phase15/eval/phase15a_online_summary.json").read_text())
    p8 = off["representations"]["prefix8"]
    raw_r = p8["raw_cosine"]["retrieval"]["r1"] or 0.0
    raw_m = p8["raw_cosine"]["retrieval"]["map"] or 0.0
    temp_r = p8["temporal_only"]["retrieval"]["r1"] or 0.0
    rel_entries = [p8[f"relation_seed{s}"] for s in SEEDS]
    rel_r = float(np.mean([x["retrieval"]["r1"] or 0.0 for x in rel_entries]))
    rel_m = float(np.mean([x["retrieval"]["map"] or 0.0 for x in rel_entries]))
    rel_auc = float(np.mean([x["pair_cross_video"]["roc_auc"] or 0.0 for x in rel_entries]))
    raw_auc = p8["raw_cosine"]["pair_cross_video"]["roc_auc"] or 0.0
    temp_auc = p8["temporal_only"]["pair_cross_video"]["roc_auc"] or 0.0
    offline_strong = bool(rel_r >= raw_r + .02 and rel_m >= raw_m + .02 and
                          rel_auc >= max(raw_auc, temp_auc) + .01)
    offline_weak = bool(rel_r <= raw_r - .02 or rel_m <= raw_m - .02 or
                        rel_auc <= raw_auc - .01)
    def strict(name):
        return on["candidates"][name]["strict"]
    raw = strict("raw_cosine")
    rel = [strict(f"relation_seed{s}") for s in SEEDS]
    raw_known, raw_ct = float(raw["known_occurrence_acc"]), float(raw["ct_reuse"])
    rel_known = float(np.mean([x["known_occurrence_acc"] for x in rel]))
    rel_ct = float(np.mean([x["ct_reuse"] for x in rel]))
    online_improves = bool(rel_known >= raw_known - .05 and rel_ct > raw_ct)
    known_only = bool(rel_known > raw_known + .01 and rel_ct <= raw_ct)
    if offline_strong and online_improves:
        branch, status = "A", "authorize_phase15b_full_episodic_linker"
    elif offline_strong:
        branch, status = "B", "authorize_phase15b_explicit_three_way_state_probe"
    elif offline_weak:
        branch, status = "D", "run_one_crop_tube_diagnostic_then_stop_architecture_tuning"
    elif known_only:
        branch, status = "C", "authorize_phase15b_novelty_focused_probe"
    else:
        branch, status = "D", "run_one_crop_tube_diagnostic_then_stop_architecture_tuning"
    atomic(ROOT / "outputs/iclr27_phase15/eval/phase15a_decision.json", {
        "protocol": "phase15a", "branch": branch, "status": status,
        "offline_strong": offline_strong, "offline_weak": offline_weak,
        "online_improves": online_improves, "known_only": known_only,
        "criteria": {"offline_r1_margin": .02, "offline_map_margin": .02,
                     "offline_auc_margin": .01, "online_known_floor": -.05,
                     "online_ct_strictly_above_raw": True},
        "evidence": {"raw_r1": raw_r, "relation_mean_r1": rel_r,
                     "raw_map": raw_m, "relation_mean_map": rel_m,
                     "raw_auc": raw_auc, "relation_mean_auc": rel_auc,
                     "temporal_r1": temp_r, "temporal_auc": temp_auc,
                     "raw_known": raw_known, "relation_mean_known": rel_known,
                     "raw_ct_reuse": raw_ct, "relation_mean_ct_reuse": rel_ct},
        "q1_opened": False, "final_gate_passed": False,
    })
    print(json.dumps({"branch": branch, "offline_weak": offline_weak,
                      "status": status}, indent=2))


if __name__ == "__main__":
    main()
