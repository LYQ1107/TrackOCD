"""Paired bootstrap of ORBIT-D1 vs TrackOCD-Ref (Pure Full, GT tracks)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.orbit.protocol import load_gt

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def correctness(preds, gt_rows, assignment):
    gt = {g["sample_id"]: g for g in gt_rows}
    by_sid = {p["sample_id"]: p for p in preds}
    assign = {int(k): int(v) for k, v in (assignment or {}).items()}
    out = {}
    for sid, g in gt.items():
        if g["protocol_role"] == "distractor":
            continue
        p = by_sid.get(sid, {})
        if g["protocol_role"] in ("supported_known", "zero_shot_known"):
            correct = (p.get("prediction_type") == "known"
                       and p.get("semantic_category_id") == g["ground_truth_category_id"])
            out[sid] = (correct, "known")
        else:
            routed = p.get("prediction_type") == "novel"
            correct = False
            if routed and assign.get(int(p.get("virtual_category_id"))) == g["ground_truth_category_id"]:
                correct = True
            out[sid] = (correct, "novel")
    return out


def bootstrap(seed, n=10000, rng_seed=123):
    gt = load_gt("pure")
    ref = json.load(open(ROOT / f"runs/orbit/ref_{seed}.json"))
    orb = json.load(open(ROOT / f"runs/orbit/final_D1_pure_full_{seed}.json"))
    c_ref = correctness(ref["prediction_log"], gt, ref.get("hungarian_assignment", {}))
    c_orb = correctness(orb["prediction_log"], gt, orb.get("hungarian_assignment", {}))
    sids = sorted(c_ref)
    arr_ref = np.array([c_ref[s][0] for s in sids], dtype=float)
    arr_orb = np.array([c_orb[s][0] for s in sids], dtype=float)
    role = np.array([c_ref[s][1] for s in sids])
    rng = np.random.RandomState(rng_seed)
    metrics = {}
    for name, mask in [("all", np.ones(len(sids), dtype=bool)),
                       ("known", role == "known"), ("novel", role == "novel")]:
        diffs = []
        for _ in range(n):
            idx = rng.randint(0, len(sids), size=len(sids))
            d = (arr_orb[idx] - arr_ref[idx])[mask[idx]].mean()
            diffs.append(d)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        metrics[name] = {"mean_diff": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}
    return metrics


def main():
    out = {}
    for seed in ["main_seed1027", "main_seed1028", "main_seed1029"]:
        out[seed] = bootstrap(seed)
        print(seed, out[seed])
    (ROOT / "outputs/orbit/paired_bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
