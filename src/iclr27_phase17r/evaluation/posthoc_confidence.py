"""Result-after-the-fact video bootstrap intervals; never model selection."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SEEDS = (20260825, 20260826, 20260827)


def interval(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "low95": float(np.quantile(values, .025)), "high95": float(np.quantile(values, .975)), "resamples": len(values)}


def main() -> None:
    path = ROOT / "outputs/iclr27_phase17r/eval/public_final_audit.json"; audit = json.loads(path.read_text())
    den = json.loads((ROOT / "outputs/iclr27_phase17r/eval/fixed_ct_denominators.json").read_text())
    rng = np.random.default_rng(1701); cis = {}
    for seed in SEEDS:
        rows = list(csv.DictReader((ROOT / ("outputs/iclr27_phase17r/csv/public_final_audit_decisions_" + str(seed) + ".csv")).open()))
        by_video = defaultdict(list)
        for r in rows: by_video[int(r["video_id"])].append(r)
        videos = sorted(by_video); known_draws, ct_draws, obs_draws = [], [], []
        eligible = set(den["denominators"]["audit"][str(seed)]["row_keys"])
        # Reconstruct births once; state IDs and actions are immutable.
        birth = {}
        for r in rows:
            if r["action"] == "new": birth[int(r["semantic_id"])] = (int(r["gt_category_id_common"]) if r["gt_role_common"] == "novel" else -1, int(r["video_id"]))
        for _ in range(1000):
            chosen = rng.choice(videos, size=len(videos), replace=True); sample = [r for v in chosen for r in by_video[int(v)]]
            known = [int(r["action"] == "known" and int(r["semantic_id"]) == int(r["gt_category_id_common"])) for r in sample if r["gt_role_common"] == "supported_known"]
            ct = []
            for r in sample:
                if r["row_key"] not in eligible: continue
                b = birth.get(int(r["semantic_id"])); ct.append(int(r["action"] == "existing" and b is not None and b[0] == int(r["gt_category_id_common"]) and b[1] != int(r["video_id"])))
            labels = np.asarray([int(r["assigned"]) == 1 and float(r["row_iou"]) >= .5 for r in sample]); scores = np.asarray([float(r["observability_score"]) for r in sample])
            known_draws.append(float(np.mean(known)) if known else 0.0); ct_draws.append(float(np.mean(ct)) if ct else 0.0)
            obs_draws.append(float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else .5)
        cis[str(seed)] = {"known_occurrence_accuracy_video_bootstrap": interval(known_draws), "fixed_ct_recall_video_bootstrap": interval(ct_draws), "observability_auroc_video_bootstrap": interval(obs_draws)}
    audit["confidence_intervals"] = cis
    audit["confidence_intervals_computed_after_full_result"] = True
    audit["confidence_intervals_used_for_selection"] = False
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, path)
    print(json.dumps(cis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
