"""Legal known-stage classifier/prototype construction for Phase19R."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.iclr27_phase19r.data.stream import Phase19RData


def fit_one(fold: int, final: bool, seed: int, out: Path) -> dict:
    data = Phase19RData(fold, final=final)
    idx = [i for key, cat in data.track_category.items()
           if cat in data.supported_set and cat not in data.held_categories
           and data.track_video[key] in data.fit_videos
           and data.track_role[key] == "supported_known"
           for i in data.track_rows[key]]
    idx = sorted(set(idx)); X = data.raw[idx]; y = np.asarray([int(data.rows[i]["gt_category_id_common"]) for i in idx], np.int64)
    proto = np.zeros_like(data.known_prototypes); bias = np.zeros(len(data.supported_ids), np.float32); counts = np.zeros(len(data.supported_ids), np.int64)
    if len(set(y.tolist())) >= 2:
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=160, tol=1e-4, random_state=seed, n_jobs=8, average=True)
        clf.fit(X, y)
        for row, cat in zip(clf.coef_, clf.classes_):
            j = data.known_to_index[int(cat)]; v = row.astype(np.float32); norm = max(float(np.linalg.norm(v)), 1e-6); v /= norm; proto[j] = v; bias[j] = float(12.0 * clf.intercept_[list(clf.classes_).index(cat)] / norm); counts[j] = int((y == cat).sum())
    # Fall back to the legal centroid for classes that a fold cannot fit.
    for j in range(len(counts)):
        if counts[j] == 0 and data.known_counts[j] > 0:
            proto[j] = data.known_prototypes[j]; counts[j] = data.known_counts[j]
    path = out / ("known_stage_final.npz" if final else f"known_stage_fold{fold}.npz"); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz"); np.savez(tmp, prototypes=proto.astype(np.float32), counts=counts.astype(np.int64), bias=bias.astype(np.float32)); tmp.replace(path)
    result = {"protocol": "trackocd_iclr27_phase19r_known_stage", "fold": fold, "final": final, "seed": seed, "fit_rows": len(idx), "fit_categories": len(set(y.tolist())), "active_supported_known_ids": [data.supported_ids[j] for j, x in enumerate(counts) if x], "prototype_sha256": hashlib.sha256(proto.astype(np.float32).tobytes()).hexdigest(), "true_novel_labels": False}
    summary = out.parent / "metrics" / ("known_stage_final.json" if final else f"known_stage_fold{fold}.json"); summary.parent.mkdir(parents=True, exist_ok=True); summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True)); return result


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, default=0); p.add_argument("--final", action="store_true"); p.add_argument("--seed", type=int, default=1902); p.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase19r/checkpoints")); a = p.parse_args(); fit_one(a.fold, a.final, a.seed, a.out)


if __name__ == "__main__": main()
