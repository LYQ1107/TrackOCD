"""Audit: joint objectness must not equal max known-class confidence."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = json.loads(Path(args.joint_stats).read_text())
    base = np.asarray([r["base_known_conf"] for r in stats], dtype=np.float64)
    joint = np.asarray([r["joint_objectness"] for r in stats],
                       dtype=np.float64)
    by_action = defaultdict(list)
    for r in stats:
        by_action[r["action"]].append(
            (r["base_known_conf"], r["joint_objectness"]))

    corr = float(np.corrcoef(base, joint)[0, 1]) if len(base) > 1 else None
    spearman = None
    if len(base) > 1:
        spearman = float(
            __import__("scipy").stats.spearmanr(base, joint).correlation)
    per_action = {}
    for a, pairs in by_action.items():
        b = np.asarray([p[0] for p in pairs])
        j = np.asarray([p[1] for p in pairs])
        per_action[a] = {
            "n": int(len(pairs)),
            "base_known_conf_mean": float(b.mean()),
            "joint_objectness_mean": float(j.mean()),
            "admitted_with_low_known_conf_frac": float(
                np.mean(b < 0.3)),
        }
    low_base_admitted = int(np.sum(base < 0.3))
    # Novel/existing decisions must not require high known-class confidence.
    non_known = np.asarray([r["base_known_conf"] for r in stats
                            if r["action"] in ("new", "existing")],
                           dtype=np.float64)
    result = {
        "n_rows": len(stats),
        "pearson_corr_base_joint": corr,
        "spearman_corr_base_joint": spearman,
        "per_action": per_action,
        "n_admitted_with_low_known_conf": low_base_admitted,
        "frac_admitted_with_low_known_conf": float(
            low_base_admitted / max(len(stats), 1)),
        "non_known_low_known_conf_frac": float(
            np.mean(non_known < 0.3)) if len(non_known) else None,
        "objectness_not_known_conf": bool(
            corr is not None and corr < 0.9
            and (len(non_known) == 0 or np.mean(non_known < 0.3) > 0.1)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
