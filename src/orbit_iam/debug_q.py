"""Diagnose compatibility score distribution on the long-stream proxy."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_train_labels
from src.iclr27_phase4d.long_stream import load_stream_cache
from src.orbit_iam.evaluate_iam import load_iam_model, run_iam_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    model, ck = load_iam_model(ROOT / args.checkpoint, args.device)
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_iam_stream(model, ck, rows, feats, labels, args.device,
                          gate_thr=0.5, compat_thr=0.5, compat_margin=0.02,
                          syn_mean=syn_mean)
    novel = [l for l in logs if l["role"] == "novel"
             and l["predicted_action"] != "KNOWN"]
    qs = np.asarray([l.get("compat_best", float("nan")) for l in novel])
    qs = qs[np.isfinite(qs)]
    print("novel routed:", len(novel), "q finite:", len(qs))
    if len(qs):
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"q p{p}: {np.percentile(qs, p):.4f}")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"q>={t}: {(qs >= t).mean():.4f}")
    actions = {}
    for l in logs:
        actions[l["predicted_action"]] = actions.get(l["predicted_action"], 0) + 1
    print("actions:", actions)
    print("first rows q:", [round(l.get("compat_best", -1), 3)
                            for l in logs[:8]])


if __name__ == "__main__":
    main()
