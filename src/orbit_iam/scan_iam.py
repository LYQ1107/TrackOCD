"""Train-side threshold scan for ORBIT-IAM candidates.

Embeddings are computed once per checkpoint; each (compat_threshold,
compat_margin) setting is replayed with the frozen policy.  Only long-stream
meta-dev results are used here; official validation is never touched.
"""
from __future__ import annotations

import argparse
import csv
import json

import torch

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, summarize
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
)
from src.orbit_iam.evaluate_iam import (
    evaluate_long,
    load_iam_model,
    run_iam_stream,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thrs", default="0.35,0.45")
    ap.add_argument("--margins", default="0.0,0.02,0.05,0.1,0.2")
    args = ap.parse_args()
    model, ck = load_iam_model(args.checkpoint, args.device)
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    from src.orbit_iam.evaluate_iam import run_iam_stream as _r
    from src.orbit_msr.evaluate import embed_many
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          args.device)
    rows_out = []
    for thr in [float(x) for x in args.thrs.split(",")]:
        for margin in [float(x) for x in args.margins.split(",")]:
            logs = _r(model, ck, rows, feats, labels, args.device,
                      gate_thr=0.5, compat_thr=thr, compat_margin=margin,
                      syn_mean=syn_mean, zs_rels=(zs, rels))
            attach_gt(logs, gt_rows)
            r = summarize(f"{thr}/{margin}", logs, gt_rows, "overall")
            r_real = summarize(f"{thr}/{margin}", logs, gt_rows, "real_only",
                               select=lambda l: l["role"] == "novel"
                               and int(l["class"]) < 1000000)
            r_syn = summarize(f"{thr}/{margin}", logs, gt_rows, "syn_only",
                              select=lambda l: l["role"] == "novel"
                              and int(l["class"]) >= 1000000)
            row = dict(r)
            row.update({"compat_threshold": thr, "compat_margin": margin,
                        "real_known_acc": r_real["known_acc"] if r_real else None,
                        "real_rn_acc": r_real["rn_acc"] if r_real else None,
                        "real_cond": r_real["cond_novel_acc"] if r_real else None,
                        "real_ari": r_real["ari"] if r_real else None,
                        "syn_cond": r_syn["cond_novel_acc"] if r_syn else None,
                        "syn_ari": r_syn["ari"] if r_syn else None})
            rows_out.append(row)
            print(json.dumps(row, default=str), flush=True)
    keys = list(rows_out[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
