"""Run ORBIT-IAM train-side evaluations and write canonical result CSVs.

Evaluates one checkpoint on:
* synthetic-augmented long-stream proxy (full stream, scale buckets, stages);
* real-only stream (known tracks + 10 real meta-dev novel classes);
* real-novel-only identity probe (only the 63 real novel tracks).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, summarize
from src.orbit_iam.evaluate_iam import (
    evaluate_long,
    load_iam_model,
    run_iam_stream,
)
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
)


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.5)
    ap.add_argument("--compat_margin", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/orbit_iam/meta_dev")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ck = load_iam_model(ROOT / args.checkpoint, args.device)

    # ---- synthetic-augmented long stream ----
    rows_out, logs, r_real, r_syn = evaluate_long(
        model, ck, args.device, args.gate_threshold,
        args.compat_threshold, args.compat_margin)
    for r in rows_out:
        r["tag"] = args.tag
        r["compat_thr"] = args.compat_threshold
        r["compat_margin"] = args.compat_margin
    _write(out_dir / "synthetic_long_stream_results.csv", rows_out)
    (out_dir / f"logs_{args.tag}.json").write_text(
        json.dumps(logs, indent=1, default=str))

    # ---- scale buckets (from long stream logs) ----
    bucket_rows = []
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        sel = [l for l in logs if active_bucket(l["active_novel_prototypes"])
               == bucket]
        if not sel:
            continue
        gt = [g for g in _gt_rows() if g["sample_id"] in {l["sample_id"]
                                                          for l in sel}]
        r = summarize(ck.get("variant", "IAM"), sel, gt, bucket)
        if r:
            r["tag"] = args.tag
            bucket_rows.append(r)
    _write(out_dir / "scale_bucket_results.csv", bucket_rows)

    # ---- real-only stream (known + real meta-dev novel) ----
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    real_rows = [r for r in rows
                 if r["role"] == "known" or int(r["class"]) < 1000000]
    logs_r = run_iam_stream(model, ck, real_rows, feats, labels, args.device,
                            gate_thr=args.gate_threshold,
                            compat_thr=args.compat_threshold,
                            compat_margin=args.compat_margin,
                            syn_mean=syn_mean)
    gt_by_sid = {g["sample_id"]: g for g in gt_rows}
    gt_r = [gt_by_sid[r["sample_id"]] for r in real_rows
            if r["sample_id"] in gt_by_sid]
    attach_gt(logs_r, gt_r)
    real_out = []
    r = summarize(ck.get("variant", "IAM"), logs_r, gt_r, "overall")
    if r:
        r["tag"] = args.tag
        real_out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        sel = [l for l in logs_r
               if active_bucket(l["active_novel_prototypes"]) == bucket]
        if not sel:
            continue
        sids = {l["sample_id"] for l in sel}
        r = summarize(ck.get("variant", "IAM"), sel,
                      [g for g in gt_r if g["sample_id"] in sids], bucket)
        if r:
            r["tag"] = args.tag
            real_out.append(r)
    _write(out_dir / "real_only_results.csv", real_out)

    # ---- real-novel-only identity probe ----
    if r_real:
        r_real = dict(r_real)
        r_real["tag"] = args.tag
    if r_syn:
        r_syn = dict(r_syn)
        r_syn["tag"] = args.tag
    (out_dir / f"identity_probe_{args.tag}.json").write_text(
        json.dumps({"real_novel_only": r_real, "synthetic_novel_only": r_syn},
                   indent=1))

    print("wrote", out_dir, "tag", args.tag)
    for r in rows_out:
        print(r, flush=True)
    print("real_novel_only:", r_real, flush=True)
    print("synthetic_novel_only:", r_syn, flush=True)


def _gt_rows():
    _, gt_rows, _, _ = load_stream_cache()
    return gt_rows


if __name__ == "__main__":
    main()
