"""Static gate-threshold Pareto diagnostic for ORBIT-MSRouting (train-side).

Shows whether lowering the static gate threshold merely converts Novel->Known
into Known->Novel / merge / birth errors.  Diagnostic only; never used to
choose the final method.
"""
from __future__ import annotations

import argparse
import csv
import json

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, embed_many, summarize
from src.iclr27_phase4d.long_stream import load_stream_cache
from src.orbit_msrouting.evaluate_msrouting import (
    load_msrouting_checkpoint,
    run_msrouting_stream,
)

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate_thrs", default="0.4,0.45,0.5,0.55")
    args = ap.parse_args()
    model, ck = load_msrouting_checkpoint(args.checkpoint, args.device)
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          args.device)
    out_rows = []
    for gthr in [float(x) for x in args.gate_thrs.split(",")]:
        logs = run_msrouting_stream(
            model, ck, rows, feats, labels, args.device, gate_thr=gthr,
            compat_thr=ck.get("compat_thr", 0.45),
            compat_margin=ck.get("compat_margin", 0.05),
            syn_mean=syn_mean, zs_rels=(zs, rels))
        attach_gt(logs, gt_rows)
        r = summarize(ck.get("variant", "G"), logs, gt_rows, "overall")
        if r is None:
            continue
        out_rows.append({"gate_threshold": gthr, **r})
        print(json.dumps({"gate_threshold": gthr, **r}, default=str),
              flush=True)
    if out_rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
