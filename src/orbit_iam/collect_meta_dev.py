"""Collect ORBIT-IAM train-side meta-dev results into the required CSVs."""
from __future__ import annotations

import argparse
import csv
import json

from src.orbit_msr.evaluate import summarize
from src.iclr27_phase4d.long_stream import active_bucket, load_stream_cache
from src.orbit.protocol import load_train_labels
from src.orbit_iam.evaluate_iam import evaluate_long, load_iam_model

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--compat_threshold", type=float, required=True)
    ap.add_argument("--compat_margin", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, ck = load_iam_model(args.checkpoint, args.device)
    rows_out, logs, r_real, r_syn = evaluate_long(
        model, ck, args.device, 0.5, args.compat_threshold, args.compat_margin)

    def dump(path, rows):
        if not rows:
            return
        fn = []
        for r in rows:
            for k in r:
                if k not in fn:
                    fn.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    out_dir = ROOT + "/outputs/orbit_iam/meta_dev"
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    dump(out_dir + "/real_only_results.csv", [r_real])
    dump(out_dir + "/synthetic_long_stream_results.csv", [r_syn])
    buckets = [r for r in rows_out if r["scope"] in
               ("0-32", "33-128", "129-256", "257+")]
    dump(out_dir + "/scale_bucket_results.csv", buckets)
    # append candidate summary
    summary_path = out_dir + "/candidate_summary.csv"
    cand_rows = []
    if pathlib.Path(summary_path).exists():
        cand_rows = list(csv.DictReader(open(summary_path)))
    cand_rows = [r for r in cand_rows if r.get("name") != args.name]
    overall = [r for r in rows_out if r["scope"] == "overall"][0]
    cand_rows.append({
        "name": args.name,
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
        **{k: overall[k] for k in
           ["known_acc", "rn_acc", "cond_novel_acc", "routing_recall",
            "nmi", "ari", "count_error", "wrong_existing", "first_merge",
            "repeated_false_birth"]},
        "real_rn_acc": r_real["rn_acc"] if r_real else "",
        "real_cond": r_real["cond_novel_acc"] if r_real else "",
        "syn_rn_acc": r_syn["rn_acc"] if r_syn else "",
        "syn_cond": r_syn["cond_novel_acc"] if r_syn else "",
    })
    dump(summary_path, cand_rows)
    print(json.dumps(cand_rows[-1], default=str))


if __name__ == "__main__":
    main()
