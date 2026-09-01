"""Train-side meta-dev evaluation for ORBIT-IAM candidates.

Runs every candidate on the frozen long-stream proxy, writes:
  outputs/orbit_iam/meta_dev/real_only_results.csv
  outputs/orbit_iam/meta_dev/synthetic_long_stream_results.csv
  outputs/orbit_iam/meta_dev/scale_bucket_results.csv
  outputs/orbit_iam/meta_dev/candidate_summary.csv

No official-validation data is used here.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit_iam.evaluate_iam import evaluate_long, load_iam_model


def row_dict(candidate, scope, r):
    keys = ["name", "known_acc", "rn_acc", "cond_novel_acc", "routing_recall",
            "nmi", "ari", "count_error", "predicted_novel_count",
            "known_to_novel", "novel_to_known", "repeated_false_birth",
            "wrong_existing", "first_merge"]
    out = {"candidate": candidate, "scope": scope}
    for k in keys:
        if k == "name":
            out[k] = r.get(k, "")
        else:
            out[k] = r.get(k, "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--compat_threshold", type=float, default=0.5)
    ap.add_argument("--compat_margin", type=float, default=0.02)
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "orbit_iam" / "meta_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    real_rows = []
    syn_rows = []
    bucket_rows = []
    for cp in args.checkpoints:
        cand = Path(cp).parent.name
        model, ck = load_iam_model(ROOT / cp, args.device)
        rows_out, logs, r_real, r_syn = evaluate_long(
            model, ck, args.device, gate_thr=0.5,
            compat_thr=args.compat_threshold,
            compat_margin=args.compat_margin)
        for r in rows_out:
            scope = r.get("scope", "overall")
            all_rows.append(row_dict(cand, scope, r))
            if scope in ("0-32", "33-128", "129-256", "257+"):
                bucket_rows.append(row_dict(cand, scope, r))
        if r_real:
            real_rows.append(row_dict(cand, "real_only", r_real))
        if r_syn:
            syn_rows.append(row_dict(cand, "synthetic_only", r_syn))
        print(f"[{cand}] overall "
              f"known={r.get('known_acc','')} rn={r.get('rn_acc','')} "
              f"cond={r.get('cond_novel_acc','')} ari={r.get('ari','')} "
              f"we={r.get('wrong_existing','')} fm={r.get('first_merge','')}",
              flush=True)

    def save(name, rows):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    save("candidate_summary.csv", all_rows)
    save("real_only_results.csv", real_rows)
    save("synthetic_long_stream_results.csv", syn_rows)
    save("scale_bucket_results.csv", bucket_rows)
    print("saved meta-dev CSVs", flush=True)


if __name__ == "__main__":
    main()
