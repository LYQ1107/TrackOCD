"""Train-side threshold scan for ORBIT-MDC checkpoints (long-stream only)."""
from __future__ import annotations

import argparse
import csv
import json

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, embed_many, summarize
from src.iclr27_phase4d.long_stream import load_stream_cache
from src.orbit_mdc.evaluate_mdc import load_mdc_model, run_mdc_stream

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policy", choices=["auto", "compat", "birth"],
                    default="auto")
    ap.add_argument("--compat_thrs", default="0.35,0.45,0.55")
    ap.add_argument("--compat_margins", default="0.0,0.02,0.05,0.1")
    ap.add_argument("--birth_thrs", default="0.3,0.4,0.5,0.6")
    ap.add_argument("--quarantine_modes", default="0")
    ap.add_argument("--quarantine_support_thr", type=int, default=3)
    ap.add_argument("--quarantine_dispersion_thr", type=float, default=0.3)
    ap.add_argument("--quarantine_coef", type=float, default=1.0)
    args = ap.parse_args()
    model, ck = load_mdc_model(args.checkpoint, args.device)
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          args.device)
    use_birth = bool(ck.get("use_birth_head", False))
    rows_out = []
    for qmode in [int(x) for x in args.quarantine_modes.split(",")]:
        if args.policy == "birth" or (args.policy == "auto" and use_birth):
            thr_iter = [("birth", t) for t in
                        [float(x) for x in args.birth_thrs.split(",")]]
        else:
            thr_iter = [("compat", thr, margin)
                        for thr in [float(x) for x in args.compat_thrs.split(",")]
                        for margin in [float(x)
                                       for x in args.compat_margins.split(",")]]
        for cfg in thr_iter:
            if len(cfg) == 2:
                _, birth_thr = cfg
                compat_thr, compat_margin = 0.45, 0.05
                policy = "birth"
            else:
                _, compat_thr, compat_margin = cfg
                birth_thr = 0.5
                policy = "compat"
            logs = run_mdc_stream(
                model, ck, rows, feats, labels, args.device, gate_thr=0.5,
                compat_thr=compat_thr, compat_margin=compat_margin,
                birth_thr=birth_thr, policy=policy,
                quarantine_mode=qmode,
                quarantine_support_thr=args.quarantine_support_thr,
                quarantine_dispersion_thr=args.quarantine_dispersion_thr,
                quarantine_coef=args.quarantine_coef,
                syn_mean=syn_mean, zs_rels=(zs, rels))
            attach_gt(logs, gt_rows)
            r = summarize("scan", logs, gt_rows, "overall")
            r_real = summarize("scan", logs, gt_rows, "real_only",
                               select=lambda l: l["role"] == "novel"
                               and int(l["class"]) < 1000000)
            r_syn = summarize("scan", logs, gt_rows, "syn_only",
                              select=lambda l: l["role"] == "novel"
                              and int(l["class"]) >= 1000000)
            row = dict(r)
            row.update({"policy": policy, "compat_threshold": compat_thr,
                        "compat_margin": compat_margin,
                        "birth_threshold": birth_thr,
                        "quarantine_mode": qmode,
                        "quarantine_support_thr": args.quarantine_support_thr,
                        "quarantine_dispersion_thr":
                            args.quarantine_dispersion_thr,
                        "quarantine_coef": args.quarantine_coef,
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
