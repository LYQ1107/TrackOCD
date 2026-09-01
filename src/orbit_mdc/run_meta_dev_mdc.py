"""Train-side meta-dev evaluation for ORBIT-MDC candidates (long-stream)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import torch

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit_mdc.evaluate_mdc import (
    evaluate_long_mdc,
    load_mdc_model,
    run_mdc_stream,
)
from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, summarize
from src.iclr27_phase4d.long_stream import active_bucket, load_stream_cache


def memory_state_rows(logs, candidate, scope="overall"):
    sel = logs if scope == "overall" else [
        l for l in logs if active_bucket(l["active_novel_prototypes"]) == scope]
    if not sel:
        return None
    by_vid = defaultdict(list)
    for l in sel:
        if l["predicted_action"] in ("EXISTING_NOVEL", "NEW_NOVEL"):
            by_vid[l["predicted_virtual_novel_id"]].append(l)
    hubs = sum(1 for ls in by_vid.values()
               if len({l["class"] for l in ls}) >= 2)
    known_origin = sum(1 for ls in by_vid.values()
                       if ls[0]["role"] == "known")
    supports = [l["prototype_support"] for l in sel
                if l["predicted_action"] == "EXISTING_NOVEL"]
    return {
        "candidate": candidate, "scope": scope,
        "final_prototype_count": len(by_vid),
        "hub_count": hubs,
        "known_origin_prototype_count": known_origin,
        "mean_prototype_support": (sum(supports) / len(supports)
                                   if supports else 0.0),
        "mean_memory_size": (sum(l["active_novel_prototypes"]
                                 for l in sel) / len(sel) if sel else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--birth_threshold", type=float, default=0.5)
    ap.add_argument("--policy", choices=["auto", "compat", "birth"],
                    default="auto")
    ap.add_argument("--quarantine_mode", type=int, default=0)
    ap.add_argument("--quarantine_support_thr", type=int, default=3)
    ap.add_argument("--quarantine_dispersion_thr", type=float, default=0.3)
    ap.add_argument("--quarantine_coef", type=float, default=1.0)
    args = ap.parse_args()
    out_dir = ROOT + "/outputs/orbit_mdc/meta_dev"

    def save(name, rows):
        if not rows:
            return
        with open(f"{out_dir}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    model_rows = []
    err_rows = []
    mem_rows = []
    for cp in args.checkpoints:
        cand = cp.split("/")[-2] if "/" in cp else cp
        model, ck = load_mdc_model(ROOT + "/" + cp, args.device)
        rows_out, logs, r_real, r_syn = evaluate_long_mdc(
            model, ck, args.device, gate_thr=0.5,
            compat_thr=args.compat_threshold,
            compat_margin=args.compat_margin,
            birth_thr=args.birth_threshold, policy=args.policy,
            quarantine_mode=args.quarantine_mode,
            quarantine_support_thr=args.quarantine_support_thr,
            quarantine_dispersion_thr=args.quarantine_dispersion_thr,
            quarantine_coef=args.quarantine_coef)
        for r in rows_out:
            row = {"candidate": cand}
            row.update(r)
            model_rows.append(row)
        for l in logs:
            l["candidate"] = cand
        err_keys = ["known_acc", "rn_acc", "cond_novel_acc", "routing_recall",
                    "nmi", "ari", "count_error", "known_to_novel",
                    "novel_to_known", "repeated_false_birth",
                    "wrong_existing", "first_merge"]
        for scope, ls in [("overall", logs)]:
            r = summarize(cand, ls, _gt(), scope)
            if r:
                err_rows.append({"candidate": cand, "scope": scope,
                                 **{k: r.get(k) for k in err_keys}})
        for scope in ["overall", "0-32", "33-128", "129-256", "257+"]:
            mr = memory_state_rows(logs, cand, scope)
            if mr:
                mem_rows.append(mr)
        print(f"[{cand}] overall rn={r['rn_acc'] if r else None} "
              f"ari={r['ari'] if r else None} "
              f"we={r['wrong_existing'] if r else None}", flush=True)
    save("model_comparison.csv", model_rows)
    save("error_comparison.csv", err_rows)
    save("memory_state_comparison.csv", mem_rows)
    print("saved", out_dir)


def _gt():
    _, gt_rows, _, _ = load_stream_cache()
    return gt_rows


if __name__ == "__main__":
    main()
