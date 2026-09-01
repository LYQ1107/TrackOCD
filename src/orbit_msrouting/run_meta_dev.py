"""Train-side meta-dev evaluation for ORBIT-MSRouting candidates."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import attach_gt, summarize
from src.iclr27_phase4d.long_stream import active_bucket, load_stream_cache
from src.orbit_msrouting.evaluate_msrouting import (
    evaluate_long_msrouting,
    load_msrouting_checkpoint,
)

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


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
    return {
        "candidate": candidate, "scope": scope,
        "final_prototype_count": len(by_vid),
        "hub_count": hubs,
        "known_origin_prototype_count": known_origin,
        "mean_memory_size": (sum(l["active_novel_prototypes"] for l in sel)
                             / len(sel) if sel else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    args = ap.parse_args()
    out_dir = ROOT + "/outputs/orbit_msrouting/meta_dev"
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_rows = []
    mem_rows = []
    _, gt_rows, _, _ = load_stream_cache()
    for cp in args.checkpoints:
        cand = cp.split("/")[-2] if "/" in cp else cp
        model, ck = load_msrouting_checkpoint(ROOT + "/" + cp, args.device)
        out, logs, r_real, r_syn = evaluate_long_msrouting(
            model, ck, args.device, gate_thr=0.5,
            compat_thr=args.compat_threshold,
            compat_margin=args.compat_margin)
        for r in out:
            model_rows.append({"candidate": cand, **r})
        for scope in ["overall", "0-32", "33-128", "129-256", "257+"]:
            mr = memory_state_rows(logs, cand, scope)
            if mr:
                mem_rows.append(mr)
        print(f"[{cand}] overall rn="
              f"{[r['rn_acc'] for r in out if r['scope']=='overall']} "
              f"ari={[r['ari'] for r in out if r['scope']=='overall']} "
              f"real_rn={r_real['rn_acc'] if r_real else None} "
              f"syn_rn={r_syn['rn_acc'] if r_syn else None}", flush=True)
    if model_rows:
        with open(f"{out_dir}/model_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
            w.writeheader()
            w.writerows(model_rows)
    if mem_rows:
        with open(f"{out_dir}/memory_state_comparison.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mem_rows[0].keys()))
            w.writeheader()
            w.writerows(mem_rows)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
