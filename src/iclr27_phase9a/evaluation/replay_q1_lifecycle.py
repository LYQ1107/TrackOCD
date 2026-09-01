"""Apply the Phase 9A lifecycle to an official Q1 proposal stream."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase9a.lifecycle import CausalLifecycle, LifecycleHeads

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--model-dir", default="outputs/iclr27_phase9a/training/lifecycle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-states", type=int, default=512)
    ap.add_argument("--no-false-birth", action="store_true")
    ap.add_argument("--no-lifecycle", action="store_true")
    ap.add_argument("--fixed-maturity", type=int, default=0)
    ap.add_argument("--no-trajectory", action="store_true")
    args = ap.parse_args()

    with open(ROOT / args.proposals, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    feats = np.asarray(np.load(ROOT / args.feats)["feats"], dtype=np.float32)
    if len(rows) != len(feats):
        raise RuntimeError(f"proposal rows {len(rows)} != feats {len(feats)}")
    dev = torch.device(args.device)
    tse, _, _ = load_tse(dev)
    h = project(dev, tse, feats)
    heads = LifecycleHeads.load(ROOT / args.model_dir / "heads.npz")
    p = np.load(ROOT / args.model_dir / "known_prototypes.npz")
    dp = ROOT / args.model_dir / "decision_prototypes.npz"
    if dp.exists():
        q = np.load(dp)
        decision_protos, decision_ids = q["prototypes"], q["known_ids"].tolist()
    else:
        decision_protos, decision_ids = None, None
    mem = CausalLifecycle(
        p["prototypes"], p["known_ids"].tolist(), heads,
        decision_prototypes=decision_protos, decision_ids=decision_ids,
        max_states=args.max_states, no_false_birth=args.no_false_birth,
        no_lifecycle=args.no_lifecycle,
        fixed_maturity=(args.fixed_maturity if args.fixed_maturity > 0 else None),
        trajectory=not args.no_trajectory)

    chrono = sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))
    out_rows = [None] * len(rows)
    for i in chrono:
        r = rows[i]
        key = (int(r["video_id"]), int(r["track_id"]))
        o = mem.step(h[i], key, float(r.get("score") or 0.0),
                     float(r.get("prior_hits") or 0.0))
        rr = dict(r)
        rr["sem_action"] = o["action"]
        rr["sem_sid"] = str(int(o["semantic_id"]))
        rr["sem_kscore"] = f"{float(o['known_score']):.6f}"
        rr["sem_maturity"] = f"{float(o['maturity_score']):.6f}"
        rr["sem_reusable"] = str(int(bool(o["reusable"])))
        out_rows[i] = rr
    if out_rows:
        # DSCT proposal files may already carry placeholder semantic columns;
        # replace them rather than writing duplicate CSV headers (DictReader
        # would otherwise silently read the stale first occurrence).
        semantic_fields = ["sem_action", "sem_sid", "sem_kscore",
                           "sem_maturity", "sem_reusable"]
        fields = [f for f in fields if f not in semantic_fields] + semantic_fields
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp, out)
    summary = {
        "n_rows": len(rows),
        "action_counts": dict(Counter(x["sem_action"] for x in out_rows)),
        "n_states": len(mem.states),
        "n_quarantined": len(mem.quarantine),
        "video_ids": sorted({int(r["video_id"]) for r in rows}),
        "output": str(args.out),
    }
    sp = out.with_suffix(".lifecycle.json")
    sp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
