"""Evaluate a pretrained TSR representation with the Phase 4U geometry and
cross-track retrieval benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4u.bench import metrics_for_embeddings
from src.iclr27_phase4u.data import ROOT, class_sets, load_source
from src.iclr27_phase4u.trajectory.model import TSR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="real,episodic,dev")
    ap.add_argument("--class-set", default="all",
                    choices=["all", "meta_train", "meta_dev"])
    ap.add_argument("--prefix-lengths", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location=args.device)
    model = TSR(arch=ck.get("arch", "gru")).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    meta_tr, meta_de = class_sets()
    allowed = None
    if args.class_set == "meta_train":
        allowed = meta_tr
    elif args.class_set == "meta_dev":
        allowed = meta_de
    prefix_lengths = [int(x) for x in args.prefix_lengths.split(",")]
    out_json = {"checkpoint": args.checkpoint, "class_set": args.class_set}
    for name in args.sources.split(","):
        src = load_source(name)
        if allowed is not None:
            src["instances"] = [x for x in src["instances"] if x["cat"] in allowed]
            src["by_cat"] = {}
            for x in src["instances"]:
                src["by_cat"].setdefault(x["cat"], []).append(x["id"])
        results = []
        inst_by_id = {x["id"]: x for x in src["instances"]}
        complete = {}
        for x in src["instances"]:
            f = torch.from_numpy(x["feats"]).to(args.device)
            q = None if x["q"] is None else torch.from_numpy(x["q"]).to(args.device)
            with torch.no_grad():
                st = model.embed_sequence(f, q)
            complete[x["id"]] = st[-1].cpu().numpy().astype(np.float32)
        for p in prefix_lengths:
            embs = {}
            for x in src["instances"]:
                if p > x["feats"].shape[0]:
                    continue
                f = torch.from_numpy(x["feats"][:p]).to(args.device)
                q = None if x["q"] is None else torch.from_numpy(x["q"][:p]).to(args.device)
                with torch.no_grad():
                    st = model.embed_sequence(f, q)
                embs[x["id"]] = st[-1].cpu().numpy().astype(np.float32)
            results.append(metrics_for_embeddings(
                embs, complete, src, args.seed, f"p{p}"))
        results.append(metrics_for_embeddings(
            complete, complete, src, args.seed, "complete"))
        out_json[name] = results
        print(name, "done", len(src["instances"]), flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_json, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
