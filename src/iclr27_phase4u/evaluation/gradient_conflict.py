"""Diagnostic: gradient conflict between semantic objectives on the Phase 4T
semantic encoder (adapter + GRU + LN), measured on one real episode.

Question: does routing/memory training reshape the semantic representation in
a direction opposed to the cross-track contrastive objective?
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.episodes import load_episodic_universe
from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4s.train import build_known_matrix
from src.iclr27_phase4t.episodes import (
    RealEpisodeConfig,
    RealStreamStore,
    make_real_episode,
    real_episode_batch,
)
from src.iclr27_phase4t.model import HierarchicalCore
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.downstream.runtime import run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/iclr27_phase4t/t3/checkpoint.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    by_train, by_dev, syn_features = load_episodic_universe()
    known_mat = build_known_matrix(syn_features, {**by_train, **by_dev})
    model = HierarchicalCore(768, 256, known_prototypes=known_mat,
                             use_defer=False, use_qphys=False).to(args.device)
    ck = torch.load(ROOT / args.checkpoint, map_location=args.device)
    ck["model"].pop("known_raw", None)
    model.load_state_dict(ck["model"], strict=False)
    model.train()

    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    store = RealStreamStore(rows, feats)
    cfg = RealEpisodeConfig()
    rng = random.Random(args.seed)
    ep = make_real_episode(store, cfg, rng)
    batch = real_episode_batch(store, ep, cfg)
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(args.device)
    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}
    memory = NovelMemory(args.device)
    teacher = {"cat_to_teacher": {}, "n_teacher": 0, "teacher_to_mem": {}}
    res = run_episode(model, batch, cfg, known_cat_index, known_list,
                      memory, teacher, "train")

    terms = {}
    if res.get("l1") is not None:
        terms["l1_route"] = -res["l1"].mean()
    if res.get("known") is not None:
        terms["known"] = res["known"].mean()
    if res.get("l2") is not None:
        terms["l2_route"] = -res["l2"].mean()
    mem = [res[k] for k in ("mem_pull", "mem_push") if res.get(k) is not None]
    if mem:
        terms["memory"] = torch.cat([t.reshape(-1) for t in mem]).mean()
    if res.get("contrast") is not None:
        terms["contrast"] = res["contrast"].mean()

    enc_params = list(model.adapter.parameters()) + list(model.gru.parameters()) \
        + list(model.ln.parameters())
    grads = {}
    for name, loss in terms.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g = [p.grad.detach().flatten() for p in enc_params if p.grad is not None]
        grads[name] = torch.cat(g) if g else torch.zeros(1, device=args.device)

    def cos(a, b):
        na, nb = a.norm(), b.norm()
        if na < 1e-12 or nb < 1e-12:
            return None
        return float((a @ b) / (na * nb))

    report = {"terms": {k: float(v) for k, v in terms.items()},
              "grad_norm": {k: float(grads[k].norm()) for k in grads}}
    names = list(grads)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report[f"cos_{a}_vs_{b}"] = cos(grads[a], grads[b])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
