"""Diagnose cold-head action distribution on Q1 dev (K=0 prefix)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import Q1_DEV, group_tracks, load_proposals
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    load_known_branch,
    load_novel_branch,
    proto_evidence,
)
from src.iclr27_phase4w.cold_start.train import ColdStartHead
from src.iclr27_phase4w.episodes.build_episodes import load_active_universe
from src.iclr27_phase4w.evaluation.dev_eval import qphys_from_rows


def main():
    device = "cuda:0"
    ktsr, kcls = load_known_branch(device)
    ntsr, l2 = load_novel_branch(device)
    protos, _, proj_t = load_active_universe(device)
    ck = torch.load(ROOT / "outputs/iclr27_phase4w/cold_start/head_cold_v3/head.pth",
                    map_location=device)
    cold = ColdStartHead(ck["dim"]).to(device)
    cold.load_state_dict(ck["model"]); cold.eval()
    active_idx = list(range(48))

    rows = load_proposals(Path(Q1_DEV))
    arr = np.load(ROOT / "outputs/iclr27_phase4s/q1_features/feats.npz")["feats"]
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    wc = r_phys_calibration(rows)
    rs = compute_r_phys(rows, wc)
    memory = NovelMemory(device)
    acts = defaultdict(lambda: defaultdict(int))
    probs = defaultdict(list)
    pe_rows = {"known": [], "novel": [], "fp": []}
    ts_rows = {"known": [], "novel": [], "fp": []}
    with torch.no_grad():
        for key in sorted(tracks):
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
            for t, r in enumerate(tracks[key][:4]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                ft = torch.from_numpy(f).unsqueeze(0).to(device)
                qt = torch.tensor([qmap[id(r)]], device=device)
                rsv = float(rs[id(r)])
                ev, s_k, s_n, nl, l2_new = ds.step(ft, qt, rsv, t + 1, memory)
                pe = proto_evidence(s_k, protos, active_idx, tau=0.1)
                skp = (torch.nn.functional.normalize(s_k, dim=-1) @ proj_t)[0]
                skp = skp.detach().cpu().numpy()
                qv = qt[0].cpu().numpy().astype(np.float32)
                x = np.concatenate([pe, skp, qv]).astype(np.float32)
                logits = cold(torch.from_numpy(x).unsqueeze(0).to(device))[0]
                p = torch.softmax(logits, -1)
                a = int(p.argmax())
                acts[r["gt_role"]][f"t{t}_a{a}"] += 1
                probs[r["gt_role"]].append(p.cpu().numpy())
                if r["gt_role"] in pe_rows:
                    pe_rows[r["gt_role"]].append(pe)
                    ts_rows[r["gt_role"]].append(t)
    summary = {k: dict(v) for k, v in acts.items()}
    for role, ps in probs.items():
        summary[role]["mean_prob"] = np.mean(np.stack(ps), axis=0).round(4).tolist()
    from sklearn.metrics import roc_auc_score
    auc = {}
    for step in range(4):
        aurocs = {}
        for fi, fname in [(0, "top1"), (1, "margin"), (2, "entropy"),
                          (3, "energy")]:
            yk = np.asarray([t == step for t in ts_rows["known"]])
            yn = np.asarray([t == step for t in ts_rows["novel"]])
            if yk.sum() < 5 or yn.sum() < 5:
                continue
            vk = np.stack(pe_rows["known"])[yk, fi]
            vn = np.stack(pe_rows["novel"])[yn, fi]
            # novel should have lower top1/margin/energy, higher entropy
            sign = -1.0 if fi in (0, 1, 3) else 1.0
            aurocs[fname] = round(float(roc_auc_score(
                np.concatenate([np.zeros(int(yk.sum())), np.ones(int(yn.sum()))]),
                np.concatenate([sign * vk, sign * vn]))), 4)
        auc[str(step)] = aurocs
    summary["auroc_by_step"] = auc
    out = ROOT / "outputs/iclr27_phase4w/active_universe_audit/dev_cold_head_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
