"""One locked residual correspondence projection pilot.

The backbone and TSE are frozen.  Public TRAIN supported-known categories
provide legal cross-instance positives; the disjoint calibration videos select
the checkpoint.  DEV+ and Q1 are never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase7a.training.train_reliability_head import load_tse


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


class ResidualProjection(nn.Module):
    def __init__(self, rank=32, out_dim=128):
        super().__init__()
        self.v = nn.Linear(768, rank, bias=False)
        self.u = nn.Linear(rank, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.orthogonal_(self.v.weight, gain=0.05)
        nn.init.orthogonal_(self.u.weight, gain=0.05)

    def forward(self, x, h0):
        # x (...,768), h0 (...,128); alpha is exactly zero at initialization.
        return F.normalize(h0 + self.alpha * self.u(self.v(x)), dim=-1)


def supcon(h, labels, vids, temp=0.07):
    sim = h @ h.t() / temp
    n = h.shape[0]; eye = torch.eye(n, device=h.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & (vids[:, None] != vids[None, :]) & ~eye
    valid = pos.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=h.device)
    logits = sim.masked_fill(eye, -1e9)
    log_den = torch.logsumexp(logits, dim=1)
    log_pos = torch.logsumexp(logits.masked_fill(~pos, -1e9), dim=1)
    return -(log_pos[valid] - log_den[valid]).mean()


def retrieval(h, labels, vids):
    h = F.normalize(h, dim=-1).detach().cpu().numpy(); labels = labels.cpu().numpy(); vids = vids.cpu().numpy()
    vals = []
    for i in range(len(h)):
        cand = [j for j in range(len(h)) if j != i and vids[j] != vids[i] and labels[j] == labels[i]]
        if not cand: continue
        allj = [j for j in range(len(h)) if j != i and vids[j] != vids[i]]
        order = sorted(allj, key=lambda j: (-float(h[i] @ h[j]), j)); vals.append(float(order[0] in set(cand)))
    return float(np.mean(vals)) if vals else 0.0


def encode_tracks(model, x, h0, device, indices, batch=128):
    out = []
    with torch.no_grad():
        for s in range(0, len(indices), batch):
            ii = indices[s:s + batch]
            out.append(model(x[ii].to(device), h0[ii].to(device)).mean(1).cpu())
    return F.normalize(torch.cat(out), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--seed", type=int, default=20260824); ap.add_argument("--rank", type=int, default=32); ap.add_argument("--out-dir", default="outputs/iclr27_phase14c/checkpoints"); ap.add_argument("--summary", default="outputs/iclr27_phase14c/eval/projection_training.json")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    z = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz")
    split = json.loads((ROOT / "outputs/iclr27_phase14b/manifests/devplus_split.json").read_text())
    known = set(int(x) for x in json.loads((ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json").read_text()))
    tr_vids, cal_vids = set(split["representation_train_videos"]), set(split["calibration_videos"])
    labels = z["labels"].astype(np.int64); vids = z["video_ids"].astype(np.int64)
    mask = np.isin(labels, list(known))
    train_idx = np.where(mask & np.isin(vids, list(tr_vids)))[0]
    cal_idx = np.where(mask & np.isin(vids, list(cal_vids)))[0]
    assert len(train_idx) >= 100 and len(cal_idx) >= 20
    x = torch.from_numpy(z["frame_feats"].astype(np.float32))
    tse, _, _ = load_tse(device)
    with torch.no_grad():
        h0_parts = []
        for s in range(0, len(x), 128):
            h0_parts.append(tse.project(x[s:s + 128].to(device)).cpu())
        h0 = torch.cat(h0_parts)
    y = torch.from_numpy(labels); v = torch.from_numpy(vids)
    outputs = {}; train_metrics = {}
    for mode in ("main", "no_cross_instance"):
        model = ResidualProjection(rank=args.rank).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        best = -1.0; best_step = 0; patience = 0; eval_log = []
        for step in range(1, args.steps + 1):
            ii = np.random.choice(train_idx, size=min(128, len(train_idx)), replace=False)
            xb, hb = x[ii].to(device), h0[ii].to(device)
            hb_proj = model(xb, hb)
            labels_b, vids_b = y[ii].to(device), v[ii].to(device)
            loss_temporal = (1.0 - (hb_proj[:, :4].mean(1) * hb_proj[:, 4:].mean(1)).sum(-1)).mean()
            loss_distill = (hb_proj.mean(1) - hb.mean(1)).pow(2).mean()
            loss = 0.5 * loss_temporal + 0.1 * loss_distill
            if mode == "main":
                loss = loss + supcon(F.normalize(hb_proj.mean(1), dim=-1), labels_b, vids_b)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 100 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    cal_h = encode_tracks(model, x, h0, device, cal_idx)
                r1 = retrieval(cal_h, y[cal_idx], v[cal_idx])
                eval_log.append({"step": step, "calibration_cross_video_r1": r1, "loss": float(loss.detach()), "alpha": float(model.alpha.detach())})
                if r1 > best:
                    best, best_step, patience = r1, step, 0
                    out = ROOT / args.out_dir / ("correspondence_projection.pth" if mode == "main" else "correspondence_projection_no_cross_instance.pth")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "rank": args.rank, "seed": args.seed, "mode": mode, "best_calibration_r1": best, "step": step, "alpha": float(model.alpha.detach())}, out)
                else:
                    patience += 1
                model.train()
                # Fixed early-stop rule: five consecutive 100-step calibration checks.
                if patience >= 5:
                    break
        train_metrics[mode] = {"best_calibration_cross_video_r1": best, "best_step": best_step, "checks": eval_log, "steps_run": eval_log[-1]["step"] if eval_log else 0, "train_tracks": int(len(train_idx)), "calibration_tracks": int(len(cal_idx)), "labels_public_supported_known_only": True, "devplus_labels_used": False, "q1_label_used": False}
    summary = {"protocol": "phase14c_projection_pilot", "rank": args.rank, "seed": args.seed, "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)", "max_steps": args.steps, "loss": "supcon + 0.5 temporal prefix consistency + 0.1 geometry distillation (main); temporal + geometry only (control)", "train_metrics": train_metrics}
    out = ROOT / args.summary; out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(summary, indent=2)); os.replace(tmp, out); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
