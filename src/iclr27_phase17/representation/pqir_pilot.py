"""Conditional Phase 17 PQIR P1/P2 pilot on frozen paired features.

This is an adapter-only pilot.  The DINO backbones, tracker, memory, and
causal decision remain frozen.  GT views are a public teacher for the paired
loss and are never consumed by the proposal-time embedding.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.iclr27_phase17.evaluation.paired_crop_metrics import _retrieval

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


class PQIR(nn.Module):
    def __init__(self, dim: int = 768, out_dim: int = 256, p2: bool = False):
        super().__init__(); self.p2 = p2
        self.net = nn.Sequential(nn.Linear(dim * 4 + 5, 512), nn.GELU(), nn.Linear(512, out_dim))

    def forward(self, views: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        # raw/context/temporal/multiscale proposal views only; GT is never in
        # this call.  Quality features are causal score/area/aspect/border and
        # prefix age, all available at the current observation.
        x = torch.cat([views[:, 2], views[:, 3], views[:, 4], views[:, 9], quality], dim=-1)
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def _norm(x):
    x = np.asarray(x, dtype=np.float32); return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _quality(rows: list[dict[str, Any]]) -> np.ndarray:
    out = []
    for r in rows:
        b = json.loads(r["bbox_xyxy"]); w, h = max(1., b[2] - b[0]), max(1., b[3] - b[1])
        # Every value is proposal/cause observable; no IoU or GT field.
        out.append([float(r["score"]), float(r.get("area_fraction", 0.0)), float(w / h),
                    float(min(b[0], b[1], 640 - b[2], 480 - b[3]) / 640.), float(r.get("proposal_track_length", 1)) / 100.])
    q = np.asarray(out, dtype=np.float32); q[:, 1] = np.clip(q[:, 1], 0, 1); q[:, 2] = np.clip(q[:, 2] / 4, 0, 1); q[:, 3] = np.clip(q[:, 3], 0, 1); q[:, 4] = np.clip(q[:, 4], 0, 1)
    return q


def _contrast(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sim = z @ z.T / .1; eye = torch.eye(len(z), device=z.device, dtype=torch.bool); pos = labels[:, None] == labels[None, :]
    pos &= ~eye; exp = torch.exp(sim.masked_fill(eye, -1e9)); denom = exp.sum(1).clamp_min(1e-6); num = (exp * pos.float()).sum(1)
    valid = pos.any(1); return (-torch.log((num[valid] / denom[valid]).clamp_min(1e-6))).mean() if valid.any() else z.sum() * 0


def _eval(feat: np.ndarray, cats: np.ndarray, vids: np.ndarray, row_iou: np.ndarray) -> dict[str, Any]:
    out = {"all": _retrieval(feat, cats, vids)}
    for key, m in {"low_quality": row_iou < .5, "high_quality": row_iou >= .5}.items():
        out[key] = {"rows": int(m.sum()), **(_retrieval(feat[m], cats[m], vids[m]) if m.sum() >= 2 else {})}
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(1717); np.random.seed(1717)
    z = np.load(args.features, allow_pickle=False); keys = [str(x) for x in z["row_keys"]]; d3 = z["dinov3"].astype(np.float32); cats = z["category_id"].astype(int); vids = z["video_id"].astype(int); roles = z["role17"].astype(str); row_iou = z["row_iou"].astype(float)
    by_key = {r["row_key"]: r for r in csv.DictReader(args.rows.open())}; rows = [by_key[k] for k in keys]
    q = _quality(rows)
    train = ~np.isin(roles, ["known_audit", "novel_audit"]); test = ~train
    if train.sum() < 8 or test.sum() < 4: raise RuntimeError("PQIR split too small")
    dev = torch.device(args.device); X = torch.from_numpy(d3).to(dev); Q = torch.from_numpy(q).to(dev); C = torch.from_numpy(cats).to(dev); T = torch.from_numpy(d3[:, 0]).to(dev)
    results = {}; ckpts = []
    for variant, p2 in [("P1_paired_proposal_consistency", False), ("P2_causal_quality_temporal", True)]:
        model = PQIR(p2=p2).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); idx = np.where(train)[0]; rng = np.random.default_rng(1717)
        model.train(); losses = []
        steps = min(args.steps, 2000)
        for step in range(steps):
            take = rng.choice(idx, size=min(args.batch, len(idx)), replace=len(idx) < args.batch)
            ti = torch.as_tensor(take, device=dev); pred = model(X[ti], Q[ti])
            # Public GT teacher: feed the clean GT views through the same
            # adapter, but detach the target.  This avoids an arbitrary
            # dimensional projection while keeping GT out of proposal-time
            # inference.
            teacher_views = X[ti].clone(); teacher_views[:, 2] = X[ti, 0]; teacher_views[:, 3] = X[ti, 1]; teacher_views[:, 4] = X[ti, 0]; teacher_views[:, 9] = X[ti, 0]
            with torch.no_grad(): teacher = model(teacher_views, Q[ti])
            loss = (1 - (pred * teacher.detach()).sum(1)).mean() + .20 * _contrast(pred, C[ti])
            # Same physical track positives use only currently visible rows;
            # no identity feature is passed to the adapter.
            track = torch.tensor([int(rows[i]["video_id"]) * 1000000 + int(rows[i]["track_id"]) for i in take], device=dev)
            loss = loss + .10 * _contrast(pred, track)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): out = model(X, Q).cpu().numpy()
        # Teacher clean retrieval is a diagnostic upper reference; only out is
        # a deployed proposal-time representation.
        clean = _eval(_norm(d3[:, 0]), cats, vids, row_iou); adapted = _eval(out, cats, vids, row_iou)
        # Low-quality gain and clean-view drop are preregistered pilot gates.
        low_gain = adapted["low_quality"].get("r_at_1", 0) - _eval(_norm(d3[:, 2]), cats, vids, row_iou)["low_quality"].get("r_at_1", 0)
        clean_drop = clean["all"].get("r_at_1", 0) - adapted["all"].get("r_at_1", 0)
        cp = args.checkpoint.parent / (variant + ".pt"); cp.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": model.state_dict(), "variant": variant, "input": "DINOv3 proposal views + causal quality", "steps": steps}, cp); ckpts.append(str(cp))
        results[variant] = {"steps": steps, "mean_loss": float(np.mean(losses)), "train_rows": int(train.sum()), "test_rows": int(test.sum()), "clean_teacher": clean, "adapted": adapted, "low_quality_r1_gain_vs_raw": float(low_gain), "clean_view_r1_drop_vs_teacher": float(clean_drop), "full_pilot_gate": bool(low_gain >= .05 and clean_drop <= .02), "gt_iou_used_in_fit": False, "future_frames_used": False, "physical_id_as_feature": False}
    result = {"protocol": "trackocd_iclr27_phase17_pqir_pilot", "backbone": "DINOv3_dense_frozen", "variants": results, "checkpoints": ckpts, "steps_max": min(args.steps, 2000), "one_gpu": True, "public_gt_teacher_only": True, "devplus_labels_for_fit": False, "q1_labels_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = args.out.with_suffix(args.out.suffix + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, args.out); print(json.dumps(result, indent=2, sort_keys=True)); return result


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--features", type=Path, default=ROOT / "outputs/iclr27_phase17/paired/public_paired_features.npz"); ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/public_role_rows.csv"); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/iclr27_phase17/features/pqir_pilot.pt"); ap.add_argument("--out", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/pqir_pilot.json"); args = ap.parse_args(); run(args)


if __name__ == "__main__": main()
