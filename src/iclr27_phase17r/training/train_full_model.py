"""Substantive full-population training for Phase17R T0 or M1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.iclr27_phase17r.training.model import ObservabilitySemanticModel, parameter_counts

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
GEOMETRY_FIELDS = [
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_box_stability_iou"
]
TRAIN_ROLES = {"known_bank", "novel_correspondence_train"}
CAL_ROLES = {"known_calibration", "novel_calibration"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_data(rows_path: Path, features_path: Path, variant: str) -> dict[str, Any]:
    rows = list(csv.DictReader(rows_path.open())); n = len(rows)
    z = np.load(features_path, allow_pickle=False)
    lookup = {str(k): i for i, k in enumerate(z["row_keys"].astype(str))}
    order = np.asarray([lookup[r["row_key"]] for r in rows], dtype=np.int64)
    if variant == "t0":
        roi = z["roi"][order].astype(np.float32); cls = z["cls"][order].astype(np.float32)
        mean = roi + cls; mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-8)
        views = np.stack([roi, cls, mean, roi], axis=1).astype(np.float32)
        gt_views = np.zeros((n, 2, 768), dtype=np.float32); teacher_mask = np.zeros(n, dtype=bool)
    else:
        all_feat = z["features"][order].astype(np.float32)
        views = all_feat[:, :4]
        gt_views = all_feat[:, 4:6]
        teacher_mask = z["teacher_mask"][order].astype(bool)
    raw = views[:, 0]
    prev_idx = np.arange(n, dtype=np.int64)
    by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows): by_track[(int(r["video_id"]), int(r["track_id"]))].append(i)
    for idx in by_track.values():
        idx.sort(key=lambda i: (int(rows[i]["frame_id"]), int(rows[i]["proposal_local_id"]), rows[i]["row_key"]))
        for pos, i in enumerate(idx): prev_idx[i] = idx[pos - 1] if pos else i
    previous_raw = raw[prev_idx]
    geometry = np.asarray([[float(r[k]) for k in GEOMETRY_FIELDS] for r in rows], dtype=np.float32)
    train_idx = np.asarray([i for i, r in enumerate(rows) if r["role17"] in TRAIN_ROLES], dtype=np.int64)
    cal_idx = np.asarray([i for i, r in enumerate(rows) if r["role17"] in CAL_ROLES], dtype=np.int64)
    gmean, gstd = geometry[train_idx].mean(0), geometry[train_idx].std(0)
    gstd = np.maximum(gstd, 1e-4); geometry = (geometry - gmean) / gstd
    roles = np.asarray([0 if r["gt_role_common"] == "fp" else 1 if r["gt_role_common"] == "supported_known" else 2 for r in rows], dtype=np.int64)
    cats = np.asarray([int(r["gt_category_id_common"]) for r in rows], dtype=np.int64)
    observable = np.asarray([int(r["assigned"]) == 1 and float(r["row_iou"]) >= .5 for r in rows], dtype=bool)
    assigned = np.asarray([int(r["assigned"]) == 1 for r in rows], dtype=bool)
    return {"rows": rows, "views": views, "gt_views": gt_views, "teacher_mask": teacher_mask,
            "previous_raw": previous_raw, "geometry": geometry, "gmean": gmean, "gstd": gstd,
            "roles": roles, "cats": cats, "observable": observable, "assigned": assigned,
            "train_idx": train_idx, "cal_idx": cal_idx}


class PairSampler:
    def __init__(self, data: dict[str, Any], seed: int, rank: int):
        self.data = data; self.rng = np.random.default_rng(seed + rank * 1009)
        rows, cats, obs = data["rows"], data["cats"], data["observable"]
        by_cat_track: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
        for i in data["train_idx"]:
            r = rows[int(i)]
            # Every legal training-role novel category is disjoint from the
            # calibration/audit novel categories. Using both training roles
            # recovers the available cross-video physical-track positives.
            if r["gt_role_common"] != "novel" or not obs[int(i)]:
                continue
            by_cat_track[int(cats[int(i)])][(int(r["video_id"]), int(r["track_id"]))].append(int(i))
        self.by_cat_track = {c: d for c, d in by_cat_track.items() if len(d) >= 2}
        self.pos_categories = sorted(self.by_cat_track)
        if not self.pos_categories:
            raise RuntimeError("no legal cross-video meta-novel positive pairs")
        base = data["views"][:, 0]
        centroids = {}
        for c, tracks in self.by_cat_track.items():
            idx = [i for vals in tracks.values() for i in vals]
            x = base[idx].mean(0); centroids[c] = x / max(np.linalg.norm(x), 1e-8)
        self.hard = {}
        for c in self.pos_categories:
            others = sorted((float(centroids[c] @ centroids[d]), d) for d in self.pos_categories if d != c)
            self.hard[c] = [d for _, d in others[-5:]]

    def sample(self, n_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left, right, label, count = [], [], [], []
        for j in range(n_pairs):
            c = int(self.rng.choice(self.pos_categories)); tracks = self.by_cat_track[c]
            keys = list(tracks); cross = [(a, b) for a in keys for b in keys if a < b and a[0] != b[0]]
            candidates = cross if cross else [(a, b) for pos, a in enumerate(keys) for b in keys[pos + 1:]]
            ka, kb = candidates[int(self.rng.integers(len(candidates)))]
            left.append(int(self.rng.choice(tracks[ka]))); right.append(int(self.rng.choice(tracks[kb]))); label.append(1.0)
            neg_c = int(self.rng.choice(self.hard[c])) if self.hard[c] else int(self.rng.choice([x for x in self.pos_categories if x != c]))
            neg_tracks = self.by_cat_track[neg_c]; nk = list(neg_tracks)[int(self.rng.integers(len(neg_tracks)))]
            left.append(int(self.rng.choice(tracks[ka]))); right.append(int(self.rng.choice(neg_tracks[nk]))); label.append(0.0)
            count.extend([float(np.log1p(self.rng.integers(1, 65)) / np.log(65.0))] * 2)
        return np.asarray(left), np.asarray(right), np.asarray(label, np.float32), np.asarray(count, np.float32)


class BatchSchedule:
    def __init__(self, data: dict[str, Any], local_batch: int, world: int, rank: int, seed: int):
        self.data = data; self.local_batch = local_batch; self.unique_n = local_batch // 2
        self.world = world; self.rank = rank; self.rng = np.random.default_rng(seed)
        self.local_rng = np.random.default_rng(seed + rank * 97); self.train = data["train_idx"]
        roles, obs = data["roles"], data["observable"]
        self.strata = [
            self.train[(roles[self.train] == 1) & obs[self.train]], self.train[(roles[self.train] == 1) & ~obs[self.train]],
            self.train[(roles[self.train] == 2) & obs[self.train]], self.train[(roles[self.train] == 2) & ~obs[self.train]],
            self.train[(roles[self.train] == 0) & (np.asarray([int(data["rows"][int(i)]["causal_prefix_age"]) for i in self.train]) > 0)],
            self.train[roles[self.train] == 0]
        ]
        self.strata = [x for x in self.strata if len(x)]
        self.order = self.rng.permutation(self.train); self.cursor = 0; self.complete_passes = 0

    def next(self) -> np.ndarray:
        need = self.unique_n * self.world
        if self.cursor + need > len(self.order):
            self.order = self.rng.permutation(self.train); self.cursor = 0; self.complete_passes += 1
        block = self.order[self.cursor:self.cursor + need]; self.cursor += need
        unique = block[self.rank * self.unique_n:(self.rank + 1) * self.unique_n]
        balanced = []
        for j in range(self.local_batch - self.unique_n):
            pool = self.strata[j % len(self.strata)]
            balanced.append(int(pool[int(self.local_rng.integers(len(pool)))]))
        return np.concatenate([unique, np.asarray(balanced, dtype=np.int64)])


def tensor(data: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data[idx])).to(device, non_blocking=True)


def embeddings(model: nn.Module, data: dict[str, Any], indices: np.ndarray, device: torch.device, batch: int = 512) -> dict[str, np.ndarray]:
    model.eval(); out = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            idx = indices[start:start + batch]
            pred = model(tensor(data["views"], idx, device), tensor(data["previous_raw"], idx, device), tensor(data["geometry"], idx, device))
            for k in ("semantic", "observability_logit", "known_logit", "class_logits"):
                out[k].append(pred[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def calibration_diagnostic(model: nn.Module, data: dict[str, Any], known_ids: list[int], device: torch.device) -> dict[str, float]:
    idx = data["cal_idx"]; pred = embeddings(model, data, idx, device)
    roles, cats, obs = data["roles"][idx], data["cats"][idx], data["observable"][idx]
    known_map = {c: i for i, c in enumerate(known_ids)}
    kmask = roles == 1
    target = np.asarray([known_map.get(int(c), -1) for c in cats[kmask]])
    top = pred["class_logits"][kmask].argmax(1)
    known_top1 = float(np.mean(top == target)) if len(target) else 0.0
    try: obs_auc = float(roc_auc_score(obs.astype(int), pred["observability_logit"]))
    except ValueError: obs_auc = .5
    try: known_auc = float(roc_auc_score((roles == 1).astype(int), pred["known_logit"]))
    except ValueError: known_auc = .5

    # Deterministic cross-video calibration pairs, capped only for the
    # checkpoint diagnostic (the final calibration remains complete).
    novel_local = np.where(roles == 2)[0]
    pair_scores, pair_labels = [], []
    if len(novel_local):
        sem = torch.from_numpy(pred["semantic"]).to(device)
        pairs = []
        for a in novel_local:
            ra = data["rows"][int(idx[a])]
            for b in novel_local:
                if b <= a: continue
                rb = data["rows"][int(idx[b])]
                if int(ra["video_id"]) == int(rb["video_id"]): continue
                pairs.append((a, b, int(cats[a] == cats[b])))
        if len(pairs) > 20000:
            take = np.linspace(0, len(pairs) - 1, 20000, dtype=int); pairs = [pairs[int(i)] for i in take]
        with torch.no_grad():
            for s in range(0, len(pairs), 2048):
                chunk = pairs[s:s + 2048]; a = torch.tensor([x[0] for x in chunk], device=device); b = torch.tensor([x[1] for x in chunk], device=device)
                count = torch.full((len(chunk),), .5, device=device)
                pair_scores.extend(model.pair_logits(sem[a], sem[b], count).cpu().tolist()); pair_labels.extend(x[2] for x in chunk)
    try: pair_auc = float(roc_auc_score(pair_labels, pair_scores))
    except ValueError: pair_auc = .5
    return {"known_closed_top1": known_top1, "observability_auroc": obs_auc,
            "known_reject_auroc": known_auc, "novel_pair_auroc": pair_auc,
            "selection_score": known_top1 + obs_auc + known_auc + pair_auc}


def run(args: argparse.Namespace) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    seed = args.seed + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    data = load_data(args.rows, args.features, args.variant)
    known_ids = [int(x) for x in json.loads(args.known.read_text())]; known_map = {c: i for i, c in enumerate(known_ids)}
    model_base = ObservabilitySemanticModel(768, len(GEOMETRY_FIELDS), args.embedding_dim, len(known_ids)).to(device)
    start_step = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu")
        model_base.load_state_dict(resume["model_state"])
        start_step = int(resume["step"])
    model = DDP(model_base, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model_base
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.updates, eta_min=args.lr * .05)
    # Torch 1.12's default 65536 scale overflowed this normalized multi-head
    # model on the second step before the scaler could settle. A conservative
    # registered initial scale preserves AMP while keeping the finite-gradient
    # smoke meaningful.
    use_scaler = args.amp_dtype == "fp16"
    amp_dtype = torch.float16 if use_scaler else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=args.amp_init_scale, growth_interval=1000)
    local_batch = args.global_batch // world
    if local_batch < 4 or local_batch * world != args.global_batch:
        raise RuntimeError("global batch must divide world size and keep local batch >=4")
    batches = BatchSchedule(data, local_batch, world, rank, args.seed)
    pairs = PairSampler(data, args.seed, rank)
    start = time.time(); curves = []; losses_window = defaultdict(list); best_score = -float("inf"); best_step = 0
    if args.best.exists():
        prior_best = torch.load(args.best, map_location="cpu")
        best_score = float(prior_best.get("calibration_diagnostic", {}).get("selection_score", -float("inf")))
        best_step = int(prior_best.get("step", 0))
    train_rows = len(data["train_idx"]); unique_per_step = (local_batch // 2) * world
    obs_train = data["observable"][data["train_idx"]]
    obs_pos_weight = float(min(12.0, max(1.0, (~obs_train).sum() / max(obs_train.sum(), 1))))
    known_train = data["roles"][data["train_idx"]] == 1
    known_pos_weight = float(min(12.0, max(1.0, (~known_train).sum() / max(known_train.sum(), 1))))

    for step in range(1, args.updates + 1):
        idx = batches.next(); pleft, pright, plabel, pcount = pairs.sample(max(2, local_batch // 2))
        views = tensor(data["views"], idx, device); previous = tensor(data["previous_raw"], idx, device); geom = tensor(data["geometry"], idx, device)
        role = torch.from_numpy(data["roles"][idx]).to(device); observable = torch.from_numpy(data["observable"][idx].astype(np.float32)).to(device)
        cats = data["cats"][idx]
        class_target = torch.tensor([known_map.get(int(c), -1) for c in cats], device=device)
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            pred = model(views, previous, geom)
            obs_loss = F.binary_cross_entropy_with_logits(pred["observability_logit"], observable, pos_weight=torch.tensor(obs_pos_weight, device=device))
            is_known = (role == 1).float()
            known_loss = F.binary_cross_entropy_with_logits(pred["known_logit"], is_known, pos_weight=torch.tensor(known_pos_weight, device=device))
            kmask = class_target >= 0
            if kmask.any():
                per = F.cross_entropy(pred["class_logits"][kmask], class_target[kmask], reduction="none")
                quality_weight = torch.where(observable[kmask] > .5, torch.ones_like(per), torch.full_like(per, .35))
                cls_loss = (per * quality_weight).mean()
            else: cls_loss = pred["semantic"].sum() * 0
            raw_sem = (model.module if world > 1 else model).project_base(views[:, 0])
            view_loss = (1.0 - (pred["semantic"] * raw_sem).sum(1)).mean()
            tmask_np = data["teacher_mask"][idx] & data["assigned"][idx]
            if tmask_np.any() and args.variant == "m1":
                tmask = torch.from_numpy(tmask_np).to(device)
                teacher = (model.module if world > 1 else model).teacher(tensor(data["gt_views"], idx, device)[tmask]).detach()
                teacher_loss = (1.0 - (pred["semantic"][tmask] * teacher).sum(1)).mean()
            else: teacher_loss = pred["semantic"].sum() * 0
            age = np.asarray([int(data["rows"][int(i)]["causal_prefix_age"]) for i in idx])
            tmask2_np = (age > 0) & data["assigned"][idx]
            if tmask2_np.any():
                tmask2 = torch.from_numpy(tmask2_np).to(device)
                prev_sem = (model.module if world > 1 else model).project_base(previous[tmask2]).detach()
                temporal_loss = (1.0 - (pred["semantic"][tmask2] * prev_sem).sum(1)).mean()
            else: temporal_loss = pred["semantic"].sum() * 0
            pair_idx = np.concatenate([pleft, pright]); pair_pred = model(tensor(data["views"], pair_idx, device), tensor(data["previous_raw"], pair_idx, device), tensor(data["geometry"], pair_idx, device))["semantic"]
            half = len(pleft); pair_logit = (model.module if world > 1 else model).pair_logits(pair_pred[:half], pair_pred[half:], torch.from_numpy(pcount).to(device))
            pair_loss = F.binary_cross_entropy_with_logits(pair_logit, torch.from_numpy(plabel).to(device))
            loss = obs_loss + .55 * known_loss + cls_loss + .45 * pair_loss + .35 * teacher_loss + .08 * view_loss + .10 * temporal_loss
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            detail = {"loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()),
                      "scale": float(scaler.get_scale()), "obs": float(obs_loss.detach().cpu()),
                      "known": float(known_loss.detach().cpu()), "class": float(cls_loss.detach().cpu()),
                      "pair": float(pair_loss.detach().cpu()), "teacher": float(teacher_loss.detach().cpu()),
                      "view": float(view_loss.detach().cpu()), "temporal": float(temporal_loss.detach().cpu())}
            raise RuntimeError("non-finite training loss/gradient at step " + str(step) + ": " + json.dumps(detail, sort_keys=True))
        scaler.step(opt); scaler.update(); scheduler.step()
        for name, value in [("total", loss), ("obs", obs_loss), ("known", known_loss), ("class", cls_loss), ("pair", pair_loss), ("teacher", teacher_loss), ("view", view_loss), ("temporal", temporal_loss)]:
            losses_window[name].append(float(value.detach().cpu()))

        global_step = start_step + step
        checkpoint = step == 20 or global_step % args.checkpoint_interval == 0 or step == args.updates
        if checkpoint:
            if world > 1: dist.barrier()
            if rank == 0:
                base = model.module if world > 1 else model
                cal = calibration_diagnostic(base, data, known_ids, device)
                rec = {"step": global_step, "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm.detach().cpu()),
                       "loss": {k: float(np.mean(v)) for k, v in losses_window.items()}, **cal}
                curves.append(rec); losses_window.clear()
                state = {"protocol": "trackocd_iclr27_phase17r_training", "variant": args.variant,
                         "model_name": args.model_name, "step": global_step, "model_state": base.state_dict(),
                         "known_ids": known_ids, "geometry_fields": GEOMETRY_FIELDS,
                         "geometry_mean": data["gmean"], "geometry_std": data["gstd"],
                         "embedding_dim": args.embedding_dim, "feature_source": str(args.features.resolve()),
                         "row_key_sha256": hashlib.sha256(json.dumps([r["row_key"] for r in data["rows"]]).encode()).hexdigest(),
                         "calibration_diagnostic": cal, "amp_dtype": args.amp_dtype, "future_frames_used": False,
                         "gt_deployment_input": False, "physical_id_semantic_feature": False}
                atomic_torch(args.latest, state)
                if cal["selection_score"] > best_score:
                    best_score, best_step = cal["selection_score"], global_step; atomic_torch(args.best, state)
                print(json.dumps(rec, sort_keys=True), flush=True)
            if world > 1: dist.barrier()

    if rank == 0:
        wall = time.time() - start
        summary = {
            "protocol": "trackocd_iclr27_phase17r_full_training",
            "model_name": args.model_name, "variant": args.variant,
            "training_population_rows": train_rows, "calibration_rows": len(data["cal_idx"]),
            "updates": start_step + args.updates, "resumed_from_step": start_step,
            "updates_in_this_run": args.updates, "global_batch": args.global_batch, "world_size": world,
            "unique_rows_advanced_per_step": unique_per_step,
            "complete_unique_row_passes": (start_step + args.updates) * unique_per_step / train_rows,
            "total_sample_effective_passes": (start_step + args.updates) * args.global_batch / train_rows,
            "best_calibration_step": best_step, "best_calibration_score": best_score,
            "parameters": parameter_counts(model.module if world > 1 else model),
            "mixed_precision": True, "amp_dtype": args.amp_dtype, "feature_backbone_frozen": True,
            "wall_seconds": wall, "devices": list(range(world)), "curves": curves,
            "best_checkpoint": str(args.best.resolve()), "latest_checkpoint": str(args.latest.resolve()),
            "observable_target": "assigned and exact row_iou >= 0.5",
            "small_smoke_cancelled_training": False, "q1_labels_used": False,
            "devplus_labels_used": False, "audit_rows_used_for_selection": False,
            "future_frames_used": False, "gt_deployment_input": False,
            "physical_id_semantic_feature": False
        }
        atomic_json(args.summary, summary)
        args.done.write_text(json.dumps({"best": str(args.best), "step": start_step + args.updates, "wall_seconds": wall}))
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["t0", "m1"], required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--known", type=Path, default=ROOT / "data/iclr27_phase17r/sources/supported_known_ids.json")
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--checkpoint-interval", type=int, default=1000)
    ap.add_argument("--amp-init-scale", type=float, default=1024.0)
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("--latest", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--done", type=Path, required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
