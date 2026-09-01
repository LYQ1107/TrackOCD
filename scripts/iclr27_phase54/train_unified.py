#!/usr/bin/env python3
"""Train one fold of the Phase51 unified causal MOT+OCD graph.

The feature source is frozen only as a visual input; proposal/objectness,
association, lifecycle, representation, semantic-state and controller heads
are all trainable.  TRAIN labels are used to form losses and are never part of
the model input tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, load_aligned_features
from src.iclr27_phase51.unified_model import UnifiedTrackOCD, metadata

ROOT = Path(__file__).resolve().parents[2]
OUT51 = ROOT / "outputs/iclr27_phase51"
OUT54 = ROOT / "outputs/iclr27_phase54"
PREFIXES = (1, 2, 4, 8, 16)
GEOM_FIELDS = (
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm",
    "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


def parse_box(row: dict[str, str], field: str = "gt_bbox_xyxy") -> np.ndarray | None:
    text = row.get(field, "")
    if not text:
        return None
    try:
        vals = np.asarray(json.loads(text), dtype=np.float32)
        if vals.shape != (4,):
            return None
        w = max(float(row.get("image_width", 1) or 1), 1.0)
        h = max(float(row.get("image_height", 1) or 1), 1.0)
        return np.clip(vals / np.asarray([w, h, w, h], dtype=np.float32), 0.0, 1.0)
    except Exception:
        return None


def geom_array(rows: list[dict[str, str]], inds: list[int]) -> np.ndarray:
    return np.asarray([[float(rows[i].get(k, 0.0) or 0.0) for k in GEOM_FIELDS] for i in inds], dtype=np.float32)


def sorted_tracks(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        tracks[row_key(r)].append(i)
    for k in tracks:
        tracks[k].sort(key=lambda i: (int(rows[i].get("event_rank", i)), i))
    return dict(tracks)


def pad_track(k: str, tracks: dict[str, list[int]], raw: np.ndarray,
              geom: np.ndarray, rows: list[dict[str, str]], prefix: int = 16,
              max_len: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inds = tracks[k][: min(prefix, max_len)]
    x = np.zeros((max_len, raw.shape[1]), dtype=np.float32)
    g = np.zeros((max_len, geom.shape[1]), dtype=np.float32)
    m = np.zeros(max_len, dtype=bool)
    obj = np.zeros(max_len, dtype=np.float32)
    ious = np.zeros(max_len, dtype=np.float32)
    gt = np.zeros((max_len, 4), dtype=np.float32)
    gt_mask = np.zeros(max_len, dtype=np.float32)
    for j, idx in enumerate(inds):
        x[j] = raw[idx]
        g[j] = geom[idx]
        m[j] = True
        obj[j] = float(rows[idx].get("assigned", "0") == "1")
        ious[j] = float(np.clip(float(rows[idx].get("row_iou", 0.0) or 0.0), 0.0, 1.0))
        box = parse_box(rows[idx])
        if box is not None:
            gt[j] = box
            gt_mask[j] = 1.0
    return x, g, m, obj, ious, gt, gt_mask


def pair_rows(k: str, tracks: dict[str, list[int]], track_video: dict[str, int],
              fit_keys: list[str], rows: list[dict[str, str]],
              rng: np.random.Generator) -> tuple[int, int, float]:
    inds = tracks[k]
    if len(inds) >= 2:
        a = int(inds[int(rng.integers(len(inds) - 1))]); b = int(inds[inds.index(a) + 1]); return a, b, 1.0
    a = int(inds[0])
    same_video = [q for q in fit_keys if q != k and track_video[q] == track_video[k] and tracks[q]]
    if not same_video:
        same_video = [q for q in fit_keys if q != k and tracks[q]]
    q = same_video[int(rng.integers(len(same_video)))] if same_video else k
    b = int(tracks[q][0])
    return a, b, 0.0


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {"proposal": [], "association": [], "track": [], "semantic": [], "controller": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name.startswith(("proposal_encoder", "objectness_head", "proposal_quality_head", "bbox_delta_head")):
            groups["proposal"].append(p.grad.detach().float().norm())
        elif name.startswith("association_head"):
            groups["association"].append(p.grad.detach().float().norm())
        elif name.startswith(("track_gru", "track_query", "lifecycle_head")):
            groups["track"].append(p.grad.detach().float().norm())
        elif name.startswith(("residual_head", "support_proj", "support_delta", "support_quality_head")):
            groups["semantic"].append(p.grad.detach().float().norm())
        elif name.startswith("controller"):
            groups["controller"].append(p.grad.detach().float().norm())
    return {k: float(torch.stack(v).norm().item()) if v else 0.0 for k, v in groups.items()}


def loss_weights(stage: str) -> dict[str, float]:
    all_w = {
        "objectness": 1.0, "bbox": 1.0, "association": 1.0, "lifecycle": 0.25,
        "continuity": 0.5, "temporal": 0.25, "correspondence": 1.0,
        "hard_negative": 1.0, "prefix": 0.25, "state": 0.5, "commit": 0.5,
        "persistent": 0.25, "mot_safety": 0.5, "raw": 0.25,
    }
    if stage == "warm":
        return {k: (v if k in {"objectness", "bbox", "association", "lifecycle", "continuity", "mot_safety"} else 0.0) for k, v in all_w.items()}
    if stage == "representation":
        return {k: (v if k in {"temporal", "correspondence", "hard_negative", "prefix", "raw", "mot_safety", "continuity"} else 0.15 * v) for k, v in all_w.items()}
    return all_w


def build_batch(records: list[dict[str, Any]], tracks: dict[str, list[int]], raw: np.ndarray,
                geom: np.ndarray, rows: list[dict[str, str]], fit_keys: list[str],
                track_video: dict[str, int], rng: np.random.Generator,
                batch_size: int, device: torch.device) -> dict[str, Any]:
    qs, qg, qm, qobj, qious, qgt, qgtm = [], [], [], [], [], [], []
    ss, sm, ns, nm = [], [], [], []
    neg_qs, neg_qg, neg_qm = [], [], []
    pair_a, pair_b, pair_ga, pair_gb, pair_y = [], [], [], [], []
    for _ in range(batch_size):
        rec = records[int(rng.integers(len(records)))]
        qk = rec["query_track_key"]
        prefix = int(rng.choice(PREFIXES))
        x, g, m, o, io, gt_box_arr, gt_mask_arr = pad_track(qk, tracks, raw, geom, rows, prefix=prefix)
        qs.append(x); qg.append(g); qm.append(m); qobj.append(o); qious.append(io)
        # Keep TRAIN GT boxes in the loss-only batch; they are not model inputs.
        qgt.append(gt_box_arr); qgtm.append(gt_mask_arr)
        sk = [k for k in rec.get("support_track_keys", []) if k in tracks]
        if not sk:
            sk = [qk]
        sx = []; nx = []
        for k in sk[:4]:
            sx.append(pad_track(k, tracks, raw, geom, rows, prefix=prefix)[0].mean(0))
        if not sx:
            sx = [x[m].mean(0) if m.any() else np.zeros(raw.shape[1], np.float32)]
        hk = rec.get("hard_negative_track_key")
        if hk not in tracks:
            hk = fit_keys[int(rng.integers(len(fit_keys)))]
        nx.append(pad_track(hk, tracks, raw, geom, rows, prefix=prefix)[0].mean(0))
        ss.append(np.asarray(sx, np.float32)); sm.append(np.ones(len(sx), dtype=bool))
        ns.append(np.asarray(nx, np.float32)); nm.append(np.ones(len(nx), dtype=bool))
        nqk = hk
        nxseq, ngseq, nmseq, _, _, _, _ = pad_track(nqk, tracks, raw, geom, rows, prefix=prefix)
        neg_qs.append(nxseq); neg_qg.append(ngseq); neg_qm.append(nmseq)
        a, b, y = pair_rows(qk, tracks, track_video, fit_keys, rows, rng)
        pair_a.append(raw[a]); pair_b.append(raw[b]); pair_ga.append(geom[a]); pair_gb.append(geom[b]); pair_y.append(y)
    def t(x, dtype=torch.float32):
        return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)
    # Episodes contain one to several support tracks.  Pad to the registered
    # maximum of four and carry an explicit mask; never concatenate ragged
    # arrays (the failed first smoke exposed this implementation issue).
    max_support = 4
    support_arr = np.zeros((batch_size, max_support, raw.shape[1]), dtype=np.float32)
    support_mask_arr = np.zeros((batch_size, max_support), dtype=bool)
    for i, arr in enumerate(ss):
        take = min(max_support, len(arr))
        if take:
            support_arr[i, :take] = arr[:take]
            support_mask_arr[i, :take] = True
    neg_support_arr = np.zeros((batch_size, 1, raw.shape[1]), dtype=np.float32)
    neg_support_mask_arr = np.ones((batch_size, 1), dtype=bool)
    for i, arr in enumerate(ns):
        if len(arr):
            neg_support_arr[i, 0] = arr[0]
    return {
        "q_raw": t(qs), "q_geom": t(qg), "q_mask": t(qm, torch.bool),
        "q_obj": t(qobj), "q_iou": t(qious), "q_gt": t(np.asarray(qgt)),
        "q_gt_mask": t(np.asarray(qgtm)),
        "support": t(support_arr), "support_mask": t(support_mask_arr, torch.bool),
        "neg_support": t(neg_support_arr), "neg_support_mask": t(neg_support_mask_arr, torch.bool),
        "neg_raw": t(neg_qs), "neg_geom": t(neg_qg), "neg_mask": t(neg_qm, torch.bool),
        "pair_a": t(pair_a), "pair_b": t(pair_b), "pair_ga": t(pair_ga), "pair_gb": t(pair_gb), "pair_y": t(pair_y),
    }


def run(args: argparse.Namespace) -> None:
    for d in ("metrics", "checkpoints", "completion", "logs"):
        (OUT54 / d).mkdir(parents=True, exist_ok=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu):
        raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = 510000 + int(args.fold) * 97 + (0 if args.stage == "warm" else 10000 if args.stage == "representation" else 20000)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    cls, roi, alignment = load_aligned_features(rows)
    raw = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-6)
    geom = geom_array(rows, list(range(len(rows))))
    tracks = sorted_tracks(rows)
    track_video = {k: int(rows[v[0]]["video_id"]) for k, v in tracks.items()}
    fm = json.loads((OUT51 / "manifests" / f"fold_{args.fold}.json").read_text())
    fit_keys = [k for k in fm["fit_track_keys"] if k in tracks]
    ep = json.loads((ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{args.fold}.json").read_text())["records"]
    fit_records = [r for r in ep if r.get("split") == "fit" and r.get("query_track_key") in set(fit_keys)]
    if not fit_records:
        raise RuntimeError(f"no fit episodes for fold {args.fold}")
    tag = args.tag
    run_name = f"{tag}_{args.stage}_f{args.fold}"
    if args.smoke:
        run_name = f"{tag}_{args.stage}_smoke_f{args.fold}"
    marker = OUT54 / "completion" / f"{run_name}.launched"
    done = OUT54 / "completion" / f"{run_name}.done"
    latest = OUT54 / "checkpoints" / f"{run_name}_latest.pt"
    bestp = OUT54 / "checkpoints" / f"{run_name}_best.pt"
    logp = OUT54 / "logs" / f"{run_name}.jsonl"
    if done.exists() and not args.resume:
        return
    if marker.exists() and not args.resume:
        raise RuntimeError(f"refusing relaunch of launched unit {marker}")
    atomic_json(marker, {"phase": 55, "fold": args.fold, "stage": args.stage, "pid": os.getpid(), "gpu": args.expected_physical_gpu, "seed": seed, "started": time.time()})
    model = UnifiedTrackOCD().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start = 0; best = -1e9; best_step = 0
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint.format(fold=args.fold))
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ck["model"], strict=True)
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", 0)); best = float(ck.get("best", -1e9)); best_step = int(ck.get("best_step", 0))
    steps = 100 if args.smoke else int(args.steps)
    batch_size = int(args.batch_size)
    rng = np.random.default_rng(seed + 17)
    weights = loss_weights(args.stage)
    history = []
    t0 = time.time(); logp.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(start + 1, steps + 1):
        b = build_batch(fit_records, tracks, raw, geom, rows, fit_keys, track_video, rng, batch_size, device)
        opt.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if device.type == "cuda" else None
        context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
        with context:
            out = model.encode_sequence(b["q_raw"], b["q_geom"], b["q_mask"], b["support"], b["support_mask"])
            neg = model.encode_sequence(b["neg_raw"], b["neg_geom"], b["neg_mask"], b["support"], b["support_mask"])
            pos_support = model._normalize(b["support"][:, 0])
            pos_sim = (out["semantic"] * pos_support).sum(-1)
            neg_sim = (out["semantic"] * model._normalize(b["neg_support"][:, 0])).sum(-1)
            proposal_logits = out["objectness_logit"]
            proposal_q = out["proposal_quality_logit"]
            valid = b["q_mask"].float()
            obj_loss = (F.binary_cross_entropy_with_logits(proposal_logits.float(), b["q_obj"], reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
            q_loss = (F.binary_cross_entropy_with_logits(proposal_q.float(), b["q_iou"], reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
            # Supervise deltas only where TRAIN GT boxes are present.  Targets
            # are causal labels; the GT never enters model inputs.
            # TRAIN GT boxes are loss-only targets.  The model sees only the
            # proposal geometry; missing GT rows are masked out.
            box_base = b["q_geom"][..., 1:5]
            gt_delta = b["q_gt"] - box_base
            gt_valid = valid * b["q_gt_mask"] * (b["q_obj"] > 0).float()
            bbox_loss = F.smooth_l1_loss(out["bbox_delta"].float(), gt_delta, reduction="none").mean(-1)
            bbox_loss = (bbox_loss * gt_valid).sum() / gt_valid.sum().clamp_min(1.0)
            assoc_logit = model.association(b["pair_a"], b["pair_ga"], b["pair_b"], b["pair_gb"])
            assoc_loss = F.binary_cross_entropy_with_logits(assoc_logit.float(), b["pair_y"])
            # Birth/continuation/termination labels are derived from causal
            # position only (not model inputs).
            lengths = b["q_mask"].long().sum(1)
            life_target = torch.ones_like(lengths[:, None].expand_as(out["lifecycle_logits"][..., 0]), dtype=torch.long)
            life_target.zero_()
            for bi, ln in enumerate(lengths.tolist()):
                if ln > 0:
                    life_target[bi, :ln] = 1
                    life_target[bi, 0] = 0
                    life_target[bi, ln - 1] = 2
            life_loss = F.cross_entropy(out["lifecycle_logits"].reshape(-1, 3), life_target.reshape(-1), reduction="none")
            life_loss = (life_loss * valid.reshape(-1)).sum() / valid.sum().clamp_min(1.0)
            # Association continuity penalty and temporal track embedding.
            continuity_loss = (1.0 - F.cosine_similarity(out["track_query"][:-1], out["track_query"][1:], dim=-1)).mean() if out["track_query"].shape[0] > 1 else out["track_query"].sum() * 0.0
            temporal_loss = (out["semantic"] - out["anchor"].detach()).pow(2).mean()
            corr_loss = (1.0 - pos_sim).mean()
            hard_loss = F.relu(0.20 - pos_sim + neg_sim).mean()
            # A second prefix pass is causal and shares the same query rows.
            prefix_embs = []
            for p in (1, 2, 4, 8):
                pm = b["q_mask"].clone(); pm[:, p:] = False
                po = model.encode_sequence(b["q_raw"], b["q_geom"], pm, b["support"], b["support_mask"])
                prefix_embs.append(po["semantic"])
            prefix_loss = torch.stack([(x - out["semantic"].detach()).pow(2).mean() for x in prefix_embs]).mean()
            state_loss = F.binary_cross_entropy_with_logits(out["support_quality_logit"].float(), torch.ones_like(out["support_quality_logit"]))
            commit_target = torch.zeros((b["q_raw"].shape[0],), dtype=torch.long, device=device)
            defer_target = torch.ones_like(commit_target)
            action_loss = F.cross_entropy(out["action_logits"], commit_target) + F.cross_entropy(neg["action_logits"], defer_target)
            persistent_loss = F.relu(0.60 - out["support_quality"] * (pos_sim + 1.) * 0.5).mean()
            mot_safety = F.binary_cross_entropy_with_logits(proposal_logits.float(), b["q_obj"])
            raw_loss = (out["semantic"] - out["anchor"].detach()).pow(2).mean()
            losses = {
                "objectness": obj_loss, "bbox": bbox_loss, "association": assoc_loss,
                "lifecycle": life_loss, "continuity": continuity_loss, "temporal": temporal_loss,
                "correspondence": corr_loss, "hard_negative": hard_loss, "prefix": prefix_loss,
                "state": state_loss, "commit": action_loss, "persistent": persistent_loss,
                "mot_safety": mot_safety, "raw": raw_loss,
            }
            total = sum(float(weights[k]) * losses[k] for k in losses)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite total at step {step}")
        total.backward()
        gn = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        rec = {"step": step, "stage": args.stage, "loss": float(total.detach().cpu()), "losses": {k: float(v.detach().cpu()) for k, v in losses.items()}, "grad_norms": gn, "pos_sim": float(pos_sim.detach().mean().cpu()), "neg_sim": float(neg_sim.detach().mean().cpu()), "support_quality": float(out["support_quality"].detach().mean().cpu()), "amp": "bf16" if amp_dtype is not None else "fp32"}
        if step == 1 or step % 25 == 0 or step == steps:
            history.append(rec)
            with logp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        if step % int(args.checkpoint_every) == 0 or step == steps:
            score = -float(total.detach().cpu())
            payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "step": step, "best": best, "best_step": best_step, "fold": args.fold, "stage": args.stage, "seed": seed, "metadata": metadata(model), "alignment": alignment, "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "protocol": "phase51_unified_train_only_causal_mot_ocd"}
            atomic_torch(latest, payload); atomic_torch(OUT54 / "checkpoints" / f"{run_name}_step{step:05d}.pt", payload)
            if score > best:
                best, best_step = score, step
                payload["best"], payload["best_step"] = best, best_step
                atomic_torch(bestp, payload)
    final = history[-1] if history else {"step": steps, "loss": None}
    result = {
        "phase": 55, "fold": args.fold, "stage": args.stage, "tag": args.tag,
        "seed": seed, "steps": steps, "smoke": bool(args.smoke), "device": str(device), "physical_gpu": args.expected_physical_gpu,
        "fit_episodes": len(fit_records), "fit_tracks": len(fit_keys), "loss_weights": weights,
        "history": history, "final": final, "best_score": best, "best_step": best_step,
        "checkpoint_best": str(bestp), "checkpoint_latest": str(latest), "marker": str(marker), "done": str(done),
        "proposal_losses_nonzero": bool(any(float(x.get("losses", {}).get("objectness", 0.0)) > 0 for x in history)),
        "association_losses_nonzero": bool(any(float(x.get("losses", {}).get("association", 0.0)) > 0 for x in history)),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "held GT as model input", "category/text/ID features"],
    }
    atomic_json(OUT54 / "metrics" / f"{run_name}.json", result)
    atomic_json(done, {"phase": 55, "fold": args.fold, "stage": args.stage, "steps": steps, "checkpoint": str(bestp), "best_step": best_step})
    print(json.dumps({"fold": args.fold, "stage": args.stage, "steps": steps, "loss": final.get("loss"), "checkpoint": str(bestp)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stage", choices=["warm", "representation", "joint"], default="joint")
    ap.add_argument("--tag", default="phase55_formal")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-checkpoint", default="", help="optional TRAIN-only curriculum checkpoint pattern, with {fold}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expected-physical-gpu", type=int, default=-1)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
