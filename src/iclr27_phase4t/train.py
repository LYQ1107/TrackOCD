"""Phase 4T hierarchical belief training.

--data real|synthetic; --use-hierarchy; --use-defer; --use-qphys select the
T-matrix variants:
  T1: hierarchy + synthetic episodes (structure-only)
  T2: flat (Phase4S core) + real stream (data-only)
  T3: hierarchy + real stream, forced decisions, scalar reliability
  T4: hierarchy + real stream + learned q_phys + defer
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.iclr27_phase4s.episodes import (
    EpisodeConfig as SynConfig,
    EpisodeDataset,
    category_prototypes,
    collate_episodes,
    load_episodic_universe,
)
from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4s.runtime import run_episode as run_episode_flat
from src.iclr27_phase4s.train import build_known_matrix
from src.iclr27_phase4t.episodes import (
    RealEpisodeConfig,
    RealEpisodeDataset,
    RealStreamStore,
    collate_real,
)
from src.iclr27_phase4t.model import HierarchicalCore
from src.iclr27_phase4t.runtime import run_episode as run_episode_hier

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--stream-csv", default="outputs/iclr27_phase4t/train_stream/proposals.csv")
    ap.add_argument("--stream-feats", default="outputs/iclr27_phase4t/train_stream/feats.npz")
    ap.add_argument("--use-hierarchy", action="store_true")
    ap.add_argument("--use-defer", action="store_true")
    ap.add_argument("--use-qphys", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes-per-epoch", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lambda-contrast", type=float, default=1.0)
    ap.add_argument("--lambda-known", type=float, default=0.3)
    ap.add_argument("--lambda-mem", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}

    by_train, by_dev, syn_features = load_episodic_universe()
    known_mat = build_known_matrix(syn_features, {**by_train, **by_dev})

    if args.use_hierarchy:
        model = HierarchicalCore(768, 256, known_prototypes=known_mat,
                                 use_defer=args.use_defer, use_qphys=args.use_qphys).to(args.device)
    else:
        model = SemanticCore(768, 256, known_prototypes=known_mat).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.data == "real":
        import csv
        rows = list(csv.DictReader(open(ROOT / args.stream_csv)))
        for r in rows:
            r["video_id"] = int(r["video_id"])
            r["frame_id"] = int(r["frame_id"])
            r["track_id"] = int(r["track_id"])
            r["score"] = float(r["score"])
            r["q_phys"] = json.loads(r["q_phys"]) if isinstance(r["q_phys"], str) else r["q_phys"]
            r["bbox_xyxy"] = json.loads(r["bbox_xyxy"]) if isinstance(r["bbox_xyxy"], str) else r["bbox_xyxy"]
            r["gt_role"] = r["gt_role"]
            r["gt_category_id"] = int(r["gt_category_id"])
            r["gt_iou"] = float(r["gt_iou"])
            r["gt_track_id"] = int(r["gt_track_id"])
            r["prior_hits"] = int(r["prior_hits"])
            r["age"] = int(r["age"])
            r["gap"] = int(r["gap"])
            r["run_score_mean"] = float(r["run_score_mean"])
        feats = np.load(ROOT / args.stream_feats)["feats"]
        store = RealStreamStore(rows, feats)
        cfg = RealEpisodeConfig(seed=args.seed)
        n_episodes = args.episodes_per_epoch
        ds = RealEpisodeDataset(store, cfg, n_episodes, seed=args.seed)
        collate = collate_real
    else:
        cfg = SynConfig(seed=args.seed)
        ds = EpisodeDataset(by_train, syn_features, cfg,
                            args.episodes_per_epoch, seed=args.seed)
        collate = collate_episodes

    t0 = time.time()
    step = 0
    log_path = out / "train.log"
    with open(log_path, "w") as logf:
        for epoch in range(args.epochs):
            if args.data == "real":
                ds = RealEpisodeDataset(store, cfg, args.episodes_per_epoch,
                                        seed=args.seed + 1000 * epoch)
            else:
                ds = EpisodeDataset(by_train, syn_features, cfg,
                                    args.episodes_per_epoch,
                                    seed=args.seed + 1000 * epoch)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)
            ep_loss = {"total": 0.0, "l1": 0.0, "known": 0.0, "l2": 0.0,
                       "con": 0.0, "mem": 0.0}
            n_ep = 0
            for batch in dl:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(args.device)
                loss_total = torch.zeros((), device=args.device)
                B = batch["feats"].shape[0]
                for b in range(B):
                    sb = {k: (v[b] if isinstance(v, torch.Tensor) else v[b]) for k, v in batch.items()}
                    memory = NovelMemory(args.device)
                    teacher = {"cat_to_teacher": {}, "n_teacher": 0, "teacher_to_mem": {}}
                    if args.use_hierarchy:
                        res = run_episode_hier(model, sb, cfg, known_cat_index,
                                               known_list, memory, teacher, "train")
                    else:
                        res = run_episode_flat(model, sb, cfg, known_cat_index,
                                               known_list, memory, teacher, "train")
                    l = -res["l1"].mean() if "l1" in res and res["l1"] is not None else 0.0
                    ep_loss["l1"] += float(l) if isinstance(l, torch.Tensor) else 0.0
                    if res.get("known") is not None:
                        lk = args.lambda_known * res["known"].mean()
                        l = l + lk
                        ep_loss["known"] += float(lk)
                    if res.get("l2") is not None:
                        l2 = -res["l2"].mean()
                        l = l + l2
                        ep_loss["l2"] += float(l2)
                    if res.get("contrast") is not None:
                        lc = args.lambda_contrast * res["contrast"].mean()
                        l = l + lc
                        ep_loss["con"] += float(lc)
                    mem_terms = [res[k] for k in ("mem_pull", "mem_push") if res.get(k) is not None]
                    if mem_terms:
                        lm = args.lambda_mem * torch.cat([t.reshape(-1) for t in mem_terms]).mean()
                        l = l + lm
                        ep_loss["mem"] += float(lm)
                    loss_total = loss_total + l
                loss_total = loss_total / B
                opt.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                step += 1
                n_ep += B
                ep_loss["total"] += float(loss_total)
                if step % 20 == 0:
                    line = (f"ep{epoch} st{step} tot {ep_loss['total']/n_ep:.4f} "
                            f"l1 {ep_loss['l1']/n_ep:.4f} kn {ep_loss['known']/n_ep:.4f} "
                            f"l2 {ep_loss['l2']/n_ep:.4f} con {ep_loss['con']/n_ep:.4f} "
                            f"mem {ep_loss['mem']/n_ep:.4f} {time.time()-t0:.0f}s")
                    print(line, flush=True)
                    logf.write(line + "\n")
                    logf.flush()
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "args": vars(args)}, out / f"checkpoint_epoch{epoch:03d}.pth")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "args": vars(args)}, out / "checkpoint.pth")
    (out / "train_meta.json").write_text(json.dumps(
        {"epochs": args.epochs, "steps": step, "wall_seconds": round(time.time() - t0, 1)}))
    print("done")


if __name__ == "__main__":
    main()
