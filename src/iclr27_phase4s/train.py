"""Episodic training of the TrackOCD semantic core."""
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
    EpisodeConfig,
    EpisodeDataset,
    category_prototypes,
    collate_episodes,
    load_episodic_universe,
)
from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4s.runtime import run_episode

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def build_known_matrix(features, by_cat):
    """48-category known prototype matrix in canonical (sorted TAO id) order."""
    k = sorted(known_ids())
    protos = category_prototypes(features, by_cat)
    rows = []
    for c in k:
        if c in protos:
            rows.append(protos[c])
        else:
            rows.append(np.zeros(768, dtype=np.float32))
    mat = torch.from_numpy(np.stack(rows).astype(np.float32))
    mat = mat / (mat.norm(dim=-1, keepdim=True) + 1e-12)
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase4s/full_model")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes-per-epoch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lambda-contrast", type=float, default=1.0)
    ap.add_argument("--lambda-known", type=float, default=0.3)
    ap.add_argument("--lambda-mem", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    by_train, by_dev, features = load_episodic_universe()
    cfg = EpisodeConfig(seed=args.seed)
    known_mat = build_known_matrix(features, {**by_train, **by_dev})
    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}

    model = SemanticCore(768, 256, known_prototypes=known_mat).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def make_ds(by_cat, n, seed):
        return EpisodeDataset(by_cat, features, cfg, n, seed=seed)

    step = 0
    t0 = time.time()
    log_path = out / "train.log"
    with open(log_path, "w") as logf:
        for epoch in range(args.epochs):
            ds = make_ds(by_train, args.episodes_per_epoch, args.seed + 1000 * epoch)
            dl = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_episodes, pin_memory=True,
            )
            ep_loss = {"total": 0.0, "decision": 0.0, "contrast": 0.0, "known": 0.0, "mem": 0.0}
            n_ep = 0
            for batch in dl:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(args.device)
                teacher = {"cat_to_teacher": {}, "n_teacher": 0, "teacher_to_mem": {}}
                B = batch["feats"].shape[0]
                loss_total = torch.zeros((), device=args.device)
                for b in range(B):
                    sb = {k: (v[b] if isinstance(v, torch.Tensor) else v[b]) for k, v in batch.items()}
                    memory = NovelMemory(args.device)
                    res = run_episode(model, sb, cfg, known_cat_index, known_list, memory, teacher, "train")
                    l_dec = -res["decision"].mean()
                    l = l_dec
                    if res["contrast"] is not None:
                        l = l + args.lambda_contrast * res["contrast"].mean()
                        ep_loss["contrast"] += float(res["contrast"].mean())
                    if res["known"] is not None:
                        l = l + args.lambda_known * res["known"].mean()
                        ep_loss["known"] += float(res["known"].mean())
                    mem_terms = []
                    if res["mem_pull"] is not None:
                        mem_terms.append(res["mem_pull"])
                    if res["mem_push"] is not None:
                        mem_terms.append(res["mem_push"])
                    if mem_terms:
                        mem_loss = torch.cat([t.reshape(-1) for t in mem_terms]).mean()
                        l = l + args.lambda_mem * mem_loss
                        ep_loss["mem"] += float(mem_loss)
                    loss_total = loss_total + l
                    ep_loss["decision"] += float(l_dec)
                loss_total = loss_total / B
                opt.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                step += 1
                n_ep += B
                ep_loss["total"] += float(loss_total)
                if step % 25 == 0:
                    line = (f"epoch {epoch} step {step} total {ep_loss['total']/n_ep:.4f} "
                            f"dec {ep_loss['decision']/n_ep:.4f} con {ep_loss['contrast']/n_ep:.4f} "
                            f"kn {ep_loss['known']/n_ep:.4f} mem {ep_loss['mem']/n_ep:.4f} "
                            f"{time.time()-t0:.0f}s")
                    print(line, flush=True)
                    logf.write(line + "\n")
                    logf.flush()
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "args": vars(args),
            }
            torch.save(ckpt, out / f"checkpoint_epoch{epoch:03d}.pth")
        torch.save(ckpt, out / "checkpoint.pth")
    meta = {
        "seed": args.seed,
        "epochs": args.epochs,
        "steps": step,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", meta)


if __name__ == "__main__":
    main()
