"""Train ADSSI (Phase 4Y Y1) with model-in-the-loop episodic rollout.

Loss: assignment CE over the dynamic state set (+ NEW proposal) on valid
occurrences; FP gets an entropy-maximization regularizer. Memory evolves
from model actions after an optional teacher-forced warm-up.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr
from src.iclr27_phase4y.model import ADSSI, DynamicStateMemory


def run_episode_train(model, ep, store, cfg, tsr, anchors, cat_index, device,
                      teacher_memory: bool, fp_entropy_w: float,
                      birth_margin: float):
    active_idx = [cat_index[c] for c in ep["pseudo_known"]]
    active_anchors = anchors[active_idx]
    mem = DynamicStateMemory(model, active_anchors, device)
    slot_cat = {}
    total = torch.zeros((), device=device)
    n_valid = 0
    stats = defaultdict(int)
    for occ in ep["occurrences"]:
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        state = tsr.init_state(1, device)
        cat = occ["category"]
        target = None
        if occ["role"] == "known":
            target = ep["pseudo_known"].index(cat)
        elif occ["role"] == "novel":
            if occ.get("first"):
                target = len(active_idx) + mem.size()  # NEW
            else:
                k = slot_cat.get(cat)
                target = len(active_idx) + k if k is not None else None
        for t in range(n):
            f = torch.from_numpy(z[t:t + 1]).to(device)
            qt = torch.from_numpy(q[t:t + 1]).to(device)
            rs = float(np.clip(q[t, 0], 0.05, 0.95))
            s, state = tsr.step(f, qt, state)
            if t < cfg.min_commit_age - 1:
                continue
            zt = model.obs(s)
            scores, prop, _, _ = mem.infer(zt, float(q[t, 0]))
            post = torch.softmax(scores, dim=-1)
            new_idx = len(active_idx) + mem.size()
            if occ["role"] == "fp":
                if fp_entropy_w > 0:
                    total = total + fp_entropy_w * (-(post * torch.log(post + 1e-9)).sum())
                action = int(post.argmax())
            else:
                if target is not None:
                    total = total + F.cross_entropy(scores.unsqueeze(0),
                                                    torch.tensor([target], device=device))
                    if birth_margin > 0 and target != new_idx and int(post.argmax()) == new_idx:
                        total = total + birth_margin * torch.relu(
                            scores[new_idx] - scores[target] + 0.5)
                    n_valid += 1
                action = int(post.argmax())
            # memory update
            if teacher_memory and occ["role"] == "novel":
                if occ.get("first") and cat not in slot_cat:
                    k = mem.create(prop, rs)
                    slot_cat[cat] = k
                elif cat in slot_cat:
                    mem.update(slot_cat[cat], zt, rs)
            elif not teacher_memory:
                if action == new_idx:
                    k = mem.create(prop, rs)
                    slot_cat[cat] = k
                    stats["birth"] += 1
                elif action >= len(active_idx) and action < len(active_idx) + mem.size():
                    k = action - len(active_idx)
                    mem.update(k, zt, rs)
                    if occ["role"] == "novel" and cat not in slot_cat:
                        slot_cat[cat] = k
            break  # one decision per occurrence
    return total, n_valid, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fp-entropy-w", type=float, default=0.1)
    ap.add_argument("--birth-margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"]
    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    model = ADSSI(in_dim=256, d=128).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cfg = WEpisodeConfig()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        teacher = epoch < args.warmup_epochs
        rng = random.Random(args.seed + 1000 * epoch)
        model.train()
        tot = 0.0
        nv = 0
        births = 0
        for _ in range(args.n_episodes // args.batch_size):
            opt.zero_grad()
            loss = torch.zeros((), device=args.device)
            nv_b = 0
            for _ in range(args.batch_size):
                ep = make_episode(store, pool, cfg, rng)
                l, n, st = run_episode_train(
                    model, ep, store, cfg, tsr, anchors, cat_index,
                    args.device, teacher, args.fp_entropy_w, args.birth_margin)
                loss = loss + l
                nv_b += n
                births += st["birth"]
            loss = loss / args.batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nv += nv_b
        print(f"ep {epoch+1} loss {tot/(args.n_episodes//args.batch_size):.4f} "
              f"nvalid {nv} births {births} teacher {teacher}", flush=True)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       out / f"checkpoint_ep{epoch+1:03d}.pth")
    torch.save({"model": model.state_dict(), "args": vars(args)},
               out / "adssi.pth")
    (out / "train_meta.json").write_text(json.dumps({
        "epochs": args.epochs, "warmup": args.warmup_epochs,
        "n_episodes": args.n_episodes, "seed": args.seed,
    }, indent=2))
    print("saved", out / "adssi.pth")


if __name__ == "__main__":
    main()
