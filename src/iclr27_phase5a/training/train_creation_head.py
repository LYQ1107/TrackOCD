"""Train the Phase 5A creation head on genuine-OOV TRAIN episodes.

Teacher replay: known -> ASSIGN; pseudo-novel first-in-stream first step ->
NEW; pseudo-novel later steps / reuse occurrences -> ASSIGN. FP rows are
excluded from supervision. Memory is updated by the teacher (not the head)
so features stay causal.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase5a.assign_create.creation_head import (
    CreationHead,
    decision_features,
    head_action,
)
from src.iclr27_phase5a.assign_create.memory import CategoryMemory
from src.iclr27_phase5a.pilot_gates import load_episodes, summarize


def teacher_replay(data, protos, known_list, embed="h", ema_alpha=0.1,
                   max_age=12.0):
    meta = data["meta"]
    x = data[embed]
    ep_known = data["ep_known"]
    proto_index = {int(c): i for i, c in enumerate(known_list)}
    n_eps = max(int(meta[:, 0].max()) + 1, len(ep_known))
    mem = CategoryMemory(torch.from_numpy(protos), known_list, ema_alpha=ema_alpha)
    X, y = [], []
    for ep in range(n_eps):
        mem.reset()
        active = [int(c) for c in ep_known[ep]]
        mem.known_protos = torch.stack(
            [torch.from_numpy(protos[proto_index[c]]) for c in active])
        mem.known_ids = active
        rows = np.where(meta[:, 0] == ep)[0]
        occ_rows = defaultdict(list)
        for i in rows:
            occ_rows[int(meta[i, 1])].append(i)
        slot_cat = {}
        for oi in sorted(occ_rows):
            idxs = sorted(occ_rows[oi], key=lambda i: int(meta[i, 2]))
            role = int(meta[idxs[0], 3])
            cat = int(meta[idxs[0], 4])
            first = int(meta[idxs[0], 5])
            for t, i in enumerate(idxs):
                h = torch.from_numpy(x[i])
                age = float(t + 1)
                feats = decision_features(h, mem, age, max_age=max_age)
                if role == 0:  # known -> assign
                    X.append(feats.numpy().astype(np.float32))
                    y.append(0)
                elif role == 1 and first and t == 0:  # first step -> new
                    X.append(feats.numpy().astype(np.float32))
                    y.append(1)
                elif role == 1:  # novel later/reuse -> assign
                    X.append(feats.numpy().astype(np.float32))
                    y.append(0)
                else:  # fp excluded
                    continue
                # teacher memory update after prediction
                if role == 1:
                    c = cat
                    if first and t == 0:
                        slot = mem.size
                        mem.novel_protos = torch.cat(
                            [mem.novel_protos, torch.nn.functional.normalize(
                                h.reshape(1, -1), dim=-1)], dim=0)
                        mem.novel_ids.append(slot)
                        slot_cat[c] = slot
                        mem.novel_birth_key[slot] = (
                            int(meta[idxs[0], 6]), int(meta[idxs[0], 7]))
                    elif c in slot_cat:
                        slot = slot_cat[c]
                        p = mem.novel_protos[slot]
                        mem.novel_protos[slot] = torch.nn.functional.normalize(
                            (1 - ema_alpha) * p + ema_alpha * h, dim=-1)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/iclr27_phase5a/pilot/episodes")
    ap.add_argument("--out", default="outputs/iclr27_phase5a/pilot/head")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--ema-alpha", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data_dir = ROOT / args.data
    train = load_episodes(data_dir / "train.npz")
    meta = load_episodes(data_dir / "metadev.npz")
    p = np.load(data_dir / "protos.npz")
    protos = np.asarray(p["protos"], dtype=np.float32)
    known_list = [int(c) for c in p["known_list"]]

    X, y = teacher_replay(train, protos, known_list,
                          ema_alpha=args.ema_alpha)
    print("train samples", X.shape, {k: int(v) for k, v in
          zip(*np.unique(y, return_counts=True))})
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(yt)
    w = torch.tensor([n / max(int((y == 0).sum()), 1),
                      n / max(int((y == 1).sum()), 1)], dtype=torch.float32)
    head = CreationHead()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    best = None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(Xt[idx])
            loss = nn.functional.cross_entropy(logits, yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        acc = float((head(Xt).argmax(1) == yt).float().mean())
        print(f"epoch {ep + 1}: loss={tot / cnt:.4f} train_acc={acc:.4f}",
              flush=True)
        if (ep + 1) % 10 == 0:
            # one end-to-end meta-dev replay (model-in-the-loop)
            head.eval()
            recs = []
            meta_arr = meta["meta"]
            h_all = meta["h"]
            ep_known_all = meta["ep_known"]
            n_eps = max(int(meta_arr[:, 0].max()) + 1, len(ep_known_all))
            mem = CategoryMemory(torch.from_numpy(protos), known_list,
                                 ema_alpha=args.ema_alpha)
            proto_index = {int(c): i for i, c in enumerate(known_list)}
            for epi in range(n_eps):
                mem.reset()
                active = [int(c) for c in ep_known_all[epi]]
                mem.known_protos = torch.stack(
                    [torch.from_numpy(protos[proto_index[c]]) for c in active])
                mem.known_ids = active
                rows = np.where(meta_arr[:, 0] == epi)[0]
                occ_rows = defaultdict(list)
                for i in rows:
                    occ_rows[int(meta_arr[i, 1])].append(i)
                for oi in sorted(occ_rows):
                    idxs = sorted(occ_rows[oi],
                                  key=lambda i: int(meta_arr[i, 2]))
                    key = (int(meta_arr[idxs[0], 6]),
                           int(meta_arr[idxs[0], 7]))
                    for t, i in enumerate(idxs):
                        h = torch.from_numpy(h_all[i])
                        a, sid, _ = head_action(head, h, mem, float(t + 1))
                        recs.append({
                            "ep": epi, "occ": int(meta_arr[i, 1]),
                            "step": t, "role": int(meta_arr[i, 3]),
                            "cat": int(meta_arr[i, 4]),
                            "first": int(meta_arr[i, 5]),
                            "key": key, "action": a, "sid": sid,
                        })
                        if a == "new":
                            slot = mem.size
                            mem.novel_protos = torch.cat(
                                [mem.novel_protos, torch.nn.functional.normalize(
                                    h.reshape(1, -1), dim=-1)], dim=0)
                            mem.novel_ids.append(slot)
                            mem.novel_birth_key[slot] = key
                        elif a == "existing":
                            slot = mem.novel_ids.index(sid)
                            p0 = mem.novel_protos[slot]
                            mem.novel_protos[slot] = torch.nn.functional.normalize(
                                (1 - args.ema_alpha) * p0 +
                                args.ema_alpha * h, dim=-1)
            s = summarize(recs)
            score = (s["known_step_acc"] + s["first_novel_birth_acc"] +
                     s["reuse_acc"]) / 3
            print(f"  meta-dev epoch {ep + 1}: known={s['known_step_acc']:.3f} "
                  f"first={s['first_novel_birth_acc']:.3f} "
                  f"reuse={s['reuse_acc']:.3f} cross={s['cross_physical_reuse_acc']:.3f}",
                  flush=True)
            if best is None or score > best["score"]:
                best = {"epoch": ep + 1, "score": score, "state": head.state_dict()}
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "score": best["score"]}, out_dir / "head.pth")
    (out_dir / "train_meta.json").write_text(json.dumps({
        "samples": int(n), "class_counts": {str(k): int(v) for k, v in
                                            zip(*np.unique(y, return_counts=True))},
        "best_epoch": best["epoch"], "best_score": best["score"],
        "ema_alpha": args.ema_alpha, "seed": args.seed,
    }, indent=2))
    print("saved", out_dir / "head.pth")


if __name__ == "__main__":
    main()
