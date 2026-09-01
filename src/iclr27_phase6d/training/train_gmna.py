"""Train Phase 6D GMNA: TSE + momentum teacher + global memory bank."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase6d.model.global_tse import GlobalTSE, GlobalMemoryBank, gmna_loss
from src.iclr27_phase6c.model.tse import KnownAnchors


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def pca_class_means(full, pca_path):
    pca = np.load(pca_path)
    comp = pca["components"].astype(np.float32)
    mean = pca["mean"].astype(np.float32)
    known_idx = np.where(full["is_known"] == 1)[0]
    labels = full["labels"][known_idx]
    ids = np.unique(labels)
    out = []
    for c in ids:
        m = full["mean_feats"][known_idx][labels == c].mean(axis=0)
        v = (m - mean) @ comp.T
        v = v / (np.linalg.norm(v) + 1e-12)
        out.append(v)
    return ids, np.stack(out)


def sample_known(full, cat2idx, n_per_class, rng):
    known_idx = np.where(full["is_known"] == 1)[0]
    labels = full["labels"][known_idx]
    idx2label = {int(p): int(c) for p, c in zip(known_idx, labels)}
    classes = np.unique(labels)
    pick = []
    for c in classes:
        idx = known_idx[labels == c]
        cnt = rng.randint(1, n_per_class + 1) if rng.random() < 0.75 else 0
        if cnt:
            pick.extend(rng.choice(idx, size=cnt, replace=True).tolist())
    if not pick:
        pick = rng.choice(known_idx, size=32, replace=True).tolist()
    pick = np.asarray(pick)
    return (
        torch.from_numpy(full["mean_feats"][pick].astype(np.float32)),
        torch.from_numpy(np.array([cat2idx[idx2label[int(p)]] for p in pick],
                                  dtype=np.int64)),
    )


def sample_unlabeled(n, batch, rng):
    return torch.from_numpy(rng.choice(n, size=batch, replace=False).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase6d/training/gmna_main")
    ap.add_argument("--pool", choices=["full", "small"], default="full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--known-per-class", type=int, default=2)
    ap.add_argument("--unlabeled-batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-attr", type=float, default=1.0)
    ap.add_argument("--w-nb", type=float, default=2.0)
    ap.add_argument("--w-ts", type=float, default=0.5)
    ap.add_argument("--w-pres", type=float, default=0.1)
    ap.add_argument("--bank-k", type=int, default=10)
    ap.add_argument("--bank-conf", type=float, default=0.45)
    ap.add_argument("--bank-alpha", type=float, default=0.90)
    ap.add_argument("--teacher-momentum", type=float, default=0.999)
    ap.add_argument("--w-open", type=float, default=0.0)
    ap.add_argument("--open-margin", type=float, default=0.50)
    ap.add_argument("--novel-thresh", type=float, default=1.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.pool == "full":
        full = {k: np.asarray(v) for k, v in np.load(
            ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz").items()}
        known_npz = full
    else:
        u = {k: np.asarray(v) for k, v in np.load(
            ROOT / "outputs/iclr27_phase6c/assets/unlabeled_tracks.npz").items()}
        ug = {k: np.asarray(v) for k, v in np.load(
            ROOT / "outputs/iclr27_phase6c/assets/unlabeled_gt_tracks.npz").items()}
        full = {
            "mean_feats": np.concatenate([u["mean_feats"], ug["mean_feats"]]),
            "video_ids": np.concatenate([u["video_ids"], ug["video_ids"]]),
            "sample_ids": np.concatenate([
                np.asarray([f"p{i}" for i in range(len(u["mean_feats"]))]),
                np.asarray([f"g{i}" for i in range(len(ug["mean_feats"]))])]),
        }
        known_npz = {k: np.asarray(v) for k, v in np.load(
            ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz").items()}
        known_npz["is_known"] = np.ones(len(known_npz["mean_feats"]),
                                        dtype=np.uint8)

    pca_path = ROOT / "outputs/iclr27_phase6c/assets/pca.npz"
    known_ids, pca_means = pca_class_means(known_npz, pca_path)
    cat2idx = {int(c): i for i, c in enumerate(known_ids)}
    model = GlobalTSE(teacher_momentum=args.teacher_momentum).to(dev)
    model.load_pca(pca_path)
    anchors = KnownAnchors(known_ids, init_feats=pca_means).to(dev)
    n_all = len(full["mean_feats"])
    bank = GlobalMemoryBank(n_all, 128, full["video_ids"], full["sample_ids"],
                            alpha=args.bank_alpha).to(dev)

    params = [{"params": model.student.parameters()},
              {"params": anchors.parameters(), "lr": args.lr * 0.5}]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(20000 / (48 + args.unlabeled_batch)))
    if args.max_steps:
        steps_per_epoch = args.max_steps
    sched = CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    print(f"pool {args.pool}: tracks {n_all}, "
          f"known-labeled {known_npz['is_known'].sum()}, "
          f"steps/epoch {steps_per_epoch}")
    for ep in range(args.epochs):
        t0 = time.time()
        # refresh global bank with teacher embeddings
        model.eval()
        with torch.no_grad():
            embs = []
            for i in range(0, n_all, 512):
                x = torch.from_numpy(
                    full["mean_feats"][i:i + 512].astype(np.float32)).to(dev)
                embs.append(model.teacher_embed(x))
            new_emb = torch.cat(embs)
        bank.update(new_emb)
        bank.build_targets(k=args.bank_k, conf_min=args.bank_conf,
                           prefer_cross_video=True,
                           anchors=anchors.normalized().detach(),
                           novel_thresh=args.novel_thresh)
        n_tgt = int(bank.has_target.sum().item())
        model.train()
        logs = {}
        for step in range(steps_per_epoch):
            kf, kl = sample_known(known_npz, cat2idx, args.known_per_class, rng)
            ui = sample_unlabeled(n_all, args.unlabeled_batch, rng)
            kf, kl = kf.to(dev), kl.to(dev)
            ui = ui.to(dev)
            uf = torch.from_numpy(full["mean_feats"][ui.cpu().numpy()].astype(
                np.float32)).to(dev)
            zk = model.student.project(kf)
            zu = model.student.project(uf)
            zt = model.teacher_embed(uf.detach())
            an = anchors.normalized()
            logits = zk @ an.t() / 0.07
            ce = F.cross_entropy(logits, kl)
            attr = (1.0 - (zk * an[kl]).sum(-1)).mean()
            tgt = bank.targets[ui]
            has = bank.has_target[ui]
            nb, ts, nb_raw, ts_raw = gmna_loss(
                zu, zt, tgt, has, an, w_nb=args.w_nb, w_ts=args.w_ts)
            pres = F.mse_loss(an, torch.from_numpy(pca_means).to(dev))
            maxk = (zu @ an.t()).max(dim=-1).values
            open_mask = maxk < args.novel_thresh
            open_loss = torch.zeros((), device=dev)
            if open_mask.any():
                open_loss = torch.relu(
                    maxk[open_mask] - args.open_margin).pow(2).mean()
            total = (ce + args.w_attr * attr + nb + ts
                     + args.w_pres * pres + args.w_open * open_loss)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.student.parameters()) + list(anchors.parameters()), 5.0)
            opt.step()
            sched.step()
            model.update_teacher()
            for name, v in (("ce", ce), ("attr", attr), ("nb", nb_raw),
                            ("ts", ts_raw), ("pres", pres), ("open", open_loss)):
                logs[name] = logs.get(name, 0.0) + float(v.detach())
        for name in logs:
            logs[name] /= steps_per_epoch
        print(f"epoch {ep + 1}/{args.epochs} "
              f"{time.time() - t0:.1f}s bank_targets={n_tgt} " +
              " ".join(f"{name}={v:.4f}" for name, v in logs.items()), flush=True)
        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            torch.save({
                "model": {k: v.detach().cpu() for k, v in
                          model.student.state_dict().items()},
                "anchors": {k: v.detach().cpu() for k, v in
                            anchors.state_dict().items()},
                "known_ids": known_ids,
                "epoch": ep + 1,
            }, out / f"checkpoint_{ep + 1:03d}.pth")
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.student.state_dict().items()},
        "anchors": {k: v.detach().cpu() for k, v in anchors.state_dict().items()},
        "known_ids": known_ids,
        "epoch": args.epochs,
    }, out / "checkpoint.pth")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print("saved", out / "checkpoint.pth")


if __name__ == "__main__":
    main()
