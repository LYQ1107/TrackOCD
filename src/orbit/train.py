"""Train ORBIT with episodic pseudo-novel action learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.orbit.action_router import KNOWN, EXISTING_NOVEL, NEW_NOVEL
from src.orbit.episodic_sampler import EpisodicSampler
from src.orbit.losses import action_loss, known_loss, novel_metric_loss, geometry_loss
from src.orbit.model import ORBITModel

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def track_batch(model, frames, mask, device):
    x = _t(np.stack(frames), device)
    m = torch.as_tensor(mask, dtype=torch.bool, device=device)
    return model.aggregate(x, m)


def memory_stats(z, known_protos, novel_protos, reliability, track_len, novel_size):
    # z: (1,d), known_protos: (K,d), novel_protos: (M,d)
    device = z.device
    ks = torch.mm(z, known_protos.t()) if known_protos.shape[0] else torch.empty(1, 0, device=device)
    if ks.numel():
        best_k = ks.max(dim=1).values
        second_k = ks.topk(2, dim=1).values[:, 0] if ks.shape[1] >= 2 else best_k
        margin_k = best_k - second_k
    else:
        best_k = torch.zeros(1, device=device)
        margin_k = torch.zeros(1, device=device)
    ns = torch.mm(z, novel_protos.t()) if novel_protos.shape[0] else torch.empty(1, 0, device=device)
    if ns.numel():
        best_n = ns.max(dim=1).values
        second_n = ns.topk(2, dim=1).values[:, 0] if ns.shape[1] >= 2 else best_n
        margin_n = best_n - second_n
    else:
        best_n = torch.full((1,), -1.0, device=device)
        margin_n = torch.zeros(1, device=device)
    return torch.cat([
        best_k, second_k if ns.numel() else best_k, margin_k,
        (1 - best_k).clamp(min=0),
        best_n, second_n if ns.numel() else best_n, margin_n,
        (1 - best_n).clamp(min=0),
        torch.as_tensor([reliability], device=device),
        torch.as_tensor([track_len / 40.0], device=device),
        torch.as_tensor([novel_size / 300.0], device=device),
    ], dim=0).unsqueeze(0)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ORBITModel(
        dim=768, bottleneck=args.bottleneck,
        use_adapter=args.variant in ("D1", "D2"),
        use_reliability=args.variant == "D2",
        stats_dim=11,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = EpisodicSampler(seed=args.seed, num_known=args.num_known,
                              support_per_class=args.support_per_class,
                              query_per_class=args.query_per_class)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(args.episodes_per_epoch):
            ep = sampler.episode()
            known_classes = ep["known_classes"]
            class_idx = {c: i for i, c in enumerate(known_classes)}
            # support prototypes
            sup_ids = [sid for c in known_classes for sid in ep["support"][c]]
            sup_frames = [sampler.frames_for(sid) for sid in sup_ids]
            sup_mask = np.ones((len(sup_ids), max(len(f) for f in sup_frames)), dtype=bool)
            sup_x = np.zeros((len(sup_ids), sup_mask.shape[1], 768), dtype=np.float32)
            for i, f in enumerate(sup_frames):
                sup_x[i, :len(f)] = f
            agg = track_batch(model, [sup_x[i] for i in range(len(sup_ids))],
                              [sup_mask[i] for i in range(len(sup_ids))], device)
            # aggregate support as one batch
            x = _t(sup_x, device)
            m = torch.as_tensor(sup_mask, dtype=torch.bool, device=device)
            sup_out = model.aggregate(x, m)
            zsup = sup_out["z"]
            protos_list = []
            for c in known_classes:
                idx = [i for i, sid in enumerate(sup_ids) if ep["support"][c].count(sid)]
                # simpler: support ids are grouped per class
            # group by class from ep support order
            protos_list = []
            for c in known_classes:
                ids = ep["support"][c]
                pos = [sup_ids.index(sid) for sid in ids]
                v = F.normalize(zsup[pos].mean(dim=0), dim=-1)
                protos_list.append(v)
            known_protos = torch.stack(protos_list) if protos_list else torch.empty(0, 768, device=device)

            novel_protos = torch.empty(0, 768, device=device)
            novel_labels = []
            loss_acc = 0.0
            n_query = 0
            for q in ep["query"]:
                frames = sampler.frames_for(q["sample_id"])
                mask = torch.ones(1, frames.shape[0], dtype=torch.bool, device=device)
                out = track_batch(model, [frames], [np.ones(len(frames), dtype=bool)], device)
                z = out["z"]
                reliability = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
                stats = memory_stats(z, known_protos, novel_protos, reliability,
                                     len(frames), novel_protos.shape[0])
                logits = model.action_net(stats)
                if q["known"]:
                    at = KNOWN
                    sem_t = torch.tensor([class_idx[q["label"]]], device=device)
                elif q["first"]:
                    at = NEW_NOVEL
                else:
                    at = EXISTING_NOVEL
                at_t = torch.tensor([at], device=device)
                loss = action_loss(logits, at_t)
                if q["known"]:
                    loss = loss + args.lambda_known * known_loss(z, known_protos, sem_t)
                if not q["known"]:
                    if q["first"] or q["label"] not in novel_labels:
                        novel_protos = torch.cat([novel_protos, z.detach()], dim=0)
                        novel_labels.append(q["label"])
                    else:
                        nt = torch.tensor([novel_labels.index(q["label"])], device=device)
                        loss = loss + args.lambda_novel * novel_metric_loss(z, novel_protos, nt)
                        vid = novel_labels.index(q["label"])
                        updated = F.normalize(
                            (1 - args.novel_update_rate) * novel_protos[vid]
                            + args.novel_update_rate * z.detach(), dim=-1)
                        updated = updated.squeeze(0)
                        novel_protos = torch.cat(
                            [novel_protos[:vid], updated.unsqueeze(0), novel_protos[vid + 1:]],
                            dim=0,
                        )
                loss = loss + args.lambda_geo * geometry_loss(z, out["z0"])
                loss_acc = loss_acc + loss
                n_query += 1
            if n_query:
                opt.zero_grad()
                (loss_acc / n_query).backward()
                opt.step()
                total += float(loss_acc.item() / n_query)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch} loss {total/max(args.episodes_per_epoch,1):.4f}", flush=True)
    out_dir = ROOT / "runs" / "orbit" / f"model_{args.variant}_b{args.bottleneck}_g{args.lambda_geo}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                "bottleneck": args.bottleneck, "lambda_geo": args.lambda_geo,
                "novel_update_rate": args.novel_update_rate, "seed": args.seed},
               out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["D0", "D1", "D2"], default="D2")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--episodes_per_epoch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_known", type=int, default=20)
    ap.add_argument("--support_per_class", type=int, default=4)
    ap.add_argument("--query_per_class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1027)
    args = ap.parse_args()
    train(args)
