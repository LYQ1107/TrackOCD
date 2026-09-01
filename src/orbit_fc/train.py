"""Train ORBIT-FC with episodic pseudo-novel factorized action learning."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.orbit.episodic_sampler import EpisodicSampler
from src.orbit_fc.losses import orbit_fc_loss
from src.orbit_fc.model import ORBITFCModel
from src.orbit_fc.protocol import (
    ROOT,
    frozen_known_protos,
    known_stats,
    novel_stats,
    stats_to_tensor,
)


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def aggregate_one(model, frames, device):
    x = _t(frames[:8], device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    return model.aggregate(x, mask)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_anchor = args.variant in ("F2", "F3")
    gate_dim = 12 if use_anchor else 11
    reuse_dim = 11
    model = ORBITFCModel(dim=768, bottleneck=args.bottleneck,
                         gate_dim=gate_dim, reuse_dim=reuse_dim, hidden=64,
                         use_adapter=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = EpisodicSampler(seed=args.seed, num_known=args.num_known,
                              support_per_class=args.support_per_class,
                              query_per_class=args.query_per_class)
    # frozen DINO prototypes for the 38 meta-train classes
    frozen = frozen_known_protos(sampler.meta_train)
    frozen_ids = sorted(frozen)
    frozen_mat = np.stack([frozen[c] for c in frozen_ids]).astype(np.float32)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n_ep = 0
        for _ in range(args.episodes_per_epoch):
            ep = sampler.episode()
            class_idx = {c: i for i, c in enumerate(ep["known_classes"])}
            # support prototypes (adapted space)
            sup_ids = [sid for c in ep["known_classes"] for sid in ep["support"][c]]
            sup_z = []
            for sid in sup_ids:
                out = aggregate_one(model, sampler.frames_for(sid), device)
                sup_z.append(out["z"][0].detach())
            zsup = torch.stack(sup_z)
            protos_list = []
            radii_dict = {}
            pos = 0
            for c in ep["known_classes"]:
                n_sup = len(ep["support"][c])
                v = torch.nn.functional.normalize(zsup[pos:pos + n_sup].mean(dim=0), dim=-1)
                protos_list.append(v)
                cos = (zsup[pos:pos + n_sup] * v.unsqueeze(0)).sum(-1)
                r = float(torch.quantile(1.0 - cos, 0.5).item())
                radii_dict[c] = max(r, 0.02)
                pos += n_sup
            P_known = torch.stack(protos_list).detach() if protos_list else torch.empty(0, 768, device=device)
            # frozen-DINO matrix restricted to episode known classes
            ep_frozen_idx = [frozen_ids.index(c) for c in ep["known_classes"] if c in frozen_ids]
            P_frozen = torch.as_tensor(frozen_mat[ep_frozen_idx], dtype=torch.float32,
                                       device=device) if ep_frozen_idx else torch.empty(0, 768, device=device)

            # episode novel memory (causal within episode)
            novel_list = []
            novel_counts = {}
            novel_radii = {}
            novel_created = {}
            novel_vid_by_label = {}

            loss_acc = 0.0
            n_query = 0
            for q in ep["query"]:
                out = aggregate_one(model, sampler.frames_for(q["sample_id"]), device)
                z = out["z"]
                z0 = out["z0"]
                rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
                length = int(out["length"][0])
                z_np = z[0].detach().cpu().numpy()
                P_known_np = P_known.detach().cpu().numpy()
                P_frozen_np = P_frozen.detach().cpu().numpy()
                P_novel_np = np.stack(novel_list).astype(np.float32) if novel_list else np.empty((0, 768), dtype=np.float32)
                best_n = -1.0; second_n = -1.0; margin_n = 0.0; dist_n = 1.0
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                    r_n = novel_radii.get(int(order[0]), 0.3)
                    dist_n = (1.0 - best_n) / max(r_n, 1e-6)
                anchor = float(np.max(z0[0].detach().cpu().numpy() @ P_frozen_np.T)) if P_frozen_np.shape[0] else -1.0
                gs = known_stats(z_np, P_known_np, radii_dict,
                                 known_ids=ep["known_classes"], anchor_best=anchor,
                                 best_n=best_n, second_n=second_n, margin_n=margin_n,
                                 dist_n=dist_n, rel=rel, track_len=length,
                                 n_novel=len(novel_list), include_anchor=use_anchor)
                gate_logit = model.gate_forward(stats_to_tensor(gs, device))
                gate_target = torch.tensor([1.0 if q["known"] else 0.0], device=device)

                known_target = novel_target = None
                reuse_logit = torch.tensor([], device=device)
                reuse_target = torch.tensor([], dtype=torch.long, device=device)
                reuse_mask = torch.tensor([], dtype=torch.bool, device=device)
                if q["known"]:
                    known_target = torch.tensor([class_idx[q["label"]]], device=device)
                else:
                    best_k, margin_k = -1.0, 0.0
                    if P_known_np.shape[0]:
                        ks = P_known_np @ z_np
                        best_k = float(ks.max())
                        order = np.argsort(ks)[::-1]
                        margin_k = float(ks[order[0]] - ks[order[1]]) if ks.shape[0] >= 2 else 0.0
                    rs = novel_stats(z_np, P_novel_np, novel_counts, novel_radii,
                                     best_k=best_k, margin_k=margin_k, rel=rel,
                                     track_len=length, n_novel=len(novel_list))
                    reuse_logit = model.reuse_forward(stats_to_tensor(rs, device))
                    reuse_target = torch.tensor([0 if q["first"] else 1], device=device)
                    reuse_mask = torch.tensor([True], dtype=torch.bool, device=device)
                    if q["first"]:
                        novel_list.append(z[0].detach().cpu().numpy().astype(np.float32))
                        vid = len(novel_list) - 1
                        novel_vid_by_label[q["label"]] = vid
                        novel_counts[vid] = 1
                        novel_radii[vid] = 0.3
                        novel_created[vid] = 0
                    else:
                        vid = novel_vid_by_label.get(q["label"])
                        if vid is None:
                            novel_list.append(z[0].detach().cpu().numpy().astype(np.float32))
                            vid = len(novel_list) - 1
                            novel_vid_by_label[q["label"]] = vid
                            novel_counts[vid] = 1
                            novel_radii[vid] = 0.3
                            novel_created[vid] = 0
                        proto_old = novel_list[vid]
                        cos_to_center = float(proto_old @ z_np)
                        new_proto = (1 - args.novel_update_rate) * proto_old + args.novel_update_rate * z_np
                        new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-12)
                        novel_list[vid] = new_proto.astype(np.float32)
                        novel_counts[vid] = novel_counts.get(vid, 1) + 1
                        if args.variant == "F3":
                            novel_radii[vid] = float(
                                (1 - 0.2) * novel_radii.get(vid, 0.3)
                                + 0.2 * max(1.0 - cos_to_center, 1e-3))
                        novel_target = torch.tensor([vid], device=device) if vid < len(novel_list) else None

                loss = orbit_fc_loss(
                    gate_logit, gate_target, reuse_logit, reuse_target, reuse_mask,
                    z, z0, P_known, known_target, P_frozen if use_anchor else None,
                    torch.as_tensor(novel_list, dtype=torch.float32, device=device) if novel_list else None,
                    novel_target,
                    torch.tensor([True], dtype=torch.bool, device=device) if novel_target is not None else torch.tensor([], dtype=torch.bool, device=device),
                    lambda_reuse=args.lambda_reuse, lambda_known=args.lambda_known,
                    lambda_novel=args.lambda_novel, lambda_geo=args.lambda_geo,
                    lambda_sem=args.lambda_sem,
                )
                loss_acc = loss_acc + loss
                n_query += 1
            if n_query:
                opt.zero_grad()
                (loss_acc / n_query).backward()
                opt.step()
                total += float(loss_acc.item() / n_query)
                n_ep += 1
        print(f"epoch {epoch} loss {total / max(n_ep, 1):.4f}", flush=True)

    out_dir = ROOT / "runs" / "orbit_fc" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "bottleneck": args.bottleneck,
        "gate_dim": gate_dim,
        "reuse_dim": reuse_dim,
        "use_anchor": use_anchor,
        "lambda_sem": args.lambda_sem,
        "novel_update_rate": args.novel_update_rate,
        "seed": args.seed,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["F1", "F2", "F3"], default="F2")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lambda_reuse", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--lambda_sem", type=float, default=0.5)
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes_per_epoch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_known", type=int, default=20)
    ap.add_argument("--support_per_class", type=int, default=4)
    ap.add_argument("--query_per_class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--output_dir", default="fc_F2")
    args = ap.parse_args()
    train(args)
