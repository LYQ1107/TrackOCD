"""Offline state-adaptive residual calibration for the M2 gate.

Replays the frozen ORBIT-MDC M2 on the train-side long stream, collects
(gate stats, memory state, known/novel label) tuples from the *frozen*
rollout, and fits a small bias head b(S_t) so that
  corrected_logit = base_gate_logit - b(S_t)
minimizes the routing BCE.  The base gate stays frozen; only the bias head
is trained.  No official data is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit.protocol import load_train_labels
from src.orbit_msr.evaluate import embed_many
from src.orbit_msr.protocol import known_stats
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_iam.model import ORBITIAMModel
from src.iclr27_phase4d.long_stream import load_stream_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="mdc_g2_cal")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    device = args.device

    ck = torch.load(f"{ROOT}/runs/orbit_mdc/mdc_m2/model.pth",
                    map_location="cpu")
    sd = ck["state_dict"]
    model = ORBITIAMModel(dim=768, bottleneck=128,
                          gate_dim=int(sd["gate.net.0.weight"].shape[1]),
                          reuse_dim=int(sd["reuse.net.0.weight"].shape[1]),
                          hidden=64, use_adapter=True,
                          compat_dim=ck["compat_dim"], state_dim=4).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bias = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1),
    ).to(device)
    opt = torch.optim.AdamW(bias.parameters(), lr=args.lr)

    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    from src.orbit.evaluate import build_known
    known_classes = sorted(set(labels.values()))
    protos, radii = build_known(model, feats, labels, set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows], device)
    mem = IamMemory(protos, radii, novel_update_rate=0.2)
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    pairs = []
    for i, r in enumerate(rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        margin_k = (float(ks[np.argsort(ks)[::-1][0]] -
                          ks[np.argsort(ks)[::-1][1]])
                    if ks.shape[0] >= 2 else 0.0)
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
        best_n = second_n = -1.0
        margin_n = 0.0
        dist_n = 1.0
        if P_novel.shape[0]:
            ns = P_novel @ z
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            dist_n = (1.0 - best_n) / max(mem.novel_radii.get(
                int(sorted(mem.novel)[int(order[0])]), 0.3), 1e-6)
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        st = mem.state_summary()
        state_vec = [st["log_mem"], st["mean_support"],
                     st["low_support_ratio"], st["mean_dispersion"]]
        with torch.no_grad():
            base_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        pairs.append({
            "gs": gs, "state": state_vec, "base_logit": base_logit,
            "label": 1.0 if r["role"] == "known" else 0.0,
        })
        # forward the frozen M2 policy to keep the memory state consistent
        gate_prob = float(torch.sigmoid(torch.as_tensor(base_logit)))
        if gate_prob >= 0.5:
            continue
        nid, ns = mem.existing_novel(z)
        if nid is not None and ns >= 0.45:
            cos = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos, update_radius=True,
                             margin=margin_n)
        else:
            mem.create_novel(z, created_at=i)

    X = torch.as_tensor([p["state"] for p in pairs], dtype=torch.float32,
                        device=device)
    base = torch.as_tensor([p["base_logit"] for p in pairs],
                           dtype=torch.float32, device=device)
    y = torch.as_tensor([p["label"] for p in pairs], dtype=torch.float32,
                        device=device)
    for ep in range(args.epochs):
        bias.train()
        opt.zero_grad()
        b = bias(X).squeeze(-1)
        logits = base - b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) >= 0.5).float() == y).float().mean()
            print(f"epoch {ep} loss {loss.item():.4f} acc {acc.item():.4f}",
                  flush=True)

    out = f"{ROOT}/runs/orbit_msrouting/{args.output_dir}"
    import pathlib
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({"bias": bias.state_dict(),
                "base_checkpoint": "runs/orbit_mdc/mdc_m2/model.pth",
                "n_pairs": len(pairs)}, f"{out}/model.pth")
    print("saved", out, "pairs", len(pairs))


if __name__ == "__main__":
    main()
