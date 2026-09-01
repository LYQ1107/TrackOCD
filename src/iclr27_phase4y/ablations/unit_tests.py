"""ADSSI unit checks: permutation invariance, empty memory, large K."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4y.model import ADSSI, DynamicStateMemory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(ROOT / args.checkpoint, map_location=args.device)
    model = ADSSI(in_dim=256, d=128).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    report = {}
    with torch.no_grad():
        # empty memory
        mem0 = DynamicStateMemory(model, anchors[:4], args.device)
        z = torch.randn(1, 128, device=args.device)
        s0, _, _, _ = mem0.infer(z, 0.5)
        report["empty_memory_scores_shape"] = list(s0.shape)
        # large K
        memK = DynamicStateMemory(model, anchors[:4], args.device)
        for i in range(20):
            memK.create(torch.randn(1, 128, device=args.device), 0.5)
        sK, _, _, _ = memK.infer(z, 0.5)
        report["large_memory_scores_shape"] = list(sK.shape)
        # permutation invariance
        memA = DynamicStateMemory(model, anchors[:4], args.device)
        memB = DynamicStateMemory(model, anchors[:4], args.device)
        hs = [torch.randn(1, 128, device=args.device) for _ in range(4)]
        for h in hs:
            memA.create(h, 0.5)
            memB.create(h, 0.5)
        perm = [2, 0, 3, 1]
        memB.novel = memB.novel[perm]
        sa, _, _, _ = memA.infer(z, 0.5)
        sb, _, _, _ = memB.infer(z, 0.5)
        # known part unchanged; novel part permuted; NEW index differs (same size)
        na = sa[4:8]
        nb = sb[4:8]
        inv = [perm.index(i) for i in range(len(perm))]
        report["permutation_matches"] = bool(
            torch.allclose(na, nb[inv], atol=1e-4))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
