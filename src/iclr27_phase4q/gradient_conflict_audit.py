#!/usr/bin/env python3
"""Gradient conflict audit between L_TCO and L_cls on shared OVTR params."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OVTR = ROOT / "third_party" / "research_refs_phase4n" / "OVTR" / "ovtr"
sys.path.insert(0, str(OVTR))
sys.path.insert(0, str(OVTR / "models"))

from main import get_args_parser  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402
from util.tool import load_model  # noqa: E402
from models import build_model  # noqa: E402
from datasets import build_dataset  # noqa: E402
from datasets.data_prefetcher import data_dict_to_cuda  # noqa: E402
import util.misc as utils  # noqa: E402


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return None
    return float((a @ b / (na * nb)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    parser = get_args_parser()
    margs = parser.parse_args([
        "--config_file", "./config/ovtr_lite_train_val.py",
        "--dataset_file", "lvis_generated_img_seqs",
        "--device", str(device),
        "--batch_size", "1",
        "--sample_mode", "random_interval",
        "--sample_interval", "1",
        "--sampler_lengths", "2",
        "--merger_dropout", "0",
        "--random_drop", "0.1",
        "--fp_ratio", "0.3",
        "--track_query_iteration", "CIP",
        "--calculate_negative_samples",
        "--max_len", "250",
        "--tco_loss_coef", "1.0",
        "--tco_alpha", "0.5",
        "--with_box_refine", "--two_stage",
        "--pretrain", args.ckpt,
        "--num_workers", "4",
    ])
    cfg = SLConfig.fromfile(margs.config_file)
    model, criterion = build_model(margs, cfg)
    model = load_model(model, args.ckpt)
    model = model.to(device)
    model.train()
    criterion.train()
    print("tco final weight norm",
          float(model.tco_head.net[-1].weight.norm().item()),
          flush=True)

    dataset = build_dataset(image_set="train", args=margs, cfg=cfg.data.train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=utils.mot_collate_fn,
        num_workers=4, pin_memory=True)
    loss_cls = loss_tco = None
    for data in loader:
        data.pop("filename", None)
        data = data_dict_to_cuda(data, device)
        outputs = model(data)
        outputs.pop("track_instances")
        loss_dict = criterion(outputs)
        cls_vals = [v for k, v in loss_dict.items() if k.endswith("_loss_ce")]
        tco_vals = [v for k, v in loss_dict.items() if k.endswith("_loss_tco")]
        loss_cls = sum(cls_vals[1:], cls_vals[0]) if cls_vals else \
            torch.tensor(0.0, requires_grad=True)
        loss_tco = sum(tco_vals[1:], tco_vals[0]) if tco_vals else \
            torch.tensor(0.0, requires_grad=True)
        print("requires_grad", loss_cls.requires_grad,
              loss_tco.requires_grad, "grad_fn_tco",
              loss_tco.grad_fn is not None, flush=True)
        print("batch", loss_cls.item(), loss_tco.item(), flush=True)
        if loss_tco.item() > 0:
            break
    print("loss_cls", loss_cls.item(), "loss_tco", loss_tco.item(),
          flush=True)

    shared = {}
    model.zero_grad(set_to_none=True)
    loss_cls.backward(retain_graph=True)
    grads_cls = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
    model.zero_grad(set_to_none=True)
    loss_tco.backward()
    grads_tco = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
    tco_nonzero = {n: v for n, v in grads_tco.items()
                   if v.flatten().norm().item() > 0}
    print("tco_nonzero sample",
          {k: float(v.flatten().norm().item())
           for k, v in list(tco_nonzero.items())[:20]}, flush=True)

    for n in grads_cls:
        if n in grads_tco:
            shared[n] = {
                "cos": cosine(grads_cls[n], grads_tco[n]),
                "norm_cls": float(grads_cls[n].flatten().norm().item()),
                "norm_tco": float(grads_tco[n].flatten().norm().item()),
            }

    active = {k: v for k, v in shared.items()
              if v["norm_cls"] > 0 and v["norm_tco"] > 0}
    tco_active = {k: v for k, v in shared.items() if v["norm_tco"] > 0}
    print("tco_active sample",
          {k: v["norm_tco"] for k, v in list(tco_active.items())[:20]},
          flush=True)
    groups = {
        "all_shared": active,
        "track_embed": {k: v for k, v in active.items()
                        if k.startswith("track_embed.")},
        "decoder": {k: v for k, v in active.items()
                    if k.startswith("transformer.decoder.")},
        "feature_align": {k: v for k, v in active.items()
                          if k.startswith("feature_align.")},
    }

    def mean_cos(g):
        vals = [v["cos"] for v in g.values() if v["cos"] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "loss_cls": loss_cls.item(),
        "loss_tco": loss_tco.item(),
        "shared_param_count": len(shared),
        "active_shared_param_count": len(active),
        "group_mean_cos": {k: mean_cos(v) for k, v in groups.items()},
        "cos_non_none": sum(1 for v in shared.values() if v["cos"] is not None),
        "sample_shared": {k: v for k, v in list(shared.items())[:12]},
        "conflict": mean_cos(groups["all_shared"]) is not None and
        mean_cos(groups["all_shared"]) < 0.1,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("GRADIENT_CONFLICT_AUDIT_DONE")


if __name__ == "__main__":
    main()
