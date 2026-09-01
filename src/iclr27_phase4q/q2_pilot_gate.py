#!/usr/bin/env python3
"""Fast Q2 pilot gate: E-state separation / birth / no-NaN / new-query
survival on a small number of training batches."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from main import get_args_parser
from models import build_model
from util.tool import load_model
from datasets import build_dataset
from util import misc as utils
from datasets.data_prefetcher import data_dict_to_cuda


def main():
    ap = argparse.ArgumentParser(parents=[get_args_parser()])
    ap.add_argument("--gate-ckpt", required=True)
    ap.add_argument("--gate-out", required=True)
    ap.add_argument("--gate-iters", type=int, default=40)
    args = ap.parse_args()

    args.config_file = "./config/ovtr_lite_train_val.py"
    args.dataset_file = "lvis_generated_img_seqs"
    args.with_box_refine = True
    args.two_stage = True
    args.sampler_steps = [4, 7, 14]
    args.sampler_lengths = [2, 3, 4, 5]
    args.batch_size = 1
    args.num_workers = 4
    args.lr = 2e-4
    args.lr_backbone = 2e-5
    args.lr_drop = [13]
    args.sample_mode = "random_interval"
    args.sample_interval = 1
    args.merger_dropout = 0
    args.random_drop = 0.1
    args.fp_ratio = 0.3
    args.track_query_iteration = "CIP"
    args.calculate_negative_samples = True
    args.max_len = 250
    args.dscq_loss_coef = 1.0
    args.dscq_alpha = 0.5
    args.dscq_detach_evidence = 1
    args.dscq_state_dim = 64
    args.start_epoch = 1
    args.epochs = 2
    args.device = "cuda"

    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(args.config_file)
    model, criterion = build_model(args, cfg)
    model = load_model(model, args.gate_ckpt)
    model.train()
    criterion.train()
    model = model.cuda()

    dataset = build_dataset(image_set="train", args=args, cfg=cfg.data.train)
    dataset.set_epoch(1)
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, 1, drop_last=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=utils.mot_collate_fn,
        num_workers=4, pin_memory=True)

    stats = {
        "persistent_valid_e": [], "persistent_fp_e": [],
        "new_valid_e": [], "new_neg_e": [],
        "new_valid_birth": [], "new_neg_birth": [],
        "new_score": [], "new_valid_score": [],
        "persistent_valid_srel": [],
        "persistent_fp_srel": [],
    }
    any_nan = False
    n = 0
    for data_dict in loader:
        data_dict.pop("filename", None)
        data_dict = data_dict_to_cuda(data_dict, model.text_embeddings.device)
        with torch.no_grad():
            outputs = model(data_dict)
        for ti in outputs.get("track_instances", []):
            if ti is None or len(ti) == 0:
                continue
            matched = (ti.obj_idxes >= 0) & (ti.matched_gt_idxes >= 0)
            pv = matched & (ti.iou > 0.5) & (ti.hit_count >= 2)
            pf = ti.is_fp & (ti.matched_gt_idxes == -1) & (ti.hit_count >= 1)
            nv = matched & (ti.iou > 0.5) & (ti.hit_count == 1)
            nn_ = (~matched) & (ti.hit_count <= 1) & (ti.scores > 0.2)
            stats["persistent_valid_e"].extend(
                ti.e_valid_logit[pv].cpu().tolist())
            stats["persistent_fp_e"].extend(
                ti.e_valid_logit[pf].cpu().tolist())
            stats["new_valid_e"].extend(ti.e_valid_logit[nv].cpu().tolist())
            stats["new_neg_e"].extend(ti.e_valid_logit[nn_].cpu().tolist())
            stats["new_valid_birth"].extend(ti.birth_logit[nv].cpu().tolist())
            stats["new_neg_birth"].extend(ti.birth_logit[nn_].cpu().tolist())
            stats["new_score"].extend(ti.scores[ti.hit_count <= 1].cpu().tolist())
            stats["new_valid_score"].extend(ti.scores[nv].cpu().tolist())
            stats["persistent_valid_srel"].extend(
                ti.s_reliability[pv].cpu().tolist())
            stats["persistent_fp_srel"].extend(
                ti.s_reliability[pf].cpu().tolist())
            for name in ("e_state", "s_state"):
                if torch.isnan(getattr(ti, name)).any():
                    any_nan = True
        n += 1
        if n >= args.gate_iters:
            break

    def mean(x):
        return float(np.mean(x)) if x else None

    def sub(a, b):
        return (a - b) if (a is not None and b is not None) else None

    report = {
        "iters": n,
        "any_nan": any_nan,
        "persistent_valid_e_mean": mean(stats["persistent_valid_e"]),
        "persistent_fp_e_mean": mean(stats["persistent_fp_e"]),
        "new_valid_e_mean": mean(stats["new_valid_e"]),
        "new_neg_e_mean": mean(stats["new_neg_e"]),
        "e_separation": sub(mean(stats["persistent_valid_e"]),
                            mean(stats["persistent_fp_e"])),
        "new_valid_birth_mean": mean(stats["new_valid_birth"]),
        "new_neg_birth_mean": mean(stats["new_neg_birth"]),
        "birth_separation": sub(mean(stats["new_valid_birth"]),
                                mean(stats["new_neg_birth"])),
        "new_score_mean": mean(stats["new_score"]),
        "new_valid_score_mean": mean(stats["new_valid_score"]),
        "persistent_valid_srel_mean": mean(stats["persistent_valid_srel"]),
        "persistent_fp_srel_mean": mean(stats["persistent_fp_srel"]),
        "srel_separation": sub(mean(stats["persistent_valid_srel"]),
                               mean(stats["persistent_fp_srel"])),
        "counts": {k: len(v) for k, v in stats.items()},
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    passed = (
        (not any_nan)
        and (report["e_separation"] is not None and report["e_separation"] > 0.05)
        and (report["new_valid_score_mean"] is not None
             and report["new_valid_score_mean"] > 0.2)
        and (report["new_score_mean"] is not None and report["new_score_mean"] > 0.01)
        and (report["new_valid_birth_mean"] is not None)
    )
    print("PILOT_GATE_PASSED" if passed else "PILOT_GATE_FAILED")


if __name__ == "__main__":
    main()
