#!/usr/bin/env python3
"""Phase 4R / Q3 pilot gate + observation-existence disentanglement audit.

Reuses the DSCQ state machinery to collect, on training batches:
  - E-state separation (persistent valid vs persistent FP)
  - birth separation
  - semantic reliability separation
  - observation score vs E-state correlation per group
  - low-O/high-E true-object and high-O/low-E persistent-FP patterns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from main import get_args_parser
from models import build_model
from util.tool import load_model
from datasets import build_dataset
from util import misc as utils
from datasets.data_prefetcher import data_dict_to_cuda
from util.slconfig import SLConfig


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()))
    return float((x * y).sum() / denom) if denom > 0 else None


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
        "persistent_valid": {"e": [], "score": [], "srel": []},
        "persistent_fp": {"e": [], "score": [], "srel": []},
        "new_valid": {"e": [], "score": [], "birth": []},
        "new_neg": {"e": [], "score": [], "birth": []},
    }
    low_o_high_e_true = 0
    high_o_low_e_fp = 0
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
            stats["persistent_valid"]["e"].extend(ti.e_valid_logit[pv].cpu().tolist())
            stats["persistent_valid"]["score"].extend(ti.scores[pv].cpu().tolist())
            stats["persistent_valid"]["srel"].extend(ti.s_reliability[pv].cpu().tolist())
            stats["persistent_fp"]["e"].extend(ti.e_valid_logit[pf].cpu().tolist())
            stats["persistent_fp"]["score"].extend(ti.scores[pf].cpu().tolist())
            stats["persistent_fp"]["srel"].extend(ti.s_reliability[pf].cpu().tolist())
            stats["new_valid"]["e"].extend(ti.e_valid_logit[nv].cpu().tolist())
            stats["new_valid"]["score"].extend(ti.scores[nv].cpu().tolist())
            stats["new_valid"]["birth"].extend(ti.birth_logit[nv].cpu().tolist())
            stats["new_neg"]["e"].extend(ti.e_valid_logit[nn_].cpu().tolist())
            stats["new_neg"]["score"].extend(ti.scores[nn_].cpu().tolist())
            stats["new_neg"]["birth"].extend(ti.birth_logit[nn_].cpu().tolist())
            if torch.isnan(ti.e_state).any() or torch.isnan(ti.s_state).any():
                any_nan = True
        n += 1
        if n >= args.gate_iters:
            break

    report = {"iters": n, "any_nan": any_nan, "groups": {}}
    for g, d in stats.items():
        e = np.asarray(d["e"], dtype=np.float64)
        s = np.asarray(d["score"], dtype=np.float64)
        report["groups"][g] = {
            "n": int(len(e)),
            "e_mean": float(e.mean()) if len(e) else None,
            "score_mean": float(s.mean()) if len(s) else None,
            "corr_score_e": pearson(s, e),
        }
        if "birth" in d:
            report["groups"][g]["birth_mean"] = (
                float(np.mean(d["birth"])) if d["birth"] else None)
        if "srel" in d:
            report["groups"][g]["srel_mean"] = (
                float(np.mean(d["srel"])) if d["srel"] else None)

    e_sep = None
    if report["groups"]["persistent_valid"]["e_mean"] is not None and \
            report["groups"]["persistent_fp"]["e_mean"] is not None:
        e_sep = (report["groups"]["persistent_valid"]["e_mean"] -
                 report["groups"]["persistent_fp"]["e_mean"])
    birth_sep = None
    if report["groups"]["new_valid"]["birth_mean"] is not None and \
            report["groups"]["new_neg"]["birth_mean"] is not None:
        birth_sep = (report["groups"]["new_valid"]["birth_mean"] -
                     report["groups"]["new_neg"]["birth_mean"])
    srel_sep = None
    if report["groups"]["persistent_valid"]["srel_mean"] is not None and \
            report["groups"]["persistent_fp"]["srel_mean"] is not None:
        srel_sep = (report["groups"]["persistent_valid"]["srel_mean"] -
                    report["groups"]["persistent_fp"]["srel_mean"])
    report["summary"] = {
        "e_separation": e_sep,
        "birth_separation": birth_sep,
        "srel_separation": srel_sep,
        "corr_score_e_persistent_valid":
            report["groups"]["persistent_valid"]["corr_score_e"],
        "corr_score_e_persistent_fp":
            report["groups"]["persistent_fp"]["corr_score_e"],
    }
    Path(args.gate_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gate_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Q3_PILOT_GATE_JSON_WRITTEN")


if __name__ == "__main__":
    main()
