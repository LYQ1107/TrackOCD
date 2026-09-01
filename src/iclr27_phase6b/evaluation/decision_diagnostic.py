"""Diagnose the DSCT 3-way decision on legal TRAIN batches.

Groups matched queries into supported-known vs held-out pseudo-novel and
reports decision features/logits per group. Legal: TRAIN stream only, no
novel GT, no external labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OVTR = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-batches", type=int, default=40)
    args = ap.parse_args()

    os.chdir(OVTR)
    sys.path.insert(0, str(OVTR))
    from util.slconfig import SLConfig
    from main import get_args_parser
    from datasets import build_dataset
    from datasets.data_prefetcher import data_dict_to_cuda
    from models import build_model
    from util import tool as util_tool
    import util.misc as utils

    train_args = get_args_parser().parse_args([
        "--config_file", str(OVTR / "config/ovtr_lite_dsct6b_train_val.py"),
        "--dataset_file", "lvis_generated_img_seqs",
        "--with_box_refine", "--two_stage",
        "--lr", "2e-4", "--lr_backbone", "2e-5", "--lr_drop", "13",
        "--num_workers", "2", "--batch_size", "1",
        "--sample_mode", "random_interval", "--sample_interval", "1",
        "--sampler_steps", "4", "7", "14",
        "--sampler_lengths", "2", "3", "4", "5",
        "--merger_dropout", "0", "--random_drop", "0.1", "--fp_ratio", "0.3",
        "--track_query_iteration", "CIP", "--calculate_negative_samples",
        "--max_len", "250", "--epochs", "1",
        "--tco_loss_coef", "1.0", "--tco_alpha", "0.5",
        "--dsct_coef", "1.0", "--dsct_state_dim", "128",
        "--dsct_alpha", "0.1", "--dsct_stage", "c",
        "--resume", args.ckpt,
        "--output_dir", str(ROOT / "outputs/iclr27_phase6b/audit"),
    ])
    cfg = SLConfig.fromfile(train_args.config_file)
    model, criterion = build_model(train_args, cfg)
    util_tool.load_model(model, args.ckpt)
    model = model.cuda()
    if getattr(criterion, "anchor_init", None) is not None:
        criterion.anchor_init = criterion.anchor_init.cuda()
    model.train()

    ds = build_dataset(image_set="train", args=train_args, cfg=cfg.data.train)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=utils.mot_collate_fn,
                        num_workers=2, pin_memory=True)
    mem = model.dsct.memory
    known_map = {int(c): i for i, c in enumerate(mem.known_ids)}

    groups = {"known": [], "pseudo_novel": [], "unlabeled": []}
    n_batches_seen = 0
    with torch.no_grad():
        for data_dict in loader:
            if n_batches_seen >= args.n_batches:
                break
            n_batches_seen += 1
            for key in ("filename", "info", "file_path"):
                data_dict.pop(key, None)
            data_dict = data_dict_to_cuda(
                data_dict, device=model.text_embeddings.device)
            outputs = model(data_dict)
            held = set(criterion.held_out)
            gts = data_dict["gt_instances"]
            for ti, gt in zip(outputs["track_instances"], gts):
                if len(ti) == 0:
                    continue
                matched = ti.matched_gt_idxes >= 0
                if not matched.any():
                    continue
                m_gt = ti.matched_gt_idxes.clamp(min=0)
                labels = gt.labels
                for qi in matched.nonzero(as_tuple=False).flatten().tolist():
                    c = int(labels[m_gt[qi]].item())
                    sem = torch.nn.functional.normalize(
                        ti.sem_state[qi].reshape(1, -1), dim=-1)
                    l3, kl, nl, feats = mem.forward_decision(
                        sem, ti.ow_obj_logit[qi],
                        int(ti.hit_count[qi]) + 1)
                    row = {
                        "feats": feats[0].tolist(),
                        "logits3": l3[0].tolist(),
                        "max_known_sim": float(kl.max()),
                    }
                    if c in held:
                        groups["pseudo_novel"].append(row)
                    elif c in known_map:
                        groups["known"].append(row)
                    else:
                        groups["unlabeled"].append(row)

    def summarize(rows):
        if not rows:
            return {"n": 0}
        import statistics
        f = [r["feats"] for r in rows]
        l = [r["logits3"] for r in rows]
        return {
            "n": len(rows),
            "feats_mean": [round(float(statistics.mean(x)), 3)
                           for x in zip(*f)],
            "logits3_mean": [round(float(statistics.mean(x)), 3)
                             for x in zip(*l)],
            "logits3_max": [round(float(max(x)), 3) for x in zip(*l)],
            "known_frac_argmax": round(
                sum(1 for r in l if r.index(max(r)) == 0) / len(l), 3),
            "new_frac_argmax": round(
                sum(1 for r in l if r.index(max(r)) == 2) / len(l), 3),
            "max_known_sim_mean": round(statistics.mean(
                r["max_known_sim"] for r in rows), 3),
        }

    result = {
        "n_batches": n_batches_seen,
        "known": summarize(groups["known"]),
        "pseudo_novel": summarize(groups["pseudo_novel"]),
        "unlabeled": summarize(groups["unlabeled"]),
        "novel_memory_size_after": mem.size,
    }
    out = ROOT / "outputs/iclr27_phase6b/audit/decision_diagnostic.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
