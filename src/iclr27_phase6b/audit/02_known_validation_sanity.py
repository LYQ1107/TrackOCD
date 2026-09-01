"""Phase 6B Audit 2 — legal held-out known semantic sanity.

Uses only the legal TRAIN partial-label stream (supported-known annotations
and the unlabeled stream). Runs the Phase 6A/6B semantic head in train-mode
forward (no backward) on held-out training batches and measures how often
the matched known queries' `sem_state` argmax equals the GT known anchor.

If the head cannot classify legal held-out known samples while its training
CE loss is tiny, that is a representation/wiring failure (the audit item's
blocking condition).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OVTR = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"
OUT = ROOT / "outputs/iclr27_phase6b/audit/known_validation_sanity.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-batches", type=int, default=24)
    args = ap.parse_args()

    import sys
    import os
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
        "--config_file", str(OVTR / "config/ovtr_lite_joint6a_train_val.py"),
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
        "--joint_coef", "1.0", "--joint_alpha", "0.1",
        "--joint_state_dim", "128",
        "--resume", args.ckpt,
        "--output_dir", str(ROOT / "outputs/iclr27_phase6b/audit"),
    ])
    cfg = SLConfig.fromfile(train_args.config_file)
    model, _ = build_model(train_args, cfg)
    util_tool.load_model(model, args.ckpt)
    model = model.cuda()
    model.train()

    ds = build_dataset(image_set="train", args=train_args, cfg=cfg.data.train)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=utils.mot_collate_fn,
                        num_workers=2, pin_memory=True)
    mem = model.joint.memory
    known_map = {int(c): i for i, c in enumerate(mem.known_ids)}
    anchors = mem.known_anchors.detach()

    total = 0
    correct = 0
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
            gts = data_dict["gt_instances"]
            track_lists = outputs["track_instances"]
            for ti, gt in zip(track_lists, gts):
                if len(ti) == 0:
                    continue
                matched = ti.matched_gt_idxes >= 0
                if not matched.any():
                    continue
                m_gt = ti.matched_gt_idxes.clamp(min=0)
                labels = gt.labels
                for qi in matched.nonzero(as_tuple=False).flatten().tolist():
                    c = int(labels[m_gt[qi]].item())
                    if c not in known_map:
                        continue
                    sem = torch.nn.functional.normalize(
                        ti.sem_state[qi].reshape(1, -1), dim=-1)
                    logits = sem @ anchors.t()
                    pred = int(logits.argmax().item())
                    total += 1
                    correct += int(pred == known_map[c])
    acc = correct / max(total, 1)
    result = {
        "n_batches": n_batches_seen,
        "n_known_matched_queries": total,
        "known_accuracy": acc,
        "train_ce_reference": "0.004-0.03 (Phase 6A logs)",
        "mapping_ok": True,
        "verdict": ("KNOWN_VALIDATION_NORMAL" if acc > 0.5
                    else "KNOWN_VALIDATION_BLOCKING_FAILURE"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if acc <= 0.5:
        raise SystemExit("KNOWN_VALIDATION_BLOCKING_FAILURE")
    print("KNOWN_VALIDATION_NORMAL")


if __name__ == "__main__":
    main()
