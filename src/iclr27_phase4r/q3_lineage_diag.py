#!/usr/bin/env python3
"""Bounded inference diagnostic for the Phase 4R Q3 failure analysis.

Runs the OVTR inference path on a capped number of validation frames and
reports per-frame proposal counts, prior-hit distribution, and online
existence-belief statistics, without writing the full TETA dump.

Run from third_party/research_refs_phase4n/OVTR/ovtr with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from eval import OVTR_inference
from main import get_args_parser
from models import build_model
from util.tool import load_model
from datasets import build_dataset
from util import misc as utils
from datasets.data_prefetcher import data_dict_to_cuda
from util.slconfig import SLConfig


def main():
    ap = argparse.ArgumentParser(parents=[get_args_parser()])
    ap.add_argument("--diag-ckpt", required=True)
    ap.add_argument("--diag-out", required=True)
    ap.add_argument("--max-frames", type=int, default=800)
    args = ap.parse_args()

    args.config_file = "./config/ovtr_lite_train_val.py"
    args.dataset_file = "lvis_generated_img_seqs"
    args.with_box_refine = True
    args.two_stage = True
    args.batch_size = 1
    args.num_workers = 4
    args.sampler_lengths = [2]
    args.device = "cuda"
    args.output_dir = "/tmp/q3_lineage_diag"
    args.result_path_track = "/tmp/q3_lineage_diag/teta"
    args.eval = "track"
    args.eval_options = None
    args.vis = False
    args.vis_output = "/tmp/q3_lineage_diag/vis"
    args.area_threshold = 1
    args.seed = 42

    cfg = SLConfig.fromfile(args.config_file)
    cfg.data.test.test_mode = True
    cfg.device = "cuda"
    model, _ = build_model(args, cfg)
    print("number of params:", sum(p.numel() for p in model.parameters()))
    model = load_model(model, args.diag_ckpt)
    model.eval()
    model = model.cuda()

    dataset = build_dataset(image_set="val", args=args, cfg=cfg.data.test)
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, sampler=sampler, drop_last=False,
        collate_fn=utils.mot_collate_fn, num_workers=args.num_workers,
        pin_memory=True)
    tracker = OVTR_inference(args, cfg, model=model)

    per_frame_proposals = []
    prior_hits = Counter()
    emitted_e = []
    emitted_score = []
    suppressed_e = []
    track_instances = None
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(loader, total=args.max_frames)):
            if i >= args.max_frames:
                break
            info = data_dict.pop("info")[0]
            file_path = data_dict.pop("file_path")[0]
            data_dict = data_dict_to_cuda(
                data_dict, device=model.text_embeddings.device)
            track_instances = tracker.detect(
                vis=False, data=data_dict, track_instances=track_instances,
                info=info, prob_threshold=args.score_thresh,
                score_threshold=args.score_thresh,
                filter_score_thresh=args.filter_score_thresh,
                miss_tolerance=args.miss_tolerance,
                maximum_quantity=args.maximum_quantity, area_threshold=1,
                ious_thresh=args.ious_thresh, file_path=file_path)
            dt = track_instances.to(torch.device("cpu"))
            # Mirror EvalTracker.filter_dt_by_score + update emission gate:
            # a proposal is emitted iff score above threshold, disappear_time
            # zero, and a valid class label was assigned (fresh det queries
            # keep cls_idxes == -1 until processed and must not be counted).
            keep = (
                (dt.scores > args.score_thresh[0])
                & (dt.disappear_time == 0)
                & (dt.cls_idxes != -1)
            )
            per_frame_proposals.append(int(keep.sum()))
            if keep.any():
                prior_hits.update(
                    int(x) for x in dt.hit_count[keep].tolist())
                emitted_score.extend(
                    float(x) for x in dt.scores[keep].tolist())
                if hasattr(dt, "e_valid_logit"):
                    emitted_e.extend(
                        float(x) for x in dt.e_valid_logit[keep].tolist())
            if hasattr(dt, "e_valid_logit") and hasattr(dt, "obj_idxes"):
                alive = dt.obj_idxes >= 0
                suppressed = alive & ~keep
                if suppressed.any():
                    suppressed_e.extend(
                        float(x) for x in dt.e_valid_logit[suppressed].tolist())

    out = {
        "frames": len(per_frame_proposals),
        "proposals": int(np.sum(per_frame_proposals)),
        "proposals_per_frame": float(np.mean(per_frame_proposals)),
        "prior_hits_top": dict(sorted(prior_hits.items(), reverse=True)[:12]),
        "prior_hits_zero_frac": (
            prior_hits.get(0, 0) / max(1, sum(prior_hits.values()))),
        "emitted_score_mean": float(np.mean(emitted_score)),
        "emitted_e_mean": float(np.mean(emitted_e)),
        "emitted_e_gt_keep_frac": float(
            np.mean(np.asarray(emitted_e) >= 0.5))
        if emitted_e else None,
        "suppressed_count": len(suppressed_e),
        "suppressed_e_mean": float(np.mean(suppressed_e))
        if suppressed_e else None,
    }
    import json as _json
    from pathlib import Path
    Path(args.diag_out).write_text(_json.dumps(out, indent=2))
    print(_json.dumps(out, indent=2))
    print("Q3_LINEAGE_DIAG_DONE")


if __name__ == "__main__":
    main()
