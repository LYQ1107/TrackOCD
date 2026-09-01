#!/usr/bin/env python3
"""Phase71 Q0-preserving TCO fold wrapper.

This imports the pinned OVTR trainer read-only.  Runtime filtering applies a
lightweight TRAIN policy manifest; no annotation or image data are copied.
The wrapper loads Q0 first and freezes every parameter except the new TCO
quality/lifecycle head.  Parent assignment, detector/query decoder and the
base score remain Q0 paths.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVTR = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"
sys.path.insert(0, str(OVTR))

FOLD = int(os.environ.get("PHASE71_FOLD", "0"))
MANIFEST = ROOT / "outputs/iclr27_phase71/manifests" / f"fold_{FOLD}_train.json"
if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)
POLICY = json.loads(MANIFEST.read_text())
ALLOWED_VIDEOS = set(int(x) for x in POLICY["allowed_videos"])
ALLOWED_CATEGORIES = set(int(x) for x in POLICY["allowed_categories"])

import datasets.lvis_seqs as _ls  # noqa: E402


def _install_filters(cls) -> None:
    orig_load = cls.load_lvis_track_anns
    orig_get = cls.get_lvis_ann_info

    def load(self, ann_file):
        infos = orig_load(self, ann_file)
        self.phase71_allowed_videos = ALLOWED_VIDEOS
        self.phase71_allowed_categories = ALLOWED_CATEGORIES
        kept = []
        # Avoid legacy get_min_wh(empty) crashes after category filtering.
        for info in infos:
            if int(info.get("video_id", -1)) not in ALLOWED_VIDEOS:
                continue
            ann_ids = getattr(self.lvis, "img_ann_map", {}).get(int(info.get("id", -1)), [])
            valid = False
            for item in ann_ids:
                ann = item if isinstance(item, dict) else self.lvis.anns.get(item, {})
                if int(ann.get("category_id", -1)) not in ALLOWED_CATEGORIES:
                    continue
                if ann.get("ignore", False) or ann.get("iscrowd", False):
                    continue
                box = ann.get("bbox", [0, 0, 0, 0])
                if len(box) >= 4 and float(box[2]) >= 1.0 and float(box[3]) >= 1.0 and float(ann.get("area", 0.0)) > 0:
                    valid = True
                    break
            if valid:
                kept.append(info)
        print("PHASE71_IMAGE_FILTER", json.dumps({"before": len(infos), "after": len(kept), "videos": len(ALLOWED_VIDEOS), "categories": len(ALLOWED_CATEGORIES)}, sort_keys=True), flush=True)
        return kept

    def get(self, img_info):
        ann = orig_get(self, img_info)
        allowed = {self.cat2label[c] for c in ALLOWED_CATEGORIES if c in self.cat2label}
        if len(ann.get("labels", [])) == 0:
            return ann
        keep = np.isin(np.asarray(ann["labels"]), list(allowed))
        for key in ("bboxes", "labels", "masks", "instance_ids"):
            if key not in ann:
                continue
            try:
                if len(ann[key]) == len(keep):
                    ann[key] = ann[key][keep]
            except (TypeError, IndexError):
                ann[key] = [v for v, q in zip(ann[key], keep) if q]
        return ann

    cls.load_lvis_track_anns = load
    cls.get_lvis_ann_info = get


_install_filters(_ls.LVIS_seqs_Dataset)

import main as _main  # noqa: E402

# Repair only the inherited one-based sampling boundary.  The embedding
# tensors have N columns indexed 0..N-1; this prevents an invalid random
# lookup without exposing category IDs to the TCO input.
try:
    import models.ovtr as _ovtr_mod  # noqa: E402

    _orig_init = _ovtr_mod.OVTR.__init__

    def _phase71_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        ncols = int(self.image_embeddings.shape[-1])
        self.all_ids = list(range(ncols))
        self.select_id = list(range(min(len(self.select_id), ncols)))

        # A strict physical adapter contract: only TCO is trainable.  This is
        # set before main() builds optimizer groups; subsequent config flags
        # are deliberately disabled in Phase71.
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("tco_head."))

    _ovtr_mod.OVTR.__init__ = _phase71_init
except Exception as exc:  # visible startup failure, never silently ignored
    print("PHASE71_ID_CONTRACT_PATCH_ERROR", repr(exc), flush=True)
    raise


if __name__ == "__main__":
    parser = _main.argparse.ArgumentParser("Phase71 Q0 TCO fold", parents=[_main.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("PHASE71_POLICY", json.dumps({
        "fold": FOLD,
        "manifest": str(MANIFEST.resolve()),
        "allowed_categories": len(ALLOWED_CATEGORIES),
        "allowed_videos": len(ALLOWED_VIDEOS),
        "initialization": str((ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth").resolve()),
        "trainable_contract": "tco_head_only",
        "base_score": "Q0 pred_logits.sigmoid().max retained",
        "parent_assignment": "frozen",
        "text_cross_attention": "Q0-compatible frozen decoder path; TCO input excludes text/category tensors",
        "semantic_labels_as_input": False,
    }, sort_keys=True), flush=True)
    # Some legal two-frame batches contain only age-0/new queries.  With all
    # Q0 parameters frozen, the native criterion then has no TCO mask and the
    # aggregate loss is a constant tensor; upstream engine.py blindly calls
    # backward() and aborts.  A process-local guard skips that optimizer step
    # (there is genuinely no adapter supervision) while preserving every
    # other batch and the registered sampling/denominator semantics.
    _orig_backward = torch.Tensor.backward
    _skipped = {"count": 0}

    def _phase71_safe_backward(self, *bargs, **bkwargs):
        if not bool(getattr(self, "requires_grad", False)):
            _skipped["count"] += 1
            if _skipped["count"] <= 5:
                print("PHASE71_SKIP_NO_TCO_GRAD_BATCH", _skipped["count"], flush=True)
            return None
        return _orig_backward(self, *bargs, **bkwargs)

    torch.Tensor.backward = _phase71_safe_backward
    try:
        _main.main(args)
    finally:
        torch.Tensor.backward = _orig_backward
        print("PHASE71_NO_GRAD_BATCHES", _skipped["count"], flush=True)
