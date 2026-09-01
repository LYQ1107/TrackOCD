#!/usr/bin/env python3
"""Phase69 bounded OVTR adaptation entry point.

This wrapper imports the pinned local OVTR trainer without editing it.  It
filters the read-only LVIS sequence source at dataset access time using the
small fold policy manifest, so category/video disjointness is enforced without
copying the 382 MB annotation or HDF5 image store.
"""
from __future__ import annotations
import json, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVTR = ROOT / 'third_party/research_refs_phase4n/OVTR/ovtr'
sys.path.insert(0, str(OVTR))

FOLD = int(os.environ.get('PHASE69_FOLD', '0'))
MANIFEST = ROOT / 'outputs/iclr27_phase69/manifests' / f'fold_{FOLD}_train.json'
policy = json.loads(MANIFEST.read_text())
ALLOWED_VIDEOS = set(int(x) for x in policy['allowed_videos'])
ALLOWED_CATEGORIES = set(int(x) for x in policy['allowed_categories'])

import datasets.lvis_seqs as _ls  # noqa: E402

def _install_filters(cls):
    orig_load = cls.load_lvis_track_anns
    orig_get = cls.get_lvis_ann_info
    def load(self, ann_file):
        infos = orig_load(self, ann_file)
        self.phase69_allowed_videos = ALLOWED_VIDEOS
        self.phase69_allowed_categories = ALLOWED_CATEGORIES
        # Category filtering happens later in ``get``.  Drop images that have
        # no valid allowed-category GT up front; otherwise the legacy
        # augmentation path calls ``get_min_wh`` on an empty bbox array and
        # aborts a long run.  This is still TRAIN-only policy filtering and
        # does not alter the evaluator or any held data.
        kept = []
        for x in infos:
            if int(x.get('video_id', -1)) not in ALLOWED_VIDEOS:
                continue
            ann_ids = getattr(self.lvis, 'img_ann_map', {}).get(int(x.get('id', -1)), [])
            valid = False
            for aid in ann_ids:
                # LVIS versions differ: ``img_ann_map`` may contain ann IDs
                # or the annotation dictionaries themselves.
                a = aid if isinstance(aid, dict) else self.lvis.anns.get(aid, {})
                if a.get('category_id') not in ALLOWED_CATEGORIES:
                    continue
                if a.get('ignore', False) or a.get('iscrowd', False):
                    continue
                bx = a.get('bbox', [0, 0, 0, 0])
                if len(bx) >= 4 and float(bx[2]) >= 1.0 and float(bx[3]) >= 1.0 and float(a.get('area', 0.0)) > 0.0:
                    valid = True
                    break
            if valid:
                kept.append(x)
        print('PHASE69_IMAGE_FILTER', json.dumps({'before':len(infos), 'after':len(kept), 'allowed_videos':len(ALLOWED_VIDEOS), 'allowed_categories':len(ALLOWED_CATEGORIES)}, sort_keys=True), flush=True)
        return kept
    def get(self, img_info):
        ann = orig_get(self, img_info)
        allowed_labels = {self.cat2label[c] for c in ALLOWED_CATEGORIES if c in self.cat2label}
        if len(ann.get('labels', [])) == 0:
            return ann
        keep = np.isin(np.asarray(ann['labels']), list(allowed_labels))
        for k in ('bboxes', 'labels', 'masks', 'instance_ids'):
            if k in ann:
                try:
                    if len(ann[k]) == len(keep): ann[k] = ann[k][keep]
                except (TypeError, IndexError):
                    ann[k] = [v for v, q in zip(ann[k], keep) if q]
        return ann
    cls.load_lvis_track_anns = load
    cls.get_lvis_ann_info = get

_install_filters(_ls.LVIS_seqs_Dataset)

# Importing main after patching ensures its `build_dataset` sees the wrapped
# class.  All command-line semantics remain those of the pinned trainer.
import main as _main  # noqa: E402

# The pinned OVTR mirror predates the phase-69 wrapper and assumes category
# columns are one-based when constructing ``all_ids``.  LVIS_seqs_Dataset,
# however, maps category ids to zero-based ``cat2label`` values and the
# embedding tensors contain exactly N columns (0..N-1).  A random padding
# draw could therefore select column N and trigger a CUDA device-side assert
# after the first batch.  Keep the upstream model read-only and repair this
# boundary locally: valid embedding columns are the only IDs sampled by the
# class-agnostic training wrapper.  This does not expose category/ID features
# to the DSCT head; it only prevents an invalid legacy embedding lookup.
try:  # imported after ``main`` so the exact class used by build_model is patched
    import models.ovtr as _ovtr_mod  # noqa: E402
    _orig_ovtr_init = _ovtr_mod.OVTR.__init__
    def _phase69_ovtr_init(self, *args, **kwargs):
        _orig_ovtr_init(self, *args, **kwargs)
        ncols = int(self.image_embeddings.shape[-1])
        self.all_ids = list(range(ncols))
        self.select_id = list(range(min(len(self.select_id), ncols)))
    _ovtr_mod.OVTR.__init__ = _phase69_ovtr_init
except Exception as exc:  # pragma: no cover - retain a visible startup error
    print('PHASE69_ID_CONTRACT_PATCH_ERROR', repr(exc), flush=True)

if __name__ == '__main__':
    parser = _main.argparse.ArgumentParser('Phase69 OVTR fold', parents=[_main.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('PHASE69_POLICY', json.dumps({'fold':FOLD,'manifest':str(MANIFEST),'allowed_categories':len(ALLOWED_CATEGORIES),'allowed_videos':len(ALLOWED_VIDEOS),'text_cross_attention':'disabled_in_config','semantic_labels_as_input':False}, sort_keys=True), flush=True)
    _main.main(args)
