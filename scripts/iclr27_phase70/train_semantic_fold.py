#!/usr/bin/env python3
"""Phase70 wrapper around the pinned OVTR trainer.

Runtime filtering and the legacy LVIS category-column repair are copied as a
small namespace-local adapter from Phase69.  The physical checkpoint remains
read-only input; no category or physical ID is passed as a model feature.
"""
from __future__ import annotations
import json, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVTR = ROOT / 'third_party/research_refs_phase4n/OVTR/ovtr'
sys.path.insert(0, str(OVTR))
FOLD = int(os.environ.get('PHASE70_FOLD', '0'))
MANIFEST = ROOT / 'outputs/iclr27_phase69/manifests' / f'fold_{FOLD}_train.json'
policy = json.loads(MANIFEST.read_text())
ALLOWED_VIDEOS = set(int(x) for x in policy['allowed_videos'])
ALLOWED_CATEGORIES = set(int(x) for x in policy['allowed_categories'])

import datasets.lvis_seqs as _ls  # noqa: E402

orig_load = _ls.LVIS_seqs_Dataset.load_lvis_track_anns
orig_get = _ls.LVIS_seqs_Dataset.get_lvis_ann_info

def load_filtered(self, ann_file):
    infos = orig_load(self, ann_file)
    self.phase70_allowed_videos = ALLOWED_VIDEOS
    self.phase70_allowed_categories = ALLOWED_CATEGORIES
    kept = []
    amap = getattr(self.lvis, 'img_ann_map', {})
    for x in infos:
        if int(x.get('video_id', -1)) not in ALLOWED_VIDEOS:
            continue
        valid = False
        for aid in amap.get(int(x.get('id', -1)), []):
            a = aid if isinstance(aid, dict) else self.lvis.anns.get(aid, {})
            if a.get('category_id') not in ALLOWED_CATEGORIES or a.get('ignore', False) or a.get('iscrowd', False):
                continue
            bx = a.get('bbox', [0, 0, 0, 0])
            if len(bx) >= 4 and float(bx[2]) >= 1.0 and float(bx[3]) >= 1.0 and float(a.get('area', 0.0)) > 0.0:
                valid = True; break
        if valid: kept.append(x)
    print('PHASE70_IMAGE_FILTER', json.dumps({'before': len(infos), 'after': len(kept), 'fold': FOLD, 'allowed_videos': len(ALLOWED_VIDEOS), 'allowed_categories': len(ALLOWED_CATEGORIES)}, sort_keys=True), flush=True)
    return kept

def get_filtered(self, img_info):
    ann = orig_get(self, img_info)
    allowed_labels = {self.cat2label[c] for c in ALLOWED_CATEGORIES if c in self.cat2label}
    if len(ann.get('labels', [])) == 0: return ann
    keep = np.isin(np.asarray(ann['labels']), list(allowed_labels))
    for k in ('bboxes', 'labels', 'masks', 'instance_ids'):
        if k in ann:
            try:
                if len(ann[k]) == len(keep): ann[k] = ann[k][keep]
            except (TypeError, IndexError):
                ann[k] = [v for v, q in zip(ann[k], keep) if q]
    return ann

_ls.LVIS_seqs_Dataset.load_lvis_track_anns = load_filtered
_ls.LVIS_seqs_Dataset.get_lvis_ann_info = get_filtered

import main as _main  # noqa: E402
try:
    import models.ovtr as _ovtr_mod  # noqa: E402
    _orig_init = _ovtr_mod.OVTR.__init__
    def _phase70_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        ncols = int(self.image_embeddings.shape[-1])
        self.all_ids = list(range(ncols))
        self.select_id = list(range(min(len(self.select_id), ncols)))
    _ovtr_mod.OVTR.__init__ = _phase70_init
except Exception as exc:
    print('PHASE70_ID_CONTRACT_PATCH_ERROR', repr(exc), flush=True)

if __name__ == '__main__':
    parser = _main.argparse.ArgumentParser('Phase70 OVTR semantic fold', parents=[_main.get_args_parser()])
    args = parser.parse_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('PHASE70_POLICY', json.dumps({'fold': FOLD, 'manifest': str(MANIFEST), 'semantic_labels_as_input': False, 'physical_id_as_feature': False, 'future_frames': False, 'text_cross_attention': False}, sort_keys=True), flush=True)
    # The pinned OVTR config keeps its CLIP tensor paths relative to the
    # upstream ``ovtr`` working directory.  Make that legacy path explicit so
    # Phase70 runs are reproducible regardless of the launching cwd.
    os.chdir(str(OVTR))
    _main.main(args)
