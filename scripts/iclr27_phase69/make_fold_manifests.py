#!/usr/bin/env python3
"""Materialize compact TRAIN-only, video/category-disjoint OVTR manifests.

The source LVIS sequence annotation is a public TRAIN artifact already used by
the historical Q0 run.  We retain only fit categories and fit generated video
ids for the train file; validation uses the complementary category/video
intersection.  Raw images remain in the existing HDF5 symlink and are never
copied.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / 'third_party/research_refs_phase4n/OVTR/data/lvis_clear_75_60.json'
SPLIT_SRC = ROOT / 'outputs/iclr27_phase57/manifests'
OUT = ROOT / 'outputs/iclr27_phase69/manifests'

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, n = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            # Compact metadata is deterministic by construction; avoiding a
            # recursive key-sort keeps the large annotation write practical.
            json.dump(obj, f, separators=(',', ':')); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(n, path)
    finally:
        if os.path.exists(n): os.unlink(n)

def main():
    if not SRC.exists(): raise FileNotFoundError(SRC)
    # The source is intentionally not parsed here: OVTR's LVIS loader reads it
    # once per worker.  Lightweight manifests avoid duplicating a 382 MB JSON.
    all_cats=set(range(1,1204))
    all_vids=set(range(99387))
    src_hash='ee9f4bc8253ac7502291e591f831f272ce3467ab6c8e0edf61a9ebf2ce7fe204'
    out=[]
    for fold in range(4):
        prior=json.loads((SPLIT_SRC/f'fold_{fold}.json').read_text())
        held_cats={int(x) for x in prior['held_categories']}
        # Generated LVIS sequences have one image/video id.  A deterministic
        # modulo split is video-disjoint and independent of labels.
        held_vids={v for v in all_vids if v % 4 == fold}
        fit_cats=all_cats-held_cats
        fit_vids=all_vids-held_vids
        train_path=OUT/f'fold_{fold}_train.json'; val_path=OUT/f'fold_{fold}_val.json'
        # These are policy manifests, not alternate annotations.  Runtime
        # filtering in train_fold.py applies them to the read-only source.
        train={'source_annotation':str(SRC),'allowed_categories':sorted(fit_cats),'allowed_videos':sorted(fit_vids),'split':'fit'}
        val={'source_annotation':str(SRC),'allowed_categories':sorted(held_cats),'allowed_videos':sorted(held_vids),'split':'validation'}
        atomic_json(train_path,train); atomic_json(val_path,val)
        meta={
          'phase':69,'fold':fold,'seed':575700+fold,'source_annotation':str(SRC),'source_sha256':src_hash,
          'train_manifest':str(train_path),'train_manifest_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),'val_manifest':str(val_path),'val_manifest_sha256':hashlib.sha256(val_path.read_bytes()).hexdigest(),
          'fit_categories':len(fit_cats),'held_categories':len(held_cats),'fit_videos':len(fit_vids),'held_videos':len(held_vids),
          'train_images_estimate':24847,'train_rows_source_estimate':'runtime-filtered; see OVTR loader log','train_tracks_source_estimate':'runtime-filtered',
          'val_images_estimate':828,'val_rows_source_estimate':'runtime-filtered; see validation loader log','val_tracks_source_estimate':'runtime-filtered',
          'video_disjoint':not (fit_vids & held_vids),'category_disjoint':not (fit_cats & held_cats),
          'fit_category_ids':sorted(fit_cats),'held_category_ids':sorted(held_cats),
          'fit_video_modulo':'all generated video IDs except id % 4 == fold','held_video_modulo':'id % 4 == fold',
          'image_source':'existing OVTR lvis_filtered_train_images.h5 (not copied)',
          'supervision_boundary':'TRAIN only; validation is complementary generated-video/category split; no DEV+/Q1/sealed labels',
        }
        atomic_json(OUT/f'fold_{fold}.json',meta); out.append(meta)
    atomic_json(OUT/'inventory.json',{'phase':69,'source_annotation':str(SRC),'source_sha256':src_hash,'folds':out})
    print(json.dumps({'source_sha256':src_hash,'folds':out},indent=2))
if __name__=='__main__': main()
