#!/usr/bin/env python3
"""T0 metadata-only audit and independent Train-first main scope registration."""
from __future__ import annotations
import json
import sys
import time
import resource
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.audit_assets import DATASET, read_annotation
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.train_first_split import known_tracks, split_tracks, resource_statistics


def main():
    started = time.monotonic()
    cp = ROOT/'configs/trackocd_core/training_split.json'; config = json.loads(cp.read_text())
    out = ROOT/config['output_directory']
    if out.exists(): raise ValueError('Preserve existing Train-first plan; no overwrite/resampling')
    ann = Path(config['train_annotation'])
    if sha256_file(ann) != config['train_annotation_sha256']: raise ValueError('Pinned Train annotation changed')
    roles = ROOT/config['roles']; known = set(json.loads(roles.read_text())['known_ids'])
    tracks = known_tracks(read_annotation(ann, 'train'), known, config['maximum_observations'])
    rows, summary = split_tracks(tracks, known, config)
    for row in rows:
        for obs in row['observations']:
            p = DATASET/'frames'/obs['image_path']
            if not p.resolve().is_relative_to((DATASET/'frames/train').resolve()) or not p.is_file():
                raise ValueError('Missing or non-Train registered image')
    sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in
               (cp, roles, ROOT/config['visual_protocol'], ROOT/config['scope_amendment'],
                Path(__file__), ROOT/'src/trackocd_core/train_first_split.py')}
    plan = {'schema_version':'trackocd.core.train-first-private-plan.v1', 'annotation_sha256':sha256_file(ann),
            'sources':sources, 'summary':summary, 'rows':rows}
    atomic_json(out/'selection_plan.json', plan)
    public = {'schema_version':'trackocd.core.train-only-category-split-audit.v1',
              'status':'ACTIVE_CORE_TRAINING_WITH_LIMITED_FRONTEND', 'sources':sources,
              'annotation':{'bytes':ann.stat().st_size,'sha256':sha256_file(ann)},
              'statistics':resource_statistics(tracks, known), 'split':summary,
              'private_plan':{'bytes':(out/'selection_plan.json').stat().st_size,'sha256':sha256_file(out/'selection_plan.json')},
              'training_or_inference_started':False,'val_or_test_access':False,
              'historical_M1_status':'BLOCKED_FRONTEND_QUALITY',
              'resources':{'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    atomic_json(ROOT/'outputs/trackocd_core/TRAIN_ONLY_CATEGORY_SPLIT_AUDIT.json', public)
    episodes = {'schema_version':'trackocd.core.train-only-episode-manifest.v1', 'split_audit_sha256':sha256_file(ROOT/'outputs/trackocd_core/TRAIN_ONLY_CATEGORY_SPLIT_AUDIT.json'),
                'roles':summary['categories'], 'global_video_disjointness':True,'category_disjointness':True,
                'partitions':summary['partitions'], 'prefix_caps':[1,2,4,8,16],
                'short_prefix_rule':'actual=min(cap,available); no p16-only filter or future best-frame selection',
                'known_ids_for_episode':'representation fitting category IDs only; pseudo-category IDs excluded from episode prototypes',
                'main_val_known_ids':'restore inherited78 IDs; only48 have legal Train prototypes; missing30 remain in GT denominator',
                'inference_state':'model predicted actions only; GT labels never repair memory',
                'final_heldout_policy':'no method redesign from final heldout results',
                'orders':['main','seed1027','seed1028','seed1029'], 'seeds':[1027,1028,1029]}
    atomic_json(ROOT/'outputs/trackocd_core/TRAIN_ONLY_EPISODE_MANIFEST.json', episodes)
    print(json.dumps({'status':public['status'],'summary':summary,'resources':public['resources']}))

if __name__=='__main__': main()
