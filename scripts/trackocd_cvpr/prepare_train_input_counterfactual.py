#!/usr/bin/env python3
"""No-GPU metadata/asset preflight; private paths and public aggregate receipt."""
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.metadata_only import image_video_metadata


def train_image(frame_root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'train':
        raise ValueError('Only relative Train pixels')
    resolved = (frame_root / path).resolve()
    if not resolved.is_relative_to((frame_root / 'train').resolve()) or not resolved.is_file():
        raise ValueError('Missing or escaping Train image')
    return resolved


def main():
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.physical_qualification import metadata_universe, json_digest
    from src.trackocd_core.masa_native import native_config, verified_sources
    cfg_path = ROOT / 'configs/trackocd_cvpr/cvpr0_train_input_preflight.json'
    cfg = json.loads(cfg_path.read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve retained preflight')
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < mem['MemTotal'] * safety['system_ram_reserve_fraction']:
            raise RuntimeError('RAM reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']:
            raise RuntimeError('Process RSS ceiling')
        if time.monotonic() - started > cfg['wall_seconds_ceiling']:
            raise RuntimeError('Wall ceiling')
        if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes'] + cfg['planned_physical_and_features_bytes_ceiling']:
            raise RuntimeError('Projected disk reserve')

    guard()
    annotation = Path(cfg['train_annotation'])
    if annotation.name != 'train.json' or sha256_file(annotation) != cfg['train_annotation_sha256']:
        raise ValueError('Frozen permitted Train metadata source differs')
    metadata = image_video_metadata(annotation)
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Frozen reference feature manifest differs')
    cache = TrainFirstCache(cache_root)
    keys = set(pq.read_table(cache_root / 'train_labels.parquet', columns=['key'],
                            filters=[('partition', '=', cfg['partition'])]).column(0).to_pylist())
    rows = [r for r in cache.rows if r['key'] in keys]
    video_ids = {r['video_id'] for r in rows}
    if len(rows) != cfg['reference_tracks'] or sum(r['observation_count'] for r in rows) != cfg['reference_observations']:
        raise ValueError('Registered reference universe differs')
    selected = {'videos': [v for v in metadata['videos'] if v['id'] in video_ids],
                'images': [i for i in metadata['images'] if i['video_id'] in video_ids]}
    videos = metadata_universe(selected)
    if len(videos) != cfg['expected_videos'] or sum(len(v['images']) for v in videos) != cfg['expected_images']:
        raise ValueError('Metadata-only physical plan differs')
    total_bytes = 0
    for video in videos:
        for image in video['images']:
            guard()
            path = train_image(Path(cfg['frame_root']), image['image_path'])
            image.update(image_bytes=path.stat().st_size, image_sha256=sha256_file(path))
            total_bytes += image['image_bytes']
    if total_bytes != cfg['expected_local_image_bytes']:
        raise ValueError('Local image total differs')
    common_path = ROOT / 'configs/trackocd_core/common_features.json'
    common = json.loads(common_path.read_text())
    dino_path = ROOT / common['checkpoint']
    masa_path = ROOT / 'checkpoints/masa_sam_candidate/sam_vitb_masa.pth'
    if sha256_file(dino_path) != cfg['common_DINO_checkpoint_sha256'] or sha256_file(masa_path) != cfg['frontend_checkpoint_sha256']:
        raise ValueError('Existing frozen model bytes differ')
    upstream = verified_sources()
    model_config = native_config()  # AST literals only, no model or training.
    plan = {'schema_version': 'trackocd.cvpr0.private-train-image-plan.v1',
            'videos': videos, 'GT_or_category_fields': False,
            'cadence': 'Train annotation image metadata, not dense frame-online',
            'selection': cfg['selection']}
    out.mkdir(parents=True)
    atomic_json(out / 'private_image_plan.json', plan)
    result = {'status': 'COMPLETE_METADATA_ASSET_PREFLIGHT_NOT_PHYSICAL_SMOKE',
        'config_sha256': sha256_file(cfg_path),
        'source_sha256': {n: sha256_file(ROOT / n) for n in (
            'scripts/trackocd_cvpr/prepare_train_input_counterfactual.py', 'src/trackocd_cvpr/metadata_only.py')},
        'historical_reference_manifest_sha256': cfg['feature_manifest_sha256'],
        'private_plan': {'filename': 'private_image_plan.json', 'sha256': sha256_file(out / 'private_image_plan.json'),
                         'bytes': (out / 'private_image_plan.json').stat().st_size},
        'video_count': len(videos), 'image_count': cfg['expected_images'], 'existing_image_bytes': total_bytes,
        'GT_reference_tracks': len(rows), 'GT_cached_observations': cfg['reference_observations'],
        'DINO': {'bytes': dino_path.stat().st_size, 'sha256': sha256_file(dino_path)},
        'MASA': {'bytes': masa_path.stat().st_size, 'sha256': sha256_file(masa_path),
                 'native_model_config_sha256': json_digest(model_config), 'verified_upstream_files': len(upstream),
                 'checkpoint_release_stage_binding': 'INCOMPLETE_PUBLIC_ROUTE_ONLY', 'not_qualified_primary': True},
        'reuse_existing_environment': '/data3/liuyeqiang/.venvs/trackocd-masa-smoke',
        'planned_new_physical_and_feature_payload_ceiling_bytes': cfg['planned_physical_and_features_bytes_ceiling'],
        'new_images_weights_or_environments_downloaded': False, 'model_executed': False,
        'TAO_Val_or_Test_access': False, 'final_holdout_labels_read': False,
        'annotation_or_category_arrays_JSON_decoded': False,
        'interpretation': 'Truncated-GT-reference, annotated-cadence, limited-provenance diagnostic; not clean end-to-end main',
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'GPU_hours': 0,
            'disk_free_bytes': shutil.disk_usage(ROOT).free}}
    atomic_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
