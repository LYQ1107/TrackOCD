"""Strict Train plan and explicit cross-registration shard reuse, no copying."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from src.trackocd_core.physical_qualification import completed_video

IMAGE_KEYS = {'image_id', 'frame_index', 'image_path', 'image_bytes', 'image_sha256'}


def validate_plan(plan, cfg):
    if plan.get('GT_or_category_fields') is not False or len(plan['videos']) != cfg['videos']:
        raise ValueError('Registered image-only plan differs')
    ids, image_ids, count = set(), set(), 0
    for video in plan['videos']:
        if set(video) != {'video_id', 'images'} or video['video_id'] in ids or not video['images']:
            raise ValueError('Duplicate/empty/label-bearing video')
        ids.add(video['video_id'])
        last = -1
        for image in video['images']:
            relative = Path(image.get('image_path', ''))
            if (set(image) != IMAGE_KEYS or image['image_id'] in image_ids or image['frame_index'] <= last
                    or not relative.parts or relative.parts[0] != 'train' or relative.is_absolute() or '..' in relative.parts
                    or image['image_bytes'] <= 0 or len(image['image_sha256']) != 64):
                raise ValueError('Unsafe or label-bearing/nonchronological Train image plan')
            last = image['frame_index']
            image_ids.add(image['image_id'])
            count += 1
    if count != cfg['canonical_images']:
        raise ValueError('Image universe differs')
    return plan['videos']


def explicit_reuse(root, video, cfg):
    reuse = cfg['reuse']
    if video['video_id'] != reuse['video_id']:
        return None
    run = root / reuse['directory']
    source_config = root / reuse['config_path']
    if sha256_file(source_config) != reuse['config_sha256']:
        raise ValueError('Reused registration bytes differ')
    original = json.loads(source_config.read_text())
    result_path = run / 'result.json'
    if sha256_file(result_path) != reuse['result_sha256']:
        raise ValueError('Reused completed result changed')
    proof = json.loads(result_path.read_text())
    for field in ('private_plan_sha256', 'checkpoint_sha256', 'native_model_config_sha256', 'frame_root', 'environment_python'):
        if original[field] != cfg[field]:
            raise ValueError('Reuse is not identical plan/model/cadence/environment')
    if (proof['status'] != 'PASS_BOUNDED_TRAIN_PHYSICAL_SMOKE_NOT_MAIN_QUALIFICATION'
            or proof['changed_future_prefix_equal'] is not True or proof['frozen_state_unchanged'] is not True):
        raise ValueError('Incomplete source smoke')
    record = completed_video(run, video, reuse['config_sha256'])
    if record is None or record['npz_sha256'] != reuse['npz_sha256'] or record != proof['shard']:
        raise ValueError('Reused valid shard binding differs')
    return {'directory': reuse['directory'], 'record': record,
            'source_config_sha256': reuse['config_sha256'], 'reused_without_copy_or_forward': True}
