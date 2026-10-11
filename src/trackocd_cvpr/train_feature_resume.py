"""Explicit recovered feature references and deterministic metadata assignment."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache


def assignments(videos, reused_video, workers):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Bounded worker count')
    remaining = [v['video_id'] for v in videos if v['video_id'] != reused_video]
    if len(set(remaining)) != len(remaining):
        raise ValueError('Duplicate metadata video')
    return {str(i): remaining[i::workers] for i in range(workers)}


def recovered_feature(root, cfg, physical_record):
    reuse = cfg['reuse']
    source_config = root / reuse['original_interface_config']
    result_path = root / reuse['recovery_result']
    if sha256_file(source_config) != reuse['config_sha256'] or sha256_file(result_path) != reuse['recovery_result_sha256']:
        raise ValueError('Explicit source/recovery binding changed')
    original, recovered = json.loads(source_config.read_text()), json.loads(result_path.read_text())
    if (recovered['status'] != 'RECOVERED_VALID_ALL_ID_FEATURE_INPUT_AFTER_FAILED_FINAL_JSON_NOT_NORMAL_WORKER_SUCCESS'
            or recovered['original_worker_returncode'] != 1
            or recovered['all_original_source_and_current_pixel_metadata_geometry_quality_L2_reader_verified'] is not True):
        raise ValueError('Only explicitly CPU-validated recovered feature input')
    for field in ('common_config_sha256', 'maximum_observations', 'batch_size', 'frame_root', 'environment_python'):
        if cfg[field] != original[field]:
            raise ValueError('Not the identical frozen feature protocol/environment')
    if recovered['physical_npz_sha256'] != physical_record['npz_sha256'] or recovered['private_feature_payload']['sha256'] != reuse['feature_sha256']:
        raise ValueError('Wrong physical-to-feature source binding')
    payload = recovered['private_feature_payload']
    directory, filename = Path(payload['directory']), Path(payload['filename'])
    if directory.is_absolute() or '..' in directory.parts or filename.name != str(filename):
        raise ValueError('Unsafe feature reference')
    path = root / directory / filename
    if path.is_symlink() or path.stat().st_size != payload['bytes'] or sha256_file(path) != reuse['feature_sha256']:
        raise ValueError('Retained valid payload changed')
    reader = LimitedMasaVideoCache(path, reuse['video_id'])
    if len(reader.rows) != recovered['tracks'] or len(reader.data['visual']) != recovered['observations']:
        raise ValueError('Recovered complete feature universe differs')
    reader.close()
    return {'status': 'VALID_REUSED_AFTER_CPU_RECOVERY_ORIGINAL_FAILURE_PRESERVED',
        'video_id': reuse['video_id'], 'directory': str(directory), 'npz_filename': str(filename),
        'npz_sha256': reuse['feature_sha256'], 'npz_bytes': payload['bytes'],
        'tracks': recovered['tracks'], 'observations': recovered['observations'],
        'input_npz_sha256': physical_record['npz_sha256'], 'recovery_result_sha256': reuse['recovery_result_sha256'],
        'copied_or_reencoded': False, 'original_worker_failure_overwritten': False}
