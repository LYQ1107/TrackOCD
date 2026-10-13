#!/usr/bin/env python3
"""Fresh registered numeric-only actual-prefix qualification, no semantic GT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

CONFIG = 'configs/trackocd_cvpr/qualify_frozen_raw_statistics_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/qualify_frozen_raw_statistics.py',
    'configs/trackocd_cvpr/source_frozen_raw_statistics_cache_v1.json',
    'configs/trackocd_cvpr/C_GT_causal_development_v1.json', 'configs/trackocd_cvpr/autonomous_research.json',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json', 'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'src/trackocd_cvpr/raw_statistics_qualification.py', 'src/trackocd_cvpr/frozen_raw_statistics_cache.py',
    'src/trackocd_cvpr/committed_support_bridge.py', 'src/trackocd_cvpr/reliability_support_bridge.py',
    'src/trackocd_cvpr/differentiable_support_bank.py', 'src/trackocd_cvpr/member_reliability.py',
    'src/trackocd_cvpr/prefix_member_bank.py', 'src/trackocd_cvpr/GT_video_cache.py',
    'src/trackocd_cvpr/policy_training_data.py', 'src/trackocd_cvpr/causal_observation_router.py',
    'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_core/features.py',
    'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    args = p.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text())
    data = json.loads((ROOT / cfg['data_config']).read_text()); safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual numeric qualification source/cache/authority not registered')
    if (cfg['status'] != 'REGISTERED_NUMERIC_ONLY_RAW_STATISTICS_ALL_DEV_PREFIXES_REQUIRED'
            or json.loads((ROOT / cfg['approval']).read_text())['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['qualification_or_Goal_acceptance_from_manifest']):
        raise ValueError('Exact authorized numeric-only qualification')
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('Preserve previous qualification')
    import torch
    from src.trackocd_cvpr.raw_statistics_qualification import scoped_numeric_video_rows, qualify_numeric_prefixes
    torch.set_num_threads(1); started, peak = time.monotonic(), 0
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        if resources_ok(cfg, safety, ROOT, started, peak): raise RuntimeError('Numeric-only qualification resource guard')
    guard(); directory, _, _, selected = scoped_numeric_video_rows(ROOT, data)
    allowed = {(directory / ('private_video_' + str(row['ordinal']).zfill(3) + '.npz')).resolve() for row in selected}
    forbidden_known = (ROOT / data['private_Known_support']).resolve()
    def barrier(event, values):
        if event == 'socket.connect': raise PermissionError('No network after registration')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if (path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or path == forbidden_known
                    or path.suffix in ('.pt', '.pth') or (path.suffix == '.npz' and path not in allowed)):
                raise PermissionError('Only authorized72 geometry/Raw cache; no labels/weights/pixels/other roles')
    sys.addaudithook(barrier); out.mkdir(parents=True)
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof, 'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES}})
    try:
        numeric = qualify_numeric_prefixes(ROOT, data, guard); guard()
        if torch.cuda.is_initialized(): raise ValueError('CPU-only no encoder or model')
        result = {'status': 'ACTUAL_ALL_DEV_PREFIX_RAW_STATISTIC_BYTE_EQUALITY_NOT_FULL_SHADOW_INTEGRATION_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'numeric_qualification': numeric,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': False}}
        if len(json.dumps(result).encode()) > cfg['output_bytes_ceiling']: raise RuntimeError('Whole qualification output bound')
        atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True)
        return 0
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


if __name__ == '__main__': sys.exit(main())
