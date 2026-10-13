#!/usr/bin/env python3
"""Fresh-registered PUBLIC all-nine original C matrix, never fit or select."""
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
from src.trackocd_cvpr.C_development_complete_matrix import cohort_sources, EXTRA_SOURCES, verified_matrix, aggregate_matrix
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

CONFIG = 'configs/trackocd_cvpr/C_development_complete_matrix_v1.json'
RAW_DIRECTORIES = ('outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_transport_retry1_Raw_Top1',
                   'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Raw_Top8')
RAW_RESULT_HASHES = ('e63831f1f6485ced2f09c164099eafc1674f589c7b322b05503366788d454b89',
                     'e6fae9929d29358da18005fe52833c96de3a8abe61f50d8160665fcba9d99320')
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/aggregate_C_development.py',
    'src/trackocd_cvpr/C_development_complete_matrix.py', 'src/trackocd_cvpr/C_terminal_counter_qualification.py',
    'configs/trackocd_cvpr/C_GT_causal_development_v1.json', 'configs/trackocd_cvpr/autonomous_research.json',
    'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    args = p.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if (cfg['status'] != 'REGISTERED_PUBLIC_ALL9_ORIGINAL_CAUSAL_DEVELOPMENT_AGGREGATE_REQUIRED'
            or cfg['full_scientific_Goal_acceptance'] or cfg['actual_all9_aggregate_execution_or_repair_training_from_manifest']):
        raise ValueError('Exact public original matrix/no scientific acceptance from manifest')
    raw_sources = tuple(path for d in RAW_DIRECTORIES for path in (d + '/result.json', d + '_supervisor_attempt_1.json'))
    sources = (*SOURCES, *cohort_sources(), *EXTRA_SOURCES, *raw_sources)
    for path in sources:
        if not (ROOT / path).is_file(): raise FileNotFoundError(ROOT / path)
        if (ROOT / path).is_symlink(): raise ValueError('No redirected public inputs')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in sources:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual public source/complete matrix not registered: ' + path)
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('Preserve previous aggregate')
    started = time.monotonic()
    def guard():
        rss = process_memory()['guarded_process_RSS_bytes']
        if resources_ok(cfg, safety, ROOT, started, rss): raise RuntimeError('Owned public aggregate resource guard')
        return rss
    def barrier(event, values):
        if event == 'socket.connect': raise PermissionError('No network after registration')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or path.suffix in ('.npz', '.pt', '.pth') or path.name.startswith('private_'):
                raise PermissionError('PUBLIC records only, never GT/cache/pixels/weights')
    sys.addaudithook(barrier); peak = guard(); table = verified_matrix(ROOT)
    aggregate = aggregate_matrix(table); raw = {}
    for d, expected in zip(RAW_DIRECTORIES, RAW_RESULT_HASHES):
        if sha256_file(ROOT / d / 'result.json') != expected: raise ValueError('Exact original Raw control result')
        r = json.loads((ROOT / d / 'result.json').read_text()); t = json.loads((ROOT / (d + '_supervisor_attempt_1.json')).read_text())
        if (not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None
                or r['training_seed'] is not None or r['optimizer_steps'] != 0 or r['model_or_checkpoint_selection']):
            raise ValueError('Actual complete zero-fit Raw controls, not trained seeds')
        raw[r['model']] = r['orders']
    result = {'status': 'ACTUAL_PUBLIC_ALL9_C_DEVELOPMENT_MATRIX_NOT_NEWLOSS_PRED_CT_OR_GOAL_ACCEPTANCE',
        'live_remote_registration_proof': proof, 'source_sha256': {path: sha256_file(ROOT / path) for path in sources},
        'all_original_trials': {v + '_s' + str(s): r['orders'] for (v, s), r in table.items()},
        'aggregate': aggregate, 'zero_fit_Raw_controls': raw,
        'GT_cache_pixels_weights_read_or_optimizer_or_model_selection': False,
        'resources': {'peak_guarded_RSS_bytes': max(peak, guard()), 'wall_seconds_excluding_registration': time.monotonic() - started}}
    data_size = len(json.dumps(result).encode('utf-8'))
    if data_size > cfg['output_bytes_ceiling']: raise RuntimeError('Whole public aggregate size bound')
    out.mkdir(parents=True); atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'complete_C_trials': len(table), 'public_only': True}), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
