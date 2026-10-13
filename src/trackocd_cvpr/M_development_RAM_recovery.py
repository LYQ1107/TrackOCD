"""One recorded terminal system-RAM stop, same full evaluation/new namespace.

No process management or waiting here. Existing supervisor still enforces25%
system RAM reserve/3GiB/3000s. Only actual -9/system_RAM_reserve/noresult trial
with unchanged public launch/source/protocol qualifies. No silent science retry.
"""
import json
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/M_development_system_RAM_retry1.json'
ORIGINAL = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_all_model_fingerprints_v1_relative_quarantine_s1027'
RETRY = ORIGINAL + '_RAM_retry1'
TERMINAL = ORIGINAL + '_supervisor_attempt_1.json'
TERMINAL_SHA = '5781a9a91c2c41398077299403e5c8169721f9430049c3445927251d8a2182a3'
LAUNCH_SHA = 'bef0495ef2356e8c7724e3053dae0ec5a17626563b66c3d042d9836ed5540685'
PREREG = '5ddaf135be385ef659abf678aeeaeed4d991d1af'
SOURCES = (CONFIG, TERMINAL, ORIGINAL + '/launch.json',
    'src/trackocd_cvpr/M_development_RAM_recovery.py',
    'scripts/trackocd_cvpr/evaluate_M_development_RAM_retry1.py')


def recovery_configuration(root, cfg, args):
    r = json.loads((root / CONFIG).read_text())
    expected = {'status':'REGISTERED_NAMESPACE_ONLY_SYSTEM_RAM_RECOVERY_PROTOCOL_UNCHANGED',
        'model':'relative_quarantine', 'seed':1027, 'original_directory':ORIGINAL,
        'output_directory':RETRY, 'original_terminal_sha256':TERMINAL_SHA,
        'original_launch_sha256':LAUNCH_SHA, 'original_preregistration_commit':PREREG,
        'old_namespace_overwrite_partial_metric_reuse_or_guard_relaxation':False,
        'optimizer_steps':0}
    if (r != expected or (args.model,args.seed) != ('relative_quarantine',1027)
            or cfg['output_directory'] != ORIGINAL):
        raise ValueError('Only exact resource-stopped full M replay, unchanged fixed endpoint/protocol')
    for p,digest in ((TERMINAL,TERMINAL_SHA),(ORIGINAL+'/launch.json',LAUNCH_SHA)):
        if (root / p).is_symlink() or sha256_file(root / p) != digest:
            raise ValueError('Exact preserved terminal and actual original worker launch')
    t = json.loads((root / TERMINAL).read_text())
    if (t['guard_stop_reason'] != 'system_RAM_reserve' or t['owned_worker_exit_code'] != -9
            or t['successful_owned_execution'] or t['complete_result_present']
            or not t['only_new_owned_Popen_worker_managed'] or t['training_GPU_hours'] != 0
            or t['peak_sampled_owned_worker_RSS_bytes'] > cfg['rss_bytes_ceiling']
            or t['wall_seconds_including_imports_and_network'] > cfg['wall_seconds_ceiling']):
        raise ValueError('Only recorded system-RAM failure, not model failure/partial metric salvage')
    if {p.name for p in (root / ORIGINAL).iterdir()} != {'launch.json'}:
        raise ValueError('Original has no progress/result/profile/partial metric to reuse')
    launch = json.loads((root / (ORIGINAL+'/launch.json')).read_text())
    if launch['live_remote_registration_proof']['sha'] != PREREG:
        raise ValueError('Exact original actually registered worker')
    for p in ('scripts/trackocd_cvpr/evaluate_M_development_all_model_fingerprints.py',
              'src/trackocd_cvpr/M_all_model_fingerprint_development_profile.py',
              'configs/trackocd_cvpr/M_GT_causal_development_all_model_fingerprints_v1.json',
              'src/trackocd_cvpr/qualified_M_fingerprint_cache_scope.py',
              'src/trackocd_cvpr/member_reliability.py',
              'src/trackocd_cvpr/versioned_reliability_memory.py',
              'configs/trackocd_cvpr/M_GT_causal_development_v1.json'):
        if launch['source_sha256'][p] != sha256_file(root / p):
            raise ValueError('Exact original evaluator/profile/numerical model/protocol required')
    out = dict(cfg); out['output_directory'] = RETRY
    return out
