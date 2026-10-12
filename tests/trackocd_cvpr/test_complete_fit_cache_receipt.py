import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1'


def test_real_complete_fit_partition_keeps_full_denominator_and_reuses_valid_pilot():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == 'c45ffc1e199ab558c4ee6b943b0f91c4c25ffc28e125f67adffa85267519e1f6'
    r = json.loads(p.read_text()); shards = r['shards']
    assert r['status'] == 'VERIFIED_FIT_PARTITION_UNCAPPED_GT_CONTROL_FEATURE_CACHE_NOT_PRED_OR_TRAINED'
    assert r['live_remote_registration_proof']['sha'] == '01e7383cf6e66ab225dc0b1d73604765bdd9110c'
    assert (r['videos'], r['tracks'], r['observations'], r['metadata_frames']) == (257, 1401, 28326, 9345)
    assert len(shards) == 257 and r['reused_video_shards_without_extraction'] == 2
    assert sum(s['empty_GT_frames'] for s in shards) == 580
    assert sum(s['every_visible_prefix']['every_visible_prefix_checked'] for s in shards) == 28326
    assert sum(s['every_visible_prefix']['prefixes_with_actual_future_tail_mutation'] for s in shards) == 26925
    assert all(s['reopen_exact'] for s in shards)
    assert all(shards[i]['reused_without_extraction'] for i in (0, 1))
    assert not r['development_final_or_protected_pixels_read']
    assert not r['optimizer_or_formal_training_started'] and not r['full407_cache_automatic_launch']
    assert not r['qualified_Pred_or_Goal_A_B_C_achieved']
    assert r['resources']['training_GPU_hours'] == 0
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None
