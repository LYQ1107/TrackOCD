import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry'


def test_policy_dev_complete_physical_denominators_and_no_final_or_semantic_targets():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == 'b6747cbc8d4802620cc56f2a2af25b6e172e664ab6903e79d04cda4ab2a9e0be'
    r = json.loads(p.read_text())
    assert r['status'] == 'VERIFIED_POLICY_DEV_UNCAPPED_GT_CONTROL_CACHE_NOT_FINAL_OR_PRED_OR_TRAINED'
    assert r['live_remote_registration_proof']['sha'] == 'd3eaf6035ac1deaa92582bf46fe6966f88e70032'
    assert (r['videos'], r['tracks'], r['observations'], r['metadata_frames']) == (82, 526, 10021, 2936)
    assert r['role_totals']['development_pseudo'] == dict(videos=72, tracks=472, observations=8753, metadata_frames=2562, empty_GT_frames=116)
    assert r['role_totals']['policy_train_pseudo'] == dict(videos=10, tracks=54, observations=1268, metadata_frames=374, empty_GT_frames=1)
    assert not r['independent_final_or_protected_pixels_read'] and not r['development_or_final_semantic_values_decoded']
    assert not r['optimizer_or_formal_training_started'] and not r['qualified_Pred_or_Goal_A_B_C_achieved']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
