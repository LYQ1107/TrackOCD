"""Public real-data support witness, not frozen protocol or model evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v6_global'
REGISTERED = 'b202b55615843c4166117487329ba7c358b787e1'


def test_actual_global_candidate_pool_and_verified_support_without_freeze():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['status'] == 'VERIFIED_GLOBAL_KNOWN_ASSIGNMENT_NOT_FROZEN_PROTOCOL'
    assert public['feasible_joint_assignment_verified'] and public['solver_status'] == 0
    assert public['all_original_videos_retained'] == 429
    assert public['integer_necessary_conditions']['class_pool'] == 23
    assert public['role_class_counts'] == dict(fit_known=5, policy_train_pseudo=2,
        development_pseudo=8, independent_final_pseudo=8)
    assert public['role_video_counts'] == dict(fit_known=12, policy_train_pseudo=6,
        development_pseudo=19, independent_final_pseudo=17)
    # Candidate availability is NOT the claim all429 were assigned a role.
    assert sum(public['role_video_counts'].values()) == 54 < 429
    assert not public['new_protocol_frozen'] and not public['complete_visual_category_absence_proven']
    assert result['optimization_training_steps'] == 0
    assert result['preregistration_commit'] == REGISTERED
    for path, expected in result['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f'{REGISTERED}:{path}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    assert result['live_remote_registration_proof']['sha'] == REGISTERED
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['owned_worker_exit_code'] == 0 and receipt['successful_owned_execution']
    assert receipt['guard_stop_reason'] is None and receipt['peak_sampled_worker_threads'] == 1
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
    assert receipt['wall_seconds_including_imports_network_and_solver'] <= 120
