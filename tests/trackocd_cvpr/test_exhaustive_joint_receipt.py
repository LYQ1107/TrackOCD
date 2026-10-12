"""Actual complete necessary-support negative receipt, not trained success."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v5_exhaustive'


def test_actual_exhaustive_negative_coverage_and_terminal_resource_receipt():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['status'] == 'EXHAUSTIVE_NECESSARY_SUPPORT_CONTRADICTION_FOR_DECLARED_FAMILY'
    assert public['enumerated_component_placements'] == public['expected_placements'] == 16384
    assert public['legal_class_partitions'] == 3402
    assert public['necessary_support_survivors'] == 0
    assert sum(public['first_rejecting_fold_partition_counts']) == public['legal_class_partitions']
    assert public['policy_options_examined'] == 270230
    assert public['first_deficient_role_option_counts'] == {
        'fit_known': 262483, 'policy_train_pseudo': 5205,
        'development_pseudo': 355, 'independent_final_pseudo': 66}
    assert public['all_original_videos_retained'] == 429
    assert not public['solver_executed'] and not public['new_protocol_frozen']
    assert not public['complete_visual_category_absence_proven']
    assert result['optimization_training_steps'] == 0
    assert result['live_remote_registration_proof']['sha'] == 'cd880dd436f370a87e6bad213d9d7f644ffbd68a'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['owned_worker_exit_code'] == 0 and receipt['guard_stop_reason'] is None
    assert receipt['successful_owned_execution'] and receipt['peak_sampled_worker_threads'] == 1
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
    assert receipt['wall_seconds_including_imports_network_and_solver'] <= 120
