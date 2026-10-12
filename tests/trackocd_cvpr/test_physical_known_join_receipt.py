"""Actual role physical denominators; no inferred unknown semantic classes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v11_physical_join'


def test_actual_join_Known_integrity_and_complete_annotated_physical_roles():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['retained_Known_tracks_joined'] == 1879
    assert public['physical_tracks'] == 2222 and public['physical_observations'] == 45450
    assert public['additional_geometry_only_tracks'] == 343
    assert public['Known_count_mismatches'] == public['Known_keys_missing_in_physical_GT'] == 0
    for role, tracks, observations in (('fit_known', 1401, 28326), ('policy_train_pseudo', 54, 1268),
                                       ('development_pseudo', 472, 8753), ('independent_final_pseudo', 295, 7103)):
        assert public['role_denominators'][role]['all_annotated_physical_GT'] == dict(tracks=tracks, observations=observations)
    assert not public['new_nonKnown_semantic_labels_read_or_inferred']
    assert not public['geometry_only_tracks_declared_background_or_labeled_Novel']
    assert not public['pixel_feature_availability_proven'] and not public['new_protocol_frozen']
    assert public['annotated_physical_GT_counts_not_dense_or_semantic_Novel_truth']
    assert result['optimization_training_steps'] == 0
    assert result['live_remote_registration_proof']['sha'] == 'c083ecd2d9461b978a1d379ba0cc857878bf367f'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['owned_worker_exit_code'] == 0 and receipt['guard_stop_reason'] is None
    assert receipt['peak_sampled_worker_threads'] == 1 and receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
