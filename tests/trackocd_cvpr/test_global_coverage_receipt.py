"""Real Known-only coverage extension, not model or decoded-image evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v8_coverage_schema'


def test_actual_fixed_class_coverage_all_candidates_accounted_and_potential_queries():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['base_assigned_videos'] == 54 and public['expanded_assigned_videos'] == 407
    assert public['all_original_videos_retained_as_candidates'] == 429
    assert public['unassigned_ambiguous_positive_role_videos'] == 0
    assert public['unassigned_no_legal_positive_own_role_videos'] == 22
    assert public['added_unique_role_videos'] == dict(fit_known=245, policy_train_pseudo=4,
        development_pseudo=53, independent_final_pseudo=51)
    for role, classes, videos, tracks, observations in (
        ('fit_known', 5, 257, 1158, 22279), ('policy_train_pseudo', 2, 10, 10, 206),
        ('development_pseudo', 8, 72, 119, 2743), ('independent_final_pseudo', 8, 68, 93, 2337)):
        support = public['expanded_support'][role]
        assert (support['classes'], support['assigned_videos'], support['tracks']['sum'],
                support['observations']['sum']) == (classes, videos, tracks, observations)
        assert support['possible_cross_video_query_tracks']['sum'] == tracks
    assert not public['class_assignment_changed'] and not public['new_protocol_frozen']
    assert public['possible_queries_are_annotated_track_potential_not_decoded_features_or_online_results']
    assert not public['complete_visual_category_absence_proven']
    assert result['optimization_training_steps'] == 0
    assert result['live_remote_registration_proof']['sha'] == '1c9e42b67908d9fb38d65d7c9bcf57b85eeadb28'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['successful_owned_execution'] and receipt['owned_worker_exit_code'] == 0
    assert receipt['peak_sampled_worker_threads'] == 1 and receipt['guard_stop_reason'] is None
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
