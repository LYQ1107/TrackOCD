import numpy as np
import pytest
from src.trackocd_cvpr.joint_known_protocol import build_joint, joint_feasibility, verify_assignment


def separate():
    return {c * 4 + k: {c} for c in range(8) for k in range(4)}


def test_exact_pattern_partition_retains_all_original_videos():
    full = build_joint(separate(), balanced=False)
    compact = build_joint(separate(), balanced=False, compress_patterns=True)
    assert len(full[4]) == 32 and len(compact[4]) == 8
    assert sorted(v for group in compact[4] for v in group) == list(range(32))
    assert full[0].shape[1] == 8 * 20 + 32 * 16
    assert compact[0].shape[1] == 8 * 20 + 8 * 16
    assert all(len(group) == 4 for group in compact[4])


def test_actual_compressed_solver_counts_expand_to_verified_disjoint_video_IDs():
    public, private = joint_feasibility(separate(), balanced=False, union_support=True,
        compress_patterns=True, time_limit=5, node_limit=100)
    assert public['feasible_joint_assignment_verified'] is True
    assert public['exact_presence_pattern_count_compression'] is True
    assert public['all_original_videos_retained'] == 32
    assert public['distinct_presence_patterns'] == 8
    assert verify_assignment(separate(), private['assignment'], balanced=False)


def test_binary_video_assignment_projects_to_counts_without_relaxing_constraints():
    # Hand-construct a legal witness vector for the uncompressed8class family,
    # then sum yvariables within patterns; all rows still satisfy EXACT quotas,
    # group sizes, role exclusions and capacities, not an LP relaxation.
    from tests.trackocd_cvpr.test_unbalanced_joint_protocol import assignment
    witness = assignment([[0, 1, 2], [3], [4, 5, 6], [7]])
    from src.trackocd_cvpr.known_census_protocol import ROLES
    for compressed in (False, True):
        matrix, lower, upper, pool, groups = build_joint(separate(), balanced=False, compress_patterns=compressed)
        vector = np.zeros(matrix.shape[1])
        n = len(pool)
        for g, members in enumerate(witness['groups']):
            for c in members:
                vector[pool.index(c) * 4 + g] = 1
        for f, fold in enumerate(witness['folds']):
            for r, role in enumerate(ROLES):
                for c in fold['classes'][role]:
                    vector[n * 4 + (pool.index(c) * 4 + f) * 4 + r] = 1
                for index, videos in enumerate(groups):
                    vector[n * 20 + (index * 4 + f) * 4 + r] = len(set(videos) & set(fold['video_ids'][role]))
        residual = matrix @ vector
        assert np.all(residual >= lower) and np.all(residual <= upper)


def test_shape_changes_do_not_group_different_positive_category_sets():
    presence = separate()
    presence[0] = {0, 1}
    _, _, _, _, groups = build_joint(presence, compress_patterns=True)
    assert any(group == [0] for group in groups)
    assert not any(0 in group and 1 in group for group in groups)


def test_v4_does_not_raise_memory_budget_or_change_v3_data_family():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    a = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v3_unbalanced_ref_transport.json').read_text())
    b = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v4_compressed.json').read_text())
    for field in ('expected', 'expected_eligible_pool', 'eligible_threshold', 'four_balanced_final_groups',
                  'union_support_closure', 'every_eligible_class_final_once', 'fit_videos_per_class',
                  'eval_videos_per_class', 'positive_conflict_rules_unchanged', 'rss_bytes_ceiling',
                  'solver_time_limit_seconds', 'solver_node_limit'):
        assert a[field] == b[field]
