from copy import deepcopy
import pytest
from src.trackocd_cvpr.C_development_complete_matrix import (
    VARIANTS, SEEDS, PREFIXES, directory, cohort_sources, verified_matrix, aggregate_matrix)


def test_complete_public_cohort_preflight_missing_before_aggregation_or_private_data(tmp_path):
    assert len(cohort_sources()) == len(set(cohort_sources())) == 27
    assert 'registration_retry1' in directory('Top8_pool_context', 1028)
    with pytest.raises(FileNotFoundError): verified_matrix(tmp_path)


def test_all9_aggregate_uses_per_seed_then_order_mean_not_selected_winner_or_fake9seeds():
    table = {}
    for v in VARIANTS:
        for s in SEEDS:
            orders = {}
            for o in SEEDS:
                fields = {'Known_micro_accuracy': (s - 1026) / 3, 'Known_macro_accuracy': .5,
                    'all_physical_commit_track_coverage': 1., 'declared_pseudo_absorbed_as_Known_queries': 119,
                    'declared_pseudo_anonymous_commit_queries': 0,
                    'certified_cross_video_reuse_micro_lower_bound': 0., 'certified_cross_video_reuse_macro_lower_bound': 0.}
                orders[str(o)] = {'prefix_snapshots': {p: fields for p in PREFIXES}}
            table[(v, s)] = {'orders': orders}
    result = aggregate_matrix(table)
    for row in result['models'].values():
        assert row['three_training_seed_mean']['all_available']['Known_micro_accuracy'] == pytest.approx(2 / 3)
        assert len(row['per_training_seed_order_mean']) == 3
    assert not result['selected_model_seed_order_or_prefix'] and not result['order_replays_are_independent_training_seeds']
    reduced = deepcopy(table); reduced.pop((VARIANTS[-1], SEEDS[-1]))
    with pytest.raises(ValueError, match='All9'): aggregate_matrix(reduced)
