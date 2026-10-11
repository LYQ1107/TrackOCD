"""Synthetic metadata constraints, no dataset labels or new training."""
from collections import Counter
import pytest
from src.trackocd_cvpr.train_fold_feasibility import cyclic_role_feasibility


def row(key, category, video):
    return {'key': key, 'category_id': category, 'video_id': video,
            'partition': 'representation_fit', 'simulation_role': 'known'}


def isolated_rows():
    return [row(f'{category}:{video}', category, category * 10 + video)
            for category in range(15) for video in range(7)]


def test_fixed_cyclic_final_categories_cover_each_once_with_disjoint_roles():
    result = cyclic_role_feasibility(isolated_rows())
    final = Counter()
    for fold in result['folds']:
        groups = [set(value['anonymous_category_ordinal'] for value in role['per_class'])
                  for role in fold['roles'].values()]
        assert set.union(*groups) == set(range(1, 16))
        assert all(not a & b for index, a in enumerate(groups) for b in groups[index + 1:])
        final.update(groups[-1])
        assert fold['necessary_support_only_PASS'] is True
        assert fold['global_video_disjoint_assignment_proven'] is False
        assert fold['complete_category_absence_in_video_proven'] is False
    assert final == Counter({index: 1 for index in range(1, 16)})
    assert result['new_training_protocol_frozen'] is False


def test_observed_all_category_cooccurrence_retains_deficits_not_favorable_subset():
    rows = [row(f'{category}:{video}', category, video) for category in range(15) for video in range(7)]
    result = cyclic_role_feasibility(rows)
    assert len(result['folds']) == 4
    assert all(fold['necessary_support_deficits'] == 15 for fold in result['folds'])
    assert all(fold['necessary_support_only_PASS'] is False for fold in result['folds'])
    assert result['all_source_categories_final_pseudo_once'] is True
    assert result['class_or_seed_selected_from_metric_or_support_results'] is False


def test_row_order_does_not_change_fixed_category_role_feasibility():
    rows = isolated_rows()
    assert cyclic_role_feasibility(rows) == cyclic_role_feasibility(list(reversed(rows)))


@pytest.mark.parametrize('bad', ('heldout', 'duplicate', 'too_many_folds'))
def test_forbidden_scope_and_empty_roles_refused(bad):
    rows = isolated_rows()
    if bad == 'heldout':
        rows[0]['partition'] = 'heldout_selection'
    elif bad == 'duplicate':
        rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        cyclic_role_feasibility(rows, folds=30 if bad == 'too_many_folds' else 4)
