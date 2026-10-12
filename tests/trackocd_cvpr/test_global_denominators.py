from copy import deepcopy
import pytest
from src.trackocd_cvpr.global_protocol_coverage import audit_global_denominators
from src.trackocd_cvpr.known_census_protocol import ROLES


def data():
    p = {4 * c + i: {c} for c in range(14) for i in range(4)}
    groups = (list(range(2)), [2, 3], [4, 5], list(range(6, 14)))
    assignment = {'classes': dict(zip(ROLES, groups)), 'video_ids': {
        r: [4 * c + i for c in cs for i in range(4)] for r, cs in zip(ROLES, groups)}}
    index = {'tracks': [{'category_id': next(iter(cs)), 'video_id': v,
                         'annotation_count': 1, 'track_id': v} for v, cs in p.items()]}
    wrapper = {'not_model_facing': True, 'not_a_frozen_protocol': True,
               'assignment': assignment, 'role_class_support': {}}
    return p, index, wrapper


def test_all_Known_denominators_include_fit_and_outsidepool_not_background():
    p, index, wrapper = data()
    for c in (0, 99):
        p[24].add(c)
        index['tracks'].append({'category_id': c, 'video_id': 24, 'annotation_count': 1, 'track_id': 1000 + c})
    public, _ = audit_global_denominators(p, index, wrapper)
    final = public['role_denominators'][ROLES[-1]]
    assert final['own_role_pseudo']['tracks'] == 32
    assert final['fit_Known']['tracks'] == final['outside_eligible_Known']['tracks'] == 1
    assert final['all_retained_Known']['tracks'] == 34
    assert public['reconciled_Known_tracks'] == 58
    assert not public['full_GT_denominator_proven']
    assert not public['outside_eligible_Known_declared_background']


def test_unassigned_Known_rows_reconcile_not_silently_removed():
    p, index, wrapper = data()
    p[1000] = {99}
    index['tracks'].append({'category_id': 99, 'video_id': 1000, 'annotation_count': 3, 'track_id': 1000})
    public, _ = audit_global_denominators(p, index, wrapper)
    assert public['unassigned_retained_Known'] == {'tracks': 1, 'observations': 3}
    assert public['reconciled_Known_observations'] == 59


def test_other_pseudo_conflict_and_wrong_wrapper_refuse_no_forward_repair():
    p, index, wrapper = data()
    index['tracks'].append({'category_id': 2, 'video_id': 24, 'annotation_count': 1, 'track_id': 1000})
    with pytest.raises(ValueError, match='conflicts'):
        audit_global_denominators(p, index, wrapper)
    bad = deepcopy(wrapper); bad['not_a_frozen_protocol'] = False
    with pytest.raises(ValueError, match='wrapper'):
        audit_global_denominators(p, index, bad)
