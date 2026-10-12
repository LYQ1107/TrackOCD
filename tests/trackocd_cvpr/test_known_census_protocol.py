import copy
import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.known_census_protocol import known_presence, fixed_hypotheses, role_candidates, ROLES
from scripts.trackocd_cvpr.audit_known_census_protocol import exact_open_barrier


def synthetic_index():
    tracks = [{'video_id': v, 'track_id': c, 'category_id': c,
               'annotation_count': 1, 'distinct_annotated_images': 1}
              for c in range(90000, 90012) for v in range((c - 90000) * 4, (c - 90000) * 4 + 4)]
    presence = {str(row['video_id']): [row['category_id']] for row in tracks}
    return {'schema_version': 'trackocd.cvpr.private-known-positive-census.v1',
            'not_model_facing': True, 'tracks': tracks, 'video_positive_Known_categories': presence}


def test_reconciles_every_short_track_no_hidden_cap_or_filter():
    presence, counts = known_presence(synthetic_index(), set(range(48)), set(range(90000, 90012)))
    assert counts == {'tracks': 48, 'observations': 48, 'videos': 48, 'positive_Known_classes': 12}
    assert len(presence) == 48


@pytest.mark.parametrize('mutation', ['protected', 'nonknown', 'wrong_presence', 'duplicate', 'extra_label', 'invalid_count'])
def test_census_or_scope_mutations_refused(mutation):
    index = synthetic_index()
    if mutation == 'protected':
        index['tracks'][0]['video_id'] = 999
    elif mutation == 'nonknown':
        index['tracks'][0]['category_id'] = 999
    elif mutation == 'wrong_presence':
        index['video_positive_Known_categories']['0'] = [999]
    elif mutation == 'duplicate':
        index['tracks'].append(copy.deepcopy(index['tracks'][0]))
    elif mutation == 'extra_label':
        index['tracks'][0]['category_name'] = 'FORBIDDEN'
    else:
        index['tracks'][0]['annotation_count'] = 2
    with pytest.raises(ValueError):
        known_presence(index, set(range(48)), set(range(90000, 90012)))


def test_all_fixed_cases_have_certificates_without_published_IDs_or_names():
    presence, _ = known_presence(synthetic_index(), set(range(48)), set(range(90000, 90012)))
    for threshold in (0, 3):
        public, private = fixed_hypotheses(presence, eligibility_video_support=threshold)
        assert len(public['folds']) == 4
        assert all(f['globally_video_disjoint_sufficient_witness_found'] for f in public['folds'])
        assert all(f['individual_necessary_support_deficits'] == 0 for f in public['folds'])
        assert public['class_pool'] == 12 and public['new_protocol_frozen'] is False
        assert all(str(c) not in json.dumps(public) for c in range(90000, 90012))
        finals = [c for fold in private for c in fold['classes']['independent_final_pseudo']]
        assert sorted(finals) == list(range(90000, 90012))


def test_actual_positive_conflict_added_changes_registered_candidate_filter():
    classes = dict(zip(ROLES, ({1}, {2}, {3}, {4})))
    first = {10: {1}, 20: {2}, 30: {3}, 40: {4}}
    candidates = role_candidates(first, classes)
    assert 10 in candidates['fit_known']
    changed = {**first, 10: {1, 4}}
    second = role_candidates(changed, classes)
    assert 10 not in second['fit_known'] and 10 not in second['development_pseudo']
    assert 10 in second['independent_final_pseudo']
    assert candidates != second


def test_rare_classes_not_dropped_from_unfiltered_diagnostic():
    presence, _ = known_presence(synthetic_index(), set(range(48)), set(range(90000, 90012)))
    presence[99] = {99999}
    unfiltered, _ = fixed_hypotheses(presence, eligibility_video_support=0)
    eligible, _ = fixed_hypotheses(presence, eligibility_video_support=3)
    assert unfiltered['class_pool'] == 13 and eligible['class_pool'] == 12
    assert eligible['classes_outside_pool'] == 1 and eligible['outside_pool_not_declared_background'] is True
    assert len(unfiltered['folds']) == len(eligible['folds']) == 4
    assert any(f['individual_necessary_support_deficits'] for f in unfiltered['folds'])


@pytest.mark.parametrize('kwargs', [{'eligibility_video_support': 7}, {'folds': 3}, {'policy_classes': 1}, {'minimum_fit': 2}])
def test_no_undeclared_threshold_or_quota_search(kwargs):
    presence, _ = known_presence(synthetic_index(), set(range(48)), set(range(90000, 90012)))
    with pytest.raises(ValueError):
        fixed_hypotheses(presence, **kwargs)


def test_exact_open_scope_rejects_annotation_even_train_and_other_private_index(tmp_path):
    allowed = (tmp_path / 'private_known_support_index.json').resolve()
    out = (tmp_path / 'out').resolve()
    exact_open_barrier('open', (str(allowed), 'r', 0), {allowed}, set(), out)
    exact_open_barrier('open', (str(out / 'result.json'), 'w', 0), {allowed}, set(), out)
    for path in ('annotations/train.json', 'annotations/val.json', 'roles.json', 'private_elsewhere.json', 'x.npz'):
        with pytest.raises(PermissionError):
            exact_open_barrier('open', (str(tmp_path / path), 'r', 0), {allowed}, set(), out)


def test_registration_contains_exact_existing_scope_and_post_aggregate_disclosure():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'configs/trackocd_cvpr/known_census_protocol_feasibility_v1.json').read_text())
    assert cfg['two_fixed_eligibility_thresholds'] == [0, 3]
    assert cfg['expected'] == {'tracks': 1962, 'observations': 38680, 'videos': 429, 'positive_Known_classes': 48}
    assert cfg['training_GPU_hours'] == 0
    assert 'already observed' in cfg['threshold_rationale']
