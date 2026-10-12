from copy import deepcopy
import pytest
from src.trackocd_cvpr.physical_gt_census import reconcile_physical_known
from src.trackocd_cvpr.known_census_protocol import ROLES


def data():
    p = {4 * c + i: {c} for c in range(14) for i in range(4)}
    groups = (list(range(2)), [2, 3], [4, 5], list(range(6, 14)))
    assignment = {'classes': dict(zip(ROLES, groups)), 'video_ids': {
        r: [4 * c + i for c in cs for i in range(4)] for r, cs in zip(ROLES, groups)}}
    known = {'tracks': [{'category_id': next(iter(cs)), 'video_id': v, 'track_id': v,
                        'annotation_count': 1, 'distinct_annotated_images': 1} for v, cs in p.items()]}
    wrapper = {'not_model_facing': True, 'not_a_frozen_protocol': True,
               'assignment': assignment, 'role_class_support': {}}
    physical = {'schema_version': 'trackocd.cvpr.private-physical-GT.v1', 'not_model_facing': True,
                'not_a_frozen_protocol': True, 'semantic_category_labels_retained': False,
                'tracks': [{'video_id': v, 'opaque_track_id': v,
                            'observations': [[v, 0, 0, 0, 1, 1]]} for v in p]}
    physical['tracks'].append({'video_id': 24, 'opaque_track_id': 700, 'observations': [[24, 0, 0, 0, 1, 1]]})
    return p, known, wrapper, physical


def test_all_physical_counted_unknown_not_Novel_or_background_Known_joined():
    public, private = reconcile_physical_known(*data())
    assert public['physical_tracks'] == 57 and public['retained_Known_tracks_joined'] == 56
    assert public['additional_geometry_only_tracks'] == 1
    assert public['role_denominators'][ROLES[-1]]['all_annotated_physical_GT']['tracks'] == 33
    anonymous = [r for r in private['track_references'] if r['label_status'] == 'no_retained_Known_label']
    assert len(anonymous) == 1 and anonymous[0]['retained_Known_category_id'] is None
    assert not public['geometry_only_tracks_declared_background_or_labeled_Novel']
    assert not public['new_protocol_frozen']


def test_missing_Known_key_duplicate_physical_and_count_mismatch_refuse():
    original = data()
    for kind in ('missing', 'duplicate', 'count', 'geometry', 'outside_scope'):
        p, known, wrapper, physical = deepcopy(original)
        if kind == 'missing': physical['tracks'].pop(0)
        elif kind == 'duplicate': physical['tracks'].append(physical['tracks'][0].copy())
        elif kind == 'count': known['tracks'][0]['annotation_count'] = 2
        elif kind == 'geometry': physical['tracks'][0]['observations'][0][4] = 0
        else: physical['tracks'][0]['video_id'] = 999
        with pytest.raises(ValueError):
            reconcile_physical_known(p, known, wrapper, physical)
