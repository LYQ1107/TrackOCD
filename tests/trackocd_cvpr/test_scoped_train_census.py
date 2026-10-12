"""Synthetic-only decode instrumentation and actual paired input mutations."""
import copy
import hashlib
import json
from pathlib import Path
import pytest

from src.trackocd_cvpr.scoped_train_census import census, inherited_known_ids, validate_video_scope
from scripts.trackocd_cvpr.census_clean_train_gt import validate_authorization, reject_sensitive_open

ROOT = Path(__file__).resolve().parents[2]


def write_fixture(tmp_path, payload):
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps(payload))
    return path


def fixture():
    # Target fields intentionally occur before video_id, to catch whole-object
    # decoding or value-order-dependent conditional readers.
    return {'categories': [{'name': 'NO_DECODE_CATEGORY_NAME'}],
            'videos': [{'id': 1, 'neg_category_ids': ['NO_DECODE_NEG'],
                        'not_exhaustive_category_ids': ['NO_DECODE_NOTEX']}],
            'annotations': [
                {'category_id': 'NO_DECODE_HELDOUT_CATEGORY', 'bbox': ['NO_DECODE_HELDOUT_BOX'], 'video_id': 2},
                {'category_id': 'NO_DECODE_UNLISTED_CATEGORY', 'bbox': ['NO_DECODE_UNLISTED_BOX'], 'video_id': 99},
                {'category_id': 777, 'bbox': ['NO_DECODE_NONKNOWN_BOX'], 'video_id': 1},
                {'category_id': 3, 'image_id': 4, 'track_id': 5, 'bbox': [0, 1, 2, 3], 'video_id': 1},
                {'category_id': 3, 'image_id': 6, 'track_id': 5, 'bbox': [1, 2, 3, 4], 'video_id': 1},
                {'category_id': 3, 'image_id': 7, 'track_id': 8, 'bbox': [0, 0, 1, 1], 'video_id': 1}]}


def test_video_first_decode_barrier_actual_instrumentation(tmp_path, monkeypatch):
    path = write_fixture(tmp_path, fixture())
    load, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', record)
    public, private = census(path, frozenset({1}), frozenset({2}), frozenset({3}))
    assert all(b'NO_DECODE' not in value for value in decoded)
    assert all(not value.lstrip().startswith((b'{', b'[')) for value in decoded)
    assert public['counts']['excluded_heldout_records_byte_skipped'] == 1
    assert public['counts']['other_unlisted_records_byte_skipped'] == 1
    assert public['counts']['allowed_nonKnown_records_not_retained'] == 1
    assert public['retained_Known_tracks'] == 2
    assert public['track_observation_count_histogram'] == {1: 1, 2: 1}
    assert len(private['tracks']) == 2
    assert '777' not in json.dumps(private)


def test_actual_protected_mutations_leave_census_identical(tmp_path):
    payload = fixture()
    baseline = census(write_fixture(tmp_path, payload), {1}, {2}, {3})
    payload['annotations'][0].update(category_id=923, bbox=[999, 999, 999, 999], track_id=111)
    payload['annotations'][1].update(category_id=3, bbox={'unknown': 'CHANGED_PROTECTED'}, image_id=999)
    payload['annotations'][2]['bbox'] = [999, 999, 999, 999]
    payload['categories'] = [{'name': 'CHANGED_NOVEL_NAME', 'id': 777}]
    payload['videos'][0].update(neg_category_ids=[888], not_exhaustive_category_ids=[999])
    assert census(write_fixture(tmp_path, payload), {1}, {2}, {3}) == baseline


def test_whitelist_change_rejects_before_protected_target_decode(tmp_path):
    with pytest.raises(ValueError, match='integer'):
        census(write_fixture(tmp_path, fixture()), {1, 2}, set(), {3})


def test_known_only_roles_projection_instrumented(tmp_path, monkeypatch):
    path = write_fixture(tmp_path, {'known_ids': [3, 8], 'novel_ids': ['NO_DECODE_NOVEL_IDS'],
                                   'names': ['NO_DECODE_NAMES'], 'roles': {'x': 'NO_DECODE_ROLES'}})
    load, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', record)
    assert inherited_known_ids(path) == frozenset({3, 8})
    assert all(b'NO_DECODE' not in value for value in decoded)
    assert all(not value.lstrip().startswith((b'{', b'[')) for value in decoded)


@pytest.mark.parametrize('ids', [[], [3, 3], [True], [-1], ['3'], [1.5]])
def test_invalid_known_ids_fail_closed(tmp_path, ids):
    with pytest.raises(ValueError):
        inherited_known_ids(write_fixture(tmp_path, {'known_ids': ids}))


@pytest.mark.parametrize('change', [
    {'video_id': None}, {'category_id': True}, {'bbox': [0, 0, 0, 1]},
    {'bbox': [0, 0, 1]}, {'bbox': [0, 0, 1, {'target': 'NO_DECODE_NESTED_BOX'}]},
    {'image_id': -1}, {'track_id': '5'}])
def test_invalid_allowed_known_records_fail_without_repair(tmp_path, change, monkeypatch):
    payload = fixture()
    payload['annotations'][3].update(change)
    path = write_fixture(tmp_path, payload)
    load, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', record)
    with pytest.raises(ValueError):
        census(path, {1}, {2}, {3})
    assert all(b'NO_DECODE' not in value for value in decoded)


@pytest.mark.parametrize('text', [
    '{"annotations":[],}', '{"annotations":[]}:trailing',
    '{"annotations":[{"video_id":1,"video_id":2}]}',
    '{"annotations":[{"category_id":3}]}', '{"annotations":{},"videos":[]}',
    '{"annotations":[],"annotations":[]}', '{"annotations":[],"opaque":[1,]}'])
def test_invalid_source_structure_refused(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        census(path, {1}, {2}, {3})


def test_duplicate_known_track_image_and_category_inconsistency_rejected(tmp_path):
    payload = fixture()
    payload['annotations'].append(copy.deepcopy(payload['annotations'][3]))
    with pytest.raises(ValueError, match='duplicate'):
        census(write_fixture(tmp_path, payload), {1}, {2}, {3})
    payload = fixture()
    payload['annotations'][4]['category_id'] = 4
    with pytest.raises(ValueError, match='Inconsistent'):
        census(write_fixture(tmp_path, payload), {1}, {2}, {3, 4})


def test_missing_positive_records_not_called_category_absence(tmp_path):
    public, private = census(write_fixture(tmp_path, {'annotations': []}), {1}, {2}, {3})
    assert public['allowlisted_videos_without_positive_records'] == 1
    assert public['positive_annotations_certify_visual_category_absence'] is False
    assert public['new_protocol_frozen'] is False
    assert private['tracks'] == []


def test_scope_overlap_and_wrong_counts_are_not_trimmed():
    index = {'schema_version': 'trackocd.cvpr.private-administrative-video-exclusion.v1',
             'not_model_facing': True, 'semantic_target_labels': False,
             'partitions': dict(prototype=[1], representation_fit=[2], development=[3], policy_train=[4], heldout_selection=[5])}
    assert validate_video_scope(index, 4, 1) == (frozenset({1, 2, 3, 4}), frozenset({5}))
    index['partitions']['heldout_selection'] = [1]
    with pytest.raises(ValueError, match='overlapping'):
        validate_video_scope(index, 4, 1)
    with pytest.raises(ValueError):
        census(None, {1}, {1}, {3})


def test_approval_preserves_pending_proposal_and_exact_human_scope():
    cfg = json.loads((ROOT / 'configs/trackocd_cvpr/clean_train_gt_census_v1.json').read_text())
    approval = json.loads((ROOT / cfg['approval']).read_text())
    proposal_bytes = (ROOT / cfg['proposal']).read_bytes()
    digest = hashlib.sha256(proposal_bytes).hexdigest()
    validate_authorization(cfg, approval, digest)
    assert json.loads(proposal_bytes)['new_GT_read_authorized_by_this_proposal'] is False
    assert approval['message_identifier_available'] is False
    assert approval['original_full_goal_preserved'] is True
    for key, value in [('user_message_verbatim', 'automatic continuation'), ('allowlisted_videos', 430),
                       ('external_known_only_Train_census_authorized', False),
                       ('automatic_training_or_full_cache_authorized_by_this_receipt', True)]:
        changed = {**approval, key: value}
        with pytest.raises(ValueError):
            validate_authorization(cfg, changed, digest)


def test_open_barrier_allows_only_exact_sensitive_paths(tmp_path):
    approved = (tmp_path / 'annotations/train.json').resolve()
    reject_sensitive_open('open', (str(approved), 'r', 0), {approved})
    for name in ('annotations/val.json', 'annotations/test.json', 'roles.json', 'x.npz', 'x.pt', 'x.parquet', 'frames/x.jpg'):
        with pytest.raises(PermissionError):
            reject_sensitive_open('open', (str(tmp_path / name), 'r', 0), {approved})


def test_resource_guard_runs_on_skipped_records(tmp_path):
    calls = []
    census(write_fixture(tmp_path, {'annotations': [{'video_id': 99, 'category_id': 'NO_DECODE'}] * 600}),
           {1}, {2}, {3}, lambda: calls.append(True))
    assert len(calls) >= 5
