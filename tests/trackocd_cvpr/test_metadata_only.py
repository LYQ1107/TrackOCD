import json
import pytest
from src.trackocd_cvpr.metadata_only import image_video_metadata, train_plan_source_metadata


def test_annotations_and_category_shuffle_are_not_decoded(tmp_path, monkeypatch):
    original = json.loads
    decoded = []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)
    # Synthetic fixtures contain no real dataset labels.
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    first.write_text('{"annotations":[{"category":"DO_NOT_DECODE","x":"\\\"}[]"}],"images":[{"id":1}],"videos":[{"id":2}],"categories":[3]}')
    second.write_text('{"categories":[9],"videos":[{"id":2}],"images":[{"id":1}],"annotations":[{"category":"CHANGED_FUTURE_LABEL"}]}')
    monkeypatch.setattr(json, 'loads', record)
    assert image_video_metadata(first) == image_video_metadata(second)
    assert all(b'DO_NOT_DECODE' not in v and b'CHANGED_FUTURE_LABEL' not in v for v in decoded)


@pytest.mark.parametrize('text', ['{"images":[],"videos":[]', '{"images":[],"images":[],"videos":[]}', '[]'])
def test_malformed_metadata_rejected(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises((ValueError, IndexError)):
        image_video_metadata(path)


def test_plan_source_metadata_never_decodes_private_rows_or_summary(tmp_path, monkeypatch):
    load = json.loads
    decoded = []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    path = tmp_path / 'synthetic_plan.json'
    path.write_text(json.dumps({'rows': [{'category': 'PROTECTED_ROW_DO_NOT_DECODE'}],
                                'summary': {'heldout': 'PROTECTED_SUMMARY_DO_NOT_DECODE'},
                                'sources': {'file.py': 'a' * 64}, 'annotation_sha256': 'b' * 64}))
    monkeypatch.setattr(json, 'loads', record)
    result = train_plan_source_metadata(path)
    assert result == {'sources': {'file.py': 'a' * 64}, 'annotation_sha256': 'b' * 64}
    assert all(b'PROTECTED_ROW_DO_NOT_DECODE' not in value and
               b'PROTECTED_SUMMARY_DO_NOT_DECODE' not in value for value in decoded)


@pytest.mark.parametrize('fields', (
    {'sources': [], 'annotation_sha256': 'a' * 64},
    {'sources': {'file.py': 'short'}, 'annotation_sha256': 'a' * 64},
    {'sources': {}, 'annotation_sha256': 'G' * 64}))
def test_invalid_plan_source_metadata_refused(tmp_path, fields):
    path = tmp_path / 'synthetic_plan.json'
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError):
        train_plan_source_metadata(path)


def test_nested_video_gt_and_arbitrary_image_metadata_are_never_json_decoded(tmp_path, monkeypatch):
    load, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    path = tmp_path / 'synthetic_metadata.json'
    path.write_text(json.dumps({
        'images': [{'id': 1, 'video_id': 2, 'frame_index': 0, 'file_name': 'train/x.jpg',
                    'metadata': {'GT': 'DO_NOT_DECODE_IMAGE_GT'}, 'labels': ['DO_NOT_DECODE_LABEL']}],
        'videos': [{'id': 2, 'neg_category_ids': [991122], 'not_exhaustive_category_ids': [773344],
                    'name': 'DO_NOT_DECODE_VIDEO_NAME', 'metadata': {'GT': 'DO_NOT_DECODE_VIDEO_GT'}}],
        'annotations': [{'category_id': 'DO_NOT_DECODE_ANNOTATION'}],
        'categories': [{'name': 'DO_NOT_DECODE_CATEGORY'}]}))
    monkeypatch.setattr(json, 'loads', record)
    result = image_video_metadata(path)
    assert result == {'images': [{'id': 1, 'video_id': 2, 'frame_index': 0, 'file_name': 'train/x.jpg'}],
                      'videos': [{'id': 2}]}
    # Actual decode instrumentation, not post-decode filtering.
    assert all(b'DO_NOT_DECODE' not in value and b'991122' not in value and b'773344' not in value
               for value in decoded)
    assert all(not value.lstrip().startswith((b'{', b'[')) for value in decoded)


def test_nested_gt_value_mutations_preserve_actual_projected_metadata(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    base = {'images': [{'id': 1}], 'videos': [{'id': 2, 'neg_category_ids': [111],
                                            'not_exhaustive_category_ids': [222]}]}
    first.write_text(json.dumps(base))
    base['videos'][0]['neg_category_ids'] = [333, 444]
    base['videos'][0]['not_exhaustive_category_ids'] = []
    second.write_text(json.dumps(base))
    assert image_video_metadata(first) == image_video_metadata(second) == {'images': [{'id': 1}], 'videos': [{'id': 2}]}


@pytest.mark.parametrize('payload', (
    {'images': [{'id': [123]}], 'videos': [{'id': 2}]},
    {'images': [], 'videos': [{'id': {'GT': 3}}]},
    {'images': [{}], 'videos': 2}))
def test_nonscalar_whitelisted_metadata_values_refused_before_decode(tmp_path, payload):
    path = tmp_path / 'synthetic_metadata.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        image_video_metadata(path)


@pytest.mark.parametrize('text', (
    '{"images":[] "videos":[]}',
    '{"images":[],"videos":[],}',
    '{"images":[{"id":1,}],"videos":[]}',
    '{"images":[{"id":1}{"id":2}],"videos":[]}',
    '{"images":[{"id":1 "video_id":2}],"videos":[]}',
    '{"images":[],"videos":[{"id":2,"id":3}]}',
    '{"images":[],"videos":[],"opaque":[{]}',
    '{"images":[],"videos":[],"opaque":[1,]}',
    '{"images":[],"videos":[],"opaque":NaN}',
    '{"images":[],"videos":[],"opaque":truely}',
    '{"images":[],"videos":[],"opaque":01}',
    '{"images":[],"videos":[],"opaque":"\\q"}',
    '{"images":[],"videos":[],"opaque":"\\uZZZZ"}',
    '{"images":[],"videos":[]}:garbage'))
def test_structurally_invalid_json_rejected_without_whole_object_decode(tmp_path, text):
    path = tmp_path / 'synthetic_bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        image_video_metadata(path)


@pytest.mark.parametrize('field,value', (('id', True), ('id', None), ('id', '99'),
                                         ('video_id', -1), ('frame_index', 0.5),
                                         ('file_name', 11), ('width', 0), ('height', -2)))
def test_allowlisted_metadata_requires_exact_type_and_valid_range(tmp_path, field, value):
    path = tmp_path / 'synthetic_bad_type.json'
    path.write_text(json.dumps({'images': [{field: value}], 'videos': []}))
    with pytest.raises(ValueError):
        image_video_metadata(path)


def test_valid_opaque_scalars_strings_and_deep_nesting_are_skipped(tmp_path, monkeypatch):
    load, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    path = tmp_path / 'synthetic.json'
    path.write_text('{"opaque":[null,true,false,-1.2e+3,"SENTINEL\\u1234\\\"{}[]",{}],'
                    '"images":[],"videos":[]}')
    monkeypatch.setattr(json, 'loads', record)
    assert image_video_metadata(path) == {'images': [], 'videos': []}
    assert all(b'SENTINEL' not in value and b'-1.2e+3' not in value for value in decoded)
    path.write_text('{"images":[],"videos":[],"opaque":' + '[' * 66 + '0' + ']' * 66 + '}')
    with pytest.raises(ValueError, match='excessively nested'):
        image_video_metadata(path)


def test_actual_downstream_projection_drops_gt_and_is_mutation_invariant(tmp_path):
    from src.trackocd_core.physical_qualification import metadata_universe
    from src.trackocd_cvpr.train_physical_resume import validate_plan
    payload = {'images': [{'id': 1, 'video_id': 2, 'frame_index': 0, 'file_name': 'train/v/a.jpg',
                          'labels': ['SYNTHETIC_GT']}],
               'videos': [{'id': 2, 'neg_category_ids': [111], 'not_exhaustive_category_ids': [222]}]}
    path = tmp_path / 'synthetic_metadata.json'
    path.write_text(json.dumps(payload))
    old_downstream = metadata_universe(payload)
    assert metadata_universe(image_video_metadata(path)) == old_downstream
    payload['videos'][0]['neg_category_ids'] = [444, 555]
    payload['images'][0]['labels'] = ['CHANGED_SYNTHETIC_GT']
    path.write_text(json.dumps(payload))
    assert metadata_universe(payload) == metadata_universe(image_video_metadata(path)) == old_downstream
    old_downstream[0]['images'][0].update(image_bytes=1, image_sha256='a' * 64)
    plan = {'GT_or_category_fields': False, 'videos': old_downstream}
    assert validate_plan(plan, {'videos': 1, 'canonical_images': 1}) == old_downstream
    old_downstream[0]['neg_category_ids'] = [999]
    with pytest.raises(ValueError, match='label-bearing'):
        validate_plan(plan, {'videos': 1, 'canonical_images': 1})
