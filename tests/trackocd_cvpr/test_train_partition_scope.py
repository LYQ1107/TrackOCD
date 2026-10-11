import copy
import json
import pytest
from src.trackocd_cvpr.metadata_only import train_plan_partition_metadata, TRAIN_PARTITIONS
from src.trackocd_cvpr.train_partition_scope import partition_video_scope


def synthetic_rows():
    return [{'video_id': i, 'partition': part} for i, part in enumerate(TRAIN_PARTITIONS)]


def test_actual_decode_skips_all_semantic_fields_even_in_protected_rows(tmp_path, monkeypatch):
    load, decoded = json.loads, []
    rows = [{**row, 'category_id': 987654321, 'simulation_role': 'DO_NOT_DECODE_ROLE',
             'observations': [{'bbox': [1, 2, 3, 4], 'category_name': 'DO_NOT_DECODE_GT'}]}
            for row in synthetic_rows()]
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps({'rows': rows, 'summary': {'categories': 'DO_NOT_DECODE_SUMMARY'}}))
    def record(value, *args, **kwargs):
        decoded.append(value)
        return load(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', record)
    assert train_plan_partition_metadata(path) == synthetic_rows()
    assert all(b'DO_NOT_DECODE' not in value and b'987654321' not in value for value in decoded)
    assert all(not value.lstrip().startswith((b'{', b'[')) for value in decoded)


def test_semantic_value_mutations_preserve_real_projected_scope(tmp_path):
    rows = [{**row, 'category_id': 987654321, 'observations': []} for row in synthetic_rows()]
    path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps({'rows': rows}))
    first = partition_video_scope(train_plan_partition_metadata(path))
    for row in rows:
        row.update(category_id=444, simulation_role='changed', observations=[{'x': 'changed'}])
    path.write_text(json.dumps({'rows': rows, 'summary': {'category_id': 333}}))
    assert partition_video_scope(train_plan_partition_metadata(path)) == first


def test_actual_video_partition_overlap_is_rejected_not_filtered():
    rows = synthetic_rows()
    rows[-1]['video_id'] = rows[0]['video_id']
    with pytest.raises(ValueError, match='crosses'):
        partition_video_scope(rows)


def test_administrative_plan_scope_not_category_absence_or_clearance():
    public, private = partition_video_scope(synthetic_rows(), 5)
    assert public['old_selected_plan_global_video_disjointness_verified'] is True
    assert public['category_absence_or_complete_annotation_census_proven'] is False
    assert public['new_protocol_or_scientific_access_cleared'] is False
    assert private['heldout_selection'] == [4]
    assert 'video_id' not in json.dumps(public)


@pytest.mark.parametrize('mutation', ('missing_partition', 'missing_video', 'extra_label', 'negative_video', 'empty'))
def test_administrative_projection_fail_closed(tmp_path, mutation):
    rows = copy.deepcopy(synthetic_rows())
    if mutation == 'missing_partition':
        rows[0].pop('partition')
    elif mutation == 'missing_video':
        rows[0].pop('video_id')
    elif mutation == 'extra_label':
        rows[0]['partition'] = 'DO_NOT_DECODE_SEMANTIC_LABEL'
    elif mutation == 'negative_video':
        rows[0]['video_id'] = -1
    else:
        rows = []
    path = tmp_path / 'synthetic_bad.json'
    path.write_text(json.dumps({'rows': rows}))
    with pytest.raises(ValueError):
        train_plan_partition_metadata(path)


def test_unknown_partition_and_string_id_refused_before_value_decode(tmp_path, monkeypatch):
    original, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)
    path = tmp_path / 'synthetic_bad.json'
    monkeypatch.setattr(json, 'loads', record)
    for row in ({'video_id': 0, 'partition': 'DO_NOT_DECODE_SEMANTIC_LABEL'},
                {'video_id': 'DO_NOT_DECODE_ID', 'partition': 'prototype'}):
        path.write_text(json.dumps({'rows': [row]}))
        with pytest.raises(ValueError):
            train_plan_partition_metadata(path)
    assert all(b'DO_NOT_DECODE' not in value for value in decoded)


def test_scope_does_not_drop_a_partition_or_accept_wrong_counts():
    with pytest.raises(ValueError):
        partition_video_scope(synthetic_rows()[:-1])
    with pytest.raises(ValueError):
        partition_video_scope(synthetic_rows(), 6)
