"""Synthetic external inventory tests, no actual Train label table read."""
import json
from unittest.mock import patch
import pytest

from src.trackocd_cvpr.train_protocol_inventory import (
    ALLOWED_PARTITIONS, INDEX_COLUMNS, LABEL_COLUMNS, read_registered_known_metadata, support_summary)


def item(key, category, video, count=2, part='representation_fit'):
    return {'key': key, 'category_id': category, 'video_id': video,
            'observation_count': count, 'partition': part, 'simulation_role': 'known'}


def test_category_video_support_not_confused_with_predicted_matching():
    rows = [item('a', 2, 10), item('b', 2, 10), item('c', 2, 11, 1), item('d', 9, 12)]
    result = support_summary(rows, {10, 12})
    assert result['categories'] == 2
    assert result['per_category'][0]['permitted_reference_pool']['cross_video_reference_pairs'] == 2
    assert result['per_category'][0]['reference_video_intersection_with_existing_predicted_cache']['cross_video_reference_pairs'] == 0
    assert result['permitted_reference_pool']['singleton_references'] == 1
    assert result['category_video_support_threshold_counts']['permitted_reference_pool']['2'] == 1
    assert result['category_selection_performed'] is result['new_fold_or_training_protocol_frozen'] is False
    assert result['predicted_cache_intersection_is_not_matching_or_predicted_track_denominator'] is True
    assert 'category_id' not in json.dumps(result)


def test_metadata_support_is_input_order_invariant():
    rows = [item('a', 2, 10), item('b', 2, 11), item('c', 9, 12)]
    assert support_summary(rows, {10}) == support_summary(list(reversed(rows)), {10})


@pytest.mark.parametrize('bad', ('sealed', 'pseudo', 'too_long', 'duplicate'))
def test_invalid_metadata_scope_refused(bad):
    rows = [item('a', 2, 10)]
    if bad == 'sealed':
        rows[0]['partition'] = 'heldout_selection'
    elif bad == 'pseudo':
        rows[0]['simulation_role'] = 'pseudo_novel'
    elif bad == 'too_long':
        rows[0]['observation_count'] = 17
    else:
        rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        support_summary(rows, {10})


def test_loader_requests_explicit_authorized_predicate_and_metadata_only(tmp_path):
    # Synthetic payloads; patch hashes and Parquet output, not real labels.
    manifest = {'source_role': 'TAO Train Known GT main training, not predicted/end-to-end',
                'payloads': {name: {'bytes': 1, 'sha256': 'registered'} for name in ('train_labels.parquet', 'index.parquet')}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    for name in manifest['payloads']:
        (tmp_path / name).write_bytes(b'x')
    calls = []
    def read(path, *, columns, filters):
        calls.append((path.name, columns, filters))
        if path.name == 'train_labels.parquet':
            value = [{key: item('a', 2, 10)[key] for key in LABEL_COLUMNS}]
        else:
            value = [{key: item('a', 2, 10)[key] for key in INDEX_COLUMNS}]
        return type('Table', (), {'to_pylist': lambda self: value})()
    with patch('src.trackocd_cvpr.train_protocol_inventory.sha256_file', return_value='registered'), patch('pyarrow.parquet.read_table', side_effect=read):
        rows, scope = read_registered_known_metadata(tmp_path, 'registered')
    assert rows == [item('a', 2, 10)]
    assert calls[0] == ('train_labels.parquet', list(LABEL_COLUMNS), [('partition', 'in', list(ALLOWED_PARTITIONS))])
    assert calls[1] == ('index.parquet', list(INDEX_COLUMNS), [('key', 'in', ['a'])])
    assert scope['old_final_heldout_labels_decoded'] is scope['feature_arrays_decoded'] is False
