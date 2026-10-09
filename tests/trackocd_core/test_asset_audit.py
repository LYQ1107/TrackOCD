import json
from pathlib import Path

import pytest

from scripts.trackocd_core.audit_assets import file_record, read_annotation, validate_roles
from scripts.trackocd_core.recover_role_files import recover


def test_inherited_roles_are_fixed_disjoint_sets():
    roles = json.loads((Path(__file__).resolve().parents[2] / 'configs/trackocd_core/roles.json').read_text())
    validate_roles(roles)
    assert [len(roles[k]) for k in ('known_ids', 'novel_ids', 'distractor_ids')] == [78, 209, 45]
    assert roles['source']['ids_recovered_verbatim']
    assert roles['source']['original_split_bytes_recovered']


@pytest.mark.parametrize('name,split', [('tao_test_annotations.json', 'test'), ('test.json', 'val'), ('validation.json', 'train')])
def test_test_and_unregistered_annotation_paths_fail_before_open(name, split):
    with pytest.raises(ValueError, match='FORBIDDEN'):
        read_annotation(Path('/nonexistent') / name, split)


def test_role_overlap_is_rejected():
    with pytest.raises(ValueError, match='overlapping'):
        validate_roles({'known_ids': [1], 'novel_ids': [1], 'distractor_ids': []})


def test_missing_asset_is_not_reported_as_recovered(tmp_path):
    record = file_record(tmp_path / 'missing.npz')
    assert not record['exists']
    assert 'sha256' not in record


def _roles():
    return json.loads((Path(__file__).resolve().parents[2] / 'configs/trackocd_core/roles.json').read_text())


def test_recovered_split_bytes_match_all_original_hashes_and_resume(tmp_path):
    result = recover(_roles(), tmp_path)
    assert result['status'] == 'PASS'
    assert not result['role_ids_changed']
    assert sorted(record['size_bytes'] for record in result['files'].values()) == [276, 471, 1284]
    assert all(record['matches_original_byte_hash'] for record in result['files'].values())
    assert recover(_roles(), tmp_path)['files'] == result['files']


def test_corrupt_recovery_inputs_fail_before_any_assets_are_written(tmp_path):
    roles = _roles()
    roles['novel_ids'][0] = 999999
    with pytest.raises(ValueError, match='ORIGINAL_ROLE_HASH_MISMATCH'):
        recover(roles, tmp_path / 'new-assets')
    assert not (tmp_path / 'new-assets').exists()


def test_recovery_never_overwrites_a_different_local_asset(tmp_path):
    target = tmp_path / 'unknown_ids_val.json'
    target.write_text('do not overwrite')
    with pytest.raises(ValueError, match='REFUSING_TO_OVERWRITE'):
        recover(_roles(), tmp_path)
    assert target.read_text() == 'do not overwrite'
    assert not (tmp_path / 'known_ids.json').exists()
