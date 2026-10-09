import json
from pathlib import Path

import pytest

from scripts.trackocd_core.audit_assets import file_record, read_annotation, validate_roles


def test_inherited_roles_are_fixed_disjoint_sets():
    roles = json.loads((Path(__file__).resolve().parents[2] / 'configs/trackocd_core/roles.json').read_text())
    validate_roles(roles)
    assert [len(roles[k]) for k in ('known_ids', 'novel_ids', 'distractor_ids')] == [78, 209, 45]
    assert roles['source']['ids_recovered_verbatim']
    assert roles['source']['original_files_not_recovered']


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
