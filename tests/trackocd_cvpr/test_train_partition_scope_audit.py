from pathlib import Path
import pytest
from src.trackocd_cvpr.train_partition_scope import partition_video_scope
from src.trackocd_cvpr.metadata_only import TRAIN_PARTITIONS
from scripts.trackocd_cvpr.audit_train_partition_scope import reject_sensitive_open, validate_summary


def test_admin_barrier_permits_only_exact_plan_not_labels_or_other_plan():
    allowed = Path('/synthetic/old/selection_plan.json')
    reject_sensitive_open('open', (str(allowed), 'r', 0), allowed)
    for path in ('/synthetic/other/selection_plan.json', '/synthetic/annotations/train.json',
                 '/synthetic/train_labels.parquet', '/synthetic/roles.json', '/synthetic/x.npy'):
        with pytest.raises(PermissionError):
            reject_sensitive_open('open', (path, 'r', 0), allowed)


def test_all_declared_counts_required_before_private_index_can_be_accepted():
    public, _ = partition_video_scope([{'video_id': i, 'partition': part} for i, part in enumerate(TRAIN_PARTITIONS)])
    cfg = {'expected_selected_rows': 5, 'expected_partition_rows': {part: 1 for part in TRAIN_PARTITIONS}}
    validate_summary(public, cfg)
    cfg['expected_partition_rows']['heldout_selection'] = 0
    with pytest.raises(ValueError, match='no exclusions'):
        validate_summary(public, cfg)
