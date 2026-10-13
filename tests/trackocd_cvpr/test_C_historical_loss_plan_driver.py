import json
from pathlib import Path
import subprocess
import pytest
from scripts.trackocd_cvpr.audit_C_historical_loss_plans import (
    CONFIG, BANK, SOURCES, bank_without_copy_method, checked_configuration)

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_authorized_scope_no_selected_trial_new_updates_or_guard_relaxation():
    cfg = json.loads((ROOT / CONFIG).read_text()); checked_configuration(cfg)
    for change in ({'seeds': [1027]}, {'new_optimizer_steps': 1}, {'wall_seconds_ceiling': 6000},
                   {'data_config': 'other-scope.json'}, {'private_history_or_raw_GT_upload': True}):
        with pytest.raises(ValueError): checked_configuration({**cfg, **change})


def test_bank_difference_is_only_copy_method_not_a_new_numeric_implementation():
    cfg = json.loads((ROOT / CONFIG).read_text())
    old = subprocess.check_output(['git', 'show', cfg['copy_only_bank_reference_commit'] + ':' + BANK], cwd=ROOT)
    current = (ROOT / BANK).read_bytes()
    assert bank_without_copy_method(old) == bank_without_copy_method(current)
    changed = current.replace(b'count = len(groups)', b'count = len(groups) + 1')
    assert changed != current and bank_without_copy_method(old) != bank_without_copy_method(changed)


def test_owned_source_gate_all9_history_hashes_before_decode_no_checkpoint_loading():
    source = (ROOT / 'scripts/trackocd_cvpr/audit_C_historical_loss_plans.py').read_text()
    assert 'C_cohort.verified_cohort(ROOT)' in source
    assert 'private_prediction_history' in source and 'maximum_private_history_bytes_each' in source
    assert 'torch.load' not in source and 'return runtime.main()' in source
    assert source.index('sha256_file(path) != receipt') < source.index('private = json.loads')
    assert source.index('source_gate(args, cfg, safety)') < source.index('load_completed_policy_videos(ROOT')
    assert all('/result.json' in p or 'private_' not in Path(p).name for p in SOURCES)
