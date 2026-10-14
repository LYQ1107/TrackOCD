from pathlib import Path
from types import SimpleNamespace

from scripts.trackocd_cvpr.current_source_M_endpoint_refresh_terminal_recovery1 import configuration

ROOT = Path(__file__).resolve().parents[2]


def test_terminal_recovery_binds_failed_receipt_and_new_namespace():
    cfg = configuration(SimpleNamespace(variant='relative_quarantine', seed=1027))
    assert cfg['output_directory'].endswith(
        'M_GT_window_current_source_v1_terminal_recovery1_relative_quarantine_s1027')
    assert not (ROOT / cfg['output_directory']).exists()
    assert cfg['optimizer_steps'] == 200


def test_terminal_recovery_restores_kernel_terminal_state_check():
    source = (ROOT / 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh_terminal_recovery1.py').read_text()
    assert 'kernel_terminal_status' in source
    assert 'terminal_state_check_repair' in source
    assert 'fresh_worker_no_old_checkpoint_or_history_reuse' in source
