from pathlib import Path
from types import SimpleNamespace

import scripts.trackocd_cvpr.current_source_M_endpoint_refresh_transport_retry1 as retry

ROOT = Path(__file__).resolve().parents[2]


def test_transport_retry_binds_new_namespace_and_preserves_prior_evidence():
    cfg = retry.configuration(SimpleNamespace(variant='relative_quarantine', seed=1027))
    assert cfg['output_directory'].endswith(
        'M_GT_window_current_source_v1_transport_retry1_relative_quarantine_s1027')
    assert cfg['optimizer_steps'] == 200
    assert (ROOT / retry.RECOVERY_TERMINAL).is_file()


def test_transport_retry_uses_the_repaired_terminal_supervisor():
    source = (ROOT / 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh_transport_retry1.py').read_text()
    assert 'terminal_recovery1' in source
    assert 'recovery.main()' in source
