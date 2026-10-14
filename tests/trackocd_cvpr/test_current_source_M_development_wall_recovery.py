import json
from pathlib import Path
from types import SimpleNamespace

import scripts.trackocd_cvpr.evaluate_M_development_current_source_wall_recovery1 as recovery

ROOT = Path(__file__).resolve().parents[2]


def test_wall_recovery_binds_preserved_failure_and_fresh_namespace():
    cfg = recovery.configuration(ROOT, SimpleNamespace(model='Mean_latest', seed=1027))
    assert cfg['wall_seconds_ceiling'] == 6000
    assert cfg['output_directory'].endswith('M_GT_causal_dev_current_source_wall_recovery1_Mean_latest')
    manifest = json.loads((ROOT / recovery.CONFIG).read_text())
    assert manifest['failed_case'] == {'model': 'Mean_latest', 'seed': 1027}


def test_wall_recovery_does_not_reuse_partial_result_or_checkpoint():
    source = (ROOT / recovery.SCRIPT).read_text()
    assert 'fresh_worker_no_partial_metric_or_checkpoint_reuse' in source
    assert 'RECOVERY_PREFIX' in source
