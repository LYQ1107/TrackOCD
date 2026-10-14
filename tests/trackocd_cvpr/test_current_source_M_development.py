import json
from pathlib import Path
from types import SimpleNamespace

import scripts.trackocd_cvpr.evaluate_M_development_current_source as current

ROOT = Path(__file__).resolve().parents[2]


def test_current_source_M_manifest_binds_audited_nine_endpoint_cohort():
    cfg = current.configuration(ROOT, SimpleNamespace(model='relative_quarantine', seed=1027))
    assert cfg['output_directory'].endswith('M_GT_causal_dev_current_source_v1_relative_quarantine_s1027')
    manifest = json.loads((ROOT / current.CONFIG).read_text())
    assert manifest['endpoint_cohort_manifest'].endswith('M_current_source_endpoint_cohort_audit_v1.json')
    assert not manifest['select_model_or_checkpoint']


def test_current_source_M_wrapper_uses_current_cohort_and_fresh_namespace():
    source = (ROOT / current.SCRIPT).read_text()
    assert 'current_source_M_development_cohort' in source
    assert 'base.verified_cohort' in source
    assert current.configuration(ROOT, SimpleNamespace(model='Mean_latest', seed=1027))['output_prefix'].endswith(
        'M_GT_causal_dev_current_source_v1')
