import json
from pathlib import Path

from scripts.trackocd_cvpr.shadow_M_member_feature_cache_source_compatibility import (
    CONFIG, PRIOR, configuration_gate)
from src.trackocd_cvpr.M_endpoint_source_compatibility import verified_cohort
from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_only_declared_member_source_differs_and_all_old_endpoints_are_intact():
    manifest = json.loads((ROOT / CONFIG).read_text())
    prior = configuration_gate(ROOT, json.loads((ROOT / CONFIG).read_text()))
    cohort = verified_cohort(ROOT)
    assert len(prior['cases']) == 11
    assert len(cohort) == 9
    assert manifest['current_source_sha256'] == sha256_file(ROOT / 'src/trackocd_cvpr/member_reliability.py')
    assert len(manifest['declared_endpoint_source_paths']) == 6
    assert manifest['prior_shadow_result_sha256'] == sha256_file(ROOT / PRIOR)
    assert manifest['current_source_training_provenance_established'] is False


def test_compatibility_script_is_not_a_provenance_or_goal_acceptance_route():
    source = (ROOT / 'scripts/trackocd_cvpr/shadow_M_member_feature_cache_source_compatibility.py').read_text()
    assert 'runtime.verified_cohort = verified_cohort' in source
    assert 'current_source_training_provenance_established' in source
    assert 'current_source_deployment_trace_compatibility_is_not_M_qualification' in source
