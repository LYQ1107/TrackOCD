from pathlib import Path

from src.trackocd_cvpr.current_source_endpoint_cohort import audit_cohort, configuration

ROOT = Path(__file__).resolve().parents[2]


def test_registered_cohort_has_nine_distinct_cases_and_public_clock():
    manifest, endpoint, source = configuration(ROOT)
    assert len(manifest['cases']) == 9
    assert endpoint['recovery_output_prefix'] == manifest['output_prefix']
    assert source['current_source_paths']['src/trackocd_cvpr/member_reliability.py']


def test_public_endpoint_cohort_receipts_pass_without_private_inputs():
    result = audit_cohort(ROOT)
    assert result['status'] == 'COHORT_AUDIT_PASS_CURRENT_SOURCE_ENDPOINT_PROVENANCE_ONLY'
    assert result['case_count'] == 9
    assert result['all_public_receipts_normal']
    assert result['all_current_source_hashes_match_registered_closure']
    assert result['private_history_or_checkpoint_read'] is False
    assert result['formal_M_development_or_model_selection'] is False
