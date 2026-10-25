from pathlib import Path

from scripts.trackocd_cvpr.audit_legacy_preservation import build_audit


def test_legacy_research_lineage_and_preserved_refs_are_verified():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["status"] == "PASS_LEGACY_RESEARCH_PRESERVATION_VERIFIED"
    assert audit["scope"] == {
        "git_metadata_only": True,
        "large_streams_opened": False,
        "data_or_weights_read": False,
        "tao_test_read": False,
        "external_processes_touched": False,
        "refs_moved": False,
        "model_selection_performed": False,
    }
    assert audit["repository"]["current_head_is_descendant_of_historical_parent"] is True
    assert all(item["matches_expected"] for item in audit["protected_refs"].values())
    assert audit["manifest_binding"]["matches_historical_parent"] is True
    assert audit["manifest_binding"]["manifest_generator_source_sha256_matches"] is True
    assert audit["manifest_binding"]["all_preserved_failed_attempts_marked_not_deleted"] is True
    assert audit["boundary"]["parent_namespace_was_not_assumed"] is True
