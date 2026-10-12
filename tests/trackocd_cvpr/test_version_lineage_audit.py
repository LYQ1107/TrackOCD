from dataclasses import replace
import pytest
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.versioned_reliability_memory import VersionedReliabilityMemory
from src.trackocd_cvpr.version_lineage_audit import audit_version_lineage
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs


def pollution_and_transfer():
    bank, router, memory = PrefixMemberBank(), CurrentObservationRouter(), VersionedReliabilityMemory()
    batch = router.ingest(0, [obs('known'), obs('pseudo', 1)])
    bank.apply_frame(batch, ['S:0', 'S:0']); memory.refresh(bank, MemberReliabilityMLP())
    bank.apply_frame(router.ingest(1, [obs('pseudo', 2)]), ['S:1']); memory.refresh(bank, MemberReliabilityMLP())
    targets = {(7, 'known'): ('Known', 0), (7, 'pseudo'): ('policy_pseudo', 1)}
    return bank, memory, targets


def test_bad_source_history_survives_pseudo_move_into_new_clean_looking_token():
    bank, memory, targets = pollution_and_transfer()
    result = audit_version_lineage(memory.versions(), bank.commit_history(), targets, {0: 7})
    assert result['strict_latest_version_status_counts'] == {'VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY': 2}
    assert result['final_tokens_including_tombstones'] == 2
    assert result['all_parent_history_and_empty_tombstones_retained']
    assert not result['GT_feedback_or_current_active_version_purity_or_official_CT_or_Pred_claim']


def test_unknown_is_not_declared_pseudo_purity_and_no_gt_action_feedback():
    bank, memory, targets = pollution_and_transfer(); targets.clear()
    saved = bank.commit_history(), memory.versions()
    result = audit_version_lineage(memory.versions(), bank.commit_history(), targets, {0: 7})
    assert result['strict_latest_version_status_counts'] == {'UNSCORABLE_EMPTY_OR_UNKNOWN_HISTORY': 2}
    assert (bank.commit_history(), memory.versions()) == saved


def test_full_parent_bound_missing_cycle_or_own_ancestry_refuses_not_prunes():
    bank, memory, targets = pollution_and_transfer(); versions = memory.versions()
    with pytest.raises(ValueError, match='bound'):
        audit_version_lineage(versions, bank.commit_history(), targets, {0: 7}, max_union_references=1)
    bad = (*versions[:-1], replace(versions[-1], parents=(('S:999', 0),)))
    with pytest.raises(ValueError): audit_version_lineage(bad, bank.commit_history(), targets, {0: 7})
    own = (*versions[:-1], replace(versions[-1], parents=()))
    with pytest.raises(ValueError, match='transfer source ancestry'):
        audit_version_lineage(own, bank.commit_history(), targets, {0: 7})
