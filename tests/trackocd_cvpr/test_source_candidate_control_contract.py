import json
from pathlib import Path
import torch
from src.trackocd_cvpr.candidate_score_contract import FEATURE_COUNT, CONTEXT_COUNT, CandidateScoreMLP

ROOT = Path(__file__).resolve().parents[2]


def test_contract_matches_actual_shapes_parameters_and_unfitted_scope():
    contract = json.loads((ROOT / 'configs/trackocd_cvpr/source_candidate_control_contract.json').read_text())
    with torch.random.fork_rng(devices=[]):
        model = CandidateScoreMLP()
        assert sum(p.numel() for p in model.parameters()) == contract['model_parameters'] == 708
    assert len(contract['candidate_features']) == FEATURE_COUNT == 11
    assert len(contract['query_context']) == CONTEXT_COUNT == 6
    assert contract['new_GT_reads_training_cache_or_scientific_execution_authorized'] is False
    assert contract['actual_fitted_mechanisms_or_new_scientific_metric_results'] == contract['training_GPU_hours_used'] == 0


def test_source_integration_not_mistaken_for_full_memory_or_joint_research():
    contract = json.loads((ROOT / 'configs/trackocd_cvpr/source_candidate_control_contract.json').read_text())
    assert contract['same_frame_peer_birth_reuse_or_joint_assignment'] is False
    assert contract['Raw_adapt_agreement_features_or_differentiable_joint_R_C_training'] is False
    assert contract['self_support_exclusion_cross_instance_reuse_or_GT_purity_certified'] is False
    assert contract['learned_quarantine_version_CT_evaluator_or_strong_frontend_qualification'] is False
    assert contract['callback_side_effects_or_RNG_rolled_back'] is False
    proposal = json.loads((ROOT / 'configs/trackocd_cvpr/clean_train_gt_scope_proposal.json').read_text())
    assert proposal['new_GT_read_authorized_by_this_proposal'] is False
