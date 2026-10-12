import json
from pathlib import Path
import torch
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_cvpr.differentiable_candidate_features import FEATURE_COUNT

ROOT = Path(__file__).resolve().parents[2]


def test_joint_contract_uses_actual_parameter_budgets_not_old_708_claim():
    contract = json.loads((ROOT / 'configs/trackocd_cvpr/source_joint_evidence_contract.json').read_text())
    with torch.random.fork_rng(devices=[]):
        assert sum(p.numel() for p in RawPreservingEvidence().parameters()) == contract['evidence_model_parameters'] == 50113
        assert sum(p.numel() for p in CandidateScoreMLP(feature_count=14).parameters()) == contract['joint_candidate_scorer_parameters'] == 804
        assert sum(p.numel() for p in CandidateScoreMLP().parameters()) == 708
    assert contract['joint_candidate_features'] == FEATURE_COUNT == 14


def test_tensor_prototype_does_not_clear_controller_data_or_scientific_gates():
    contract = json.loads((ROOT / 'configs/trackocd_cvpr/source_joint_evidence_contract.json').read_text())
    assert contract['existing_NumPy_controller_is_now_jointly_differentiable'] is False
    assert contract['budget_matched_R_C_controls_or_actual_training_accepted'] is False
    assert contract['new_GT_training_cache_or_scientific_execution_authorized'] is False
    assert contract['Goals_A_B_C_or_full_scientific_acceptance'] is False
    assert contract['training_GPU_hours_used'] == 0
    assert contract['pooled_gate_retains_pooled_motion_and_time_not_pure_static_control'] is True
    proposal = json.loads((ROOT / 'configs/trackocd_cvpr/clean_train_gt_scope_proposal.json').read_text())
    assert proposal['new_GT_read_authorized_by_this_proposal'] is False
