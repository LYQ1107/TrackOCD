from types import SimpleNamespace
import numpy as np
import torch

from src.trackocd_cvpr.candidate_fit_pilot import fit_candidate_pilot
from tests.trackocd_cvpr.test_policy_training_data import arrays


def run(videos, projection):
    known = {0: np.eye(1, 768, 0, dtype=np.float32)[0]}
    cfg = {'seed': 1027, 'optimizer_steps': 2, 'learning_rate': .001,
           'weight_decay': .0001, 'gradient_clip': 1., 'K_per_family': 8}
    return fit_candidate_pilot(videos, projection, known, cfg, lambda: None, lambda row: None)


def test_real_synthetic_fit_and_full_rollout_not_ten_step_scientific_claim():
    p = SimpleNamespace(targets={(1, 12): ('Known', 0), (1, 99): ('no_retained_semantic_truth', None)})
    model, _, summary, events, _ = run([(1, arrays())], p)
    assert summary['optimizer_steps'] == 2 and summary['trainable_parameters'] == 708
    assert summary['complete_rollout']['metadata_frames'] == len(events) == 4
    assert summary['complete_rollout']['empty_frames'] == 2
    assert summary['complete_rollout']['physical_observations'] == 3
    assert summary['loss_plan_counts_all_frames']['MASKED_QUERY_OR_CONFLICTING_ALIASES'] == 1
    assert not summary['GT_created_reassigned_or_repaired_bank_or_removed_wrong_history']
    assert not summary['self_support_exclusion_or_independent_category_reuse_or_C_M_scientific_acceptance']
    assert not model.training


def test_actual_fit_ID_rename_and_future_change_cannot_rewrite_earlier_decisions():
    a = arrays(); labels = {(1, 12): ('Known', 0), (1, 99): ('policy_pseudo', 0)}
    x, _, sx, ex, _ = run([(1, a)], SimpleNamespace(targets=labels))
    renamed = {k: v.copy() for k, v in a.items()}; renamed['track_id'] = np.array([111, 7], np.int64)
    y, _, sy, ey, _ = run([(1, renamed)], SimpleNamespace(targets={
        (1, 111): ('Known', 0), (1, 7): ('policy_pseudo', 0)}))
    assert sx == sy
    assert [e.events for e in ex] == [e.events for e in ey]
    assert all(torch.equal(v, y.state_dict()[k]) for k, v in x.state_dict().items())
    future = {k: v.copy() for k, v in a.items()}; future['visual'][1] = 0; future['visual'][1, 7] = 1
    _, _, _, ez, _ = run([(1, future)], SimpleNamespace(targets=labels))
    assert [e.events for e in ex[:3]] == [e.events for e in ez[:3]]


def test_registered_ten_step_pilot_is_not_the_final_scientific_training_window():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json').read_text())
    assert c['optimizer_steps'] == 10 and c['seed'] == 1027 and c['K_per_family'] == 8
    assert c['policy_input_qualification_sha256'] == 'bdc762881ad0656115ac118e8f9c6d538d749803147ee7e181ce2d5fb5719993'
    assert not c['development_final_Val_Test_or_nonKnown_supervision']
    assert not c['full_scientific_fit_automatic_start']
    assert not c['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_acceptance']
