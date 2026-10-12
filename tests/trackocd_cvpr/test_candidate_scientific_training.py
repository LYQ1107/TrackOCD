from types import SimpleNamespace
import numpy as np
import torch

from src.trackocd_cvpr.candidate_scientific_training import scientific_candidate_fit, variant_numeric_features
from tests.trackocd_cvpr.test_policy_training_data import arrays


def trial(variant='Top8_conditioned', rows=None):
    rows = arrays() if rows is None else rows
    projection = SimpleNamespace(targets={(1, 12): ('Known', 0), (1, 99): ('policy_pseudo', 0)})
    window = {'variants': {variant: {'K_per_family': 1 if variant.startswith('Top1') else 8}},
              'training': {'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 1.,
                           'episodes': 2, 'updates_per_episode': 1, 'optimizer_steps': 2, 'checkpoint_steps': [1, 2]}}
    checkpoints = []
    result = scientific_candidate_fit([(1, rows)], projection, {0: np.eye(1, 768, 0, dtype=np.float32)[0]},
        window, variant, 1027, lambda: None, lambda r, c: None,
        lambda m, o, step, e, s, rng: checkpoints.append(step))
    assert checkpoints == [1, 2]
    return result


def test_actual_synthetic_multiepisode_fits_full_input_and_optimizer_family_not_tail_counts():
    _, _, summary, private = trial()
    assert summary['optimizer_steps'] == 2 and summary['trainable_parameters'] == 708
    assert len(private) == len(summary['episodes']) == 2
    assert summary['actual_optimizer_family_groups'] == {'Known': 2, 'policy_pseudo': 2}
    assert summary['both_Known_and_pseudo_received_optimizer_loss']
    assert all(r['metadata_frames'] == 4 and r['physical_observations'] == 3 for r in summary['episodes'])
    assert not summary['GT_repair_or_deleted_wrong_history']
    assert not summary['self_support_exclusion_or_Pred_or_M_or_Goal_acceptance']


def test_pool_context_control_row_permutation_and_nominal_capacity_matching():
    rows = np.arange(33, dtype=np.float32).reshape(3, 11)
    pooled = variant_numeric_features(rows, 'Top8_pool_context')
    np.testing.assert_array_equal(pooled[[2, 0, 1]], variant_numeric_features(rows[[2, 0, 1]], 'Top8_pool_context'))
    assert np.all(pooled[:, 2:5] == pooled[:1, 2:5])
    for variant in ('Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context'):
        model, _, summary, _ = trial(variant)
        assert sum(p.numel() for p in model.parameters()) == summary['trainable_parameters'] == 708


def test_changed_future_does_not_change_retained_pre_future_training_predictions():
    _, _, _, a = trial(); future = arrays(); future['visual'][1] = 0; future['visual'][1, 7] = 1
    _, _, _, b = trial(rows=future)
    # Subsequent episodes can inherit a changed trained model; first episode's
    # sealed pre-future frames cannot. Do not assert causality across retraining.
    assert [r.events for _, r in a[0][1][:3]] == [r.events for _, r in b[0][1][:3]]


def test_frozen_C_matrix_data_steps_capacity_and_pending_evaluation_not_acceptance():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/C_GT_scientific_window_v1.json').read_text())
    assert c['training_seed_replicates'] == [1027, 1028, 1029]
    assert set(c['variants']) == {'Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context'}
    assert c['trainable_parameters_each'] == 708
    assert c['training']['episodes'] * c['training']['updates_per_episode'] == c['training']['optimizer_steps'] == 200
    assert c['training']['checkpoint_steps'] == [20, 50, 100, 200]
    assert not c['full_scientific_fit_automatic_start']
    assert not c['all3seeds_or_M_Pred_or_Goal_acceptance_from_manifest']


def test_supervisor_passes_real_scientific_config_variant_seed_to_owned_child(tmp_path, monkeypatch):
    from scripts.trackocd_cvpr import qualify_policy_training_input as script
    recorded = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'resources_ok', lambda *args: None)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: (recorded.append(command), Child())[1])
    args = SimpleNamespace(config='configs/trackocd_cvpr/C_GT_scientific_window_v1.json',
        variant='Top8_pool_context', seed=1029, preregistration_commit='source-receipt', git_proxy='')
    cfg = {'output_directory': 'new_trial', 'output_bytes_ceiling': 10000}
    assert script.supervise(args, cfg, {}, worker_script='scripts/trackocd_cvpr/pilot_candidate_optimizer.py') == 0
    command = recorded[0]
    assert command[command.index('--config') + 1] == args.config
    assert command[command.index('--variant') + 1] == args.variant
    assert command[command.index('--seed') + 1] == '1029'
