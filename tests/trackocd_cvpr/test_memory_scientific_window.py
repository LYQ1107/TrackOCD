import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.memory_scientific_training import scientific_memory_fit
from scripts.trackocd_cvpr.pilot_memory_optimizer import CONFIG, SCIENTIFIC_CONFIG, validate_configuration
from tests.trackocd_cvpr.test_memory_fit_pilot import synthetic_policy_clock

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_M_matrix_fresh_seeds_controls_steps_and_pilot_limit():
    cfg = json.loads((ROOT / SCIENTIFIC_CONFIG).read_text())
    assert cfg['training_seed_replicates'] == [1027, 1028, 1029]
    assert set(cfg['variants']) == {'relative_quarantine', 'relative_weighted', 'pooled_weighted'}
    assert cfg['training']['episodes'] * cfg['training']['updates_per_episode'] == cfg['training']['optimizer_steps'] == 200
    assert cfg['training']['checkpoint_steps'] == [20, 50, 100, 200]
    assert cfg['trainable_parameters_each'] == 161 and 'ONEKnownclass' in cfg['pilot_limitation']
    assert not cfg['full_scientific_fit_automatic_start'] and not cfg['three_seed_M_qualified_Pred_joint_method_or_Goal_accepted']
    assert not cfg['memory']['strict_counts_are_official_CT_or_full_semantic_GT_or_self_excluded_reuse']
    assert 'EMA' in cfg['baseline_status']  # still executable qualification pending
    initial = []
    for seed in cfg['training_seed_replicates']:
        paired = []
        for settings in cfg['variants'].values():
            torch.manual_seed(seed); model = MemberReliabilityMLP(pooled_control=settings['pooled_control'])
            paired.append(torch.cat([p.flatten() for p in model.parameters()]).detach())
        assert all(torch.equal(paired[0], p) for p in paired[1:])
        initial.append(paired[0])
    assert all(not torch.equal(initial[i], initial[j]) for i in range(3) for j in range(i))


def test_real_kernel_synthetic_fixed200_complete_episodes_and_strict_pollution():
    # Pure synthetic support proves execution, not actual M fit/generalization.
    cfg = json.loads((ROOT / SCIENTIFIC_CONFIG).read_text())
    videos, targets = synthetic_policy_clock(); saves = []
    model, opt, summary, private = scientific_memory_fit(videos, SimpleNamespace(targets=targets), {}, cfg,
        'relative_quarantine', 1027, lambda: None, lambda row, curve: None,
        lambda current, optimizer, step, ep, epstep, rng: saves.append((step, ep, epstep)))
    assert summary['optimizer_steps'] == 200 and summary['trainable_parameters'] == 161
    assert [s[0] for s in saves] == [20, 50, 100, 200] and len(private) == 4
    assert summary['model_parameter_sha256_before'] != summary['model_parameter_sha256_after']
    for episode in summary['episodes']:
        assert (episode['videos'], episode['metadata_frames'], episode['physical_observations'], episode['empty_frames']) == (10, 374, 1268, 1)
        assert episode['optimizer_steps'] == 50
        # Even confident homogeneous Known support is BAD anonymous history,
        # not pseudo discovery or purification by lifecycle promotion.
        assert episode['external_strict_version_lineage']['strict_latest_version_status_counts'] == {
            'VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY': 1}
    assert not summary['Novel_gain_official_CT_current_active_purity_self_exclusion_Pred_or_Goal_accepted']


def test_scientific_supervisor_forwards_configuration_variant_and_seed(tmp_path, monkeypatch):
    from scripts.trackocd_cvpr import pilot_memory_optimizer as script
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'resources_ok', lambda *args: None)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda cmd, **kw: (commands.append(cmd), Child())[1])
    args = SimpleNamespace(preregistration_commit='source', git_proxy='', variant='pooled_weighted', seed=1029)
    assert script.supervise(args, {'output_directory': 'new_owned', 'scientific_M_trial': True}, {}) == 0
    cmd = commands[0]
    assert cmd[cmd.index('--config') + 1] == SCIENTIFIC_CONFIG
    assert cmd[cmd.index('--variant') + 1] == args.variant
    assert cmd[cmd.index('--seed') + 1] == '1029'


def test_mismatched_lifecycle_manifest_or_update_budget_refused():
    base = json.loads((ROOT / CONFIG).read_text()); window = json.loads((ROOT / SCIENTIFIC_CONFIG).read_text())
    cfg = {**base, **window, **window['training'], **window['proposed_execution'],
           'scientific_M_trial': True, 'variant': 'relative_quarantine', 'seed': 1027}
    authority = {'status': 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'}
    validate_configuration(cfg, authority)
    cfg['memory']['gates']['quarantine'] = .9
    with pytest.raises(ValueError): validate_configuration(cfg, authority)
