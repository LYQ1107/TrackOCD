import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr import M_member_input_development_profile as profile
from scripts.trackocd_cvpr import evaluate_M_development as original

ROOT = Path(__file__).resolve().parents[2]


def test_new_namespace_retains_full_data_models_probes_and_original_resource_bounds():
    args = SimpleNamespace(model='relative_quarantine', seed=1027)
    base = original.configuration(ROOT, args)
    new = profile.configuration(ROOT, base, args)
    assert new['output_directory'].endswith('member_input_memo_v1_relative_quarantine_s1027')
    assert {k: v for k, v in new.items() if k not in ('output_directory', 'output_prefix')} == {
        k: v for k, v in base.items() if k not in ('output_directory', 'output_prefix')}
    assert new['wall_seconds_ceiling'] == 3000 and new['physical_tracks'] == 472


def test_pending_all11_proof_fails_before_private_or_old_qualification_lookup(monkeypatch,tmp_path):
    cfg = json.loads((ROOT / profile.CONFIG).read_text())
    cfg['shadow_result_sha256'] = cfg['shadow_execution_profile_sha256'] = cfg['shadow_terminal_sha256'] = None
    for path, text in ((profile.CONFIG, json.dumps(cfg)),
            (cfg['original_config'], (ROOT / cfg['original_config']).read_text())):
        p = tmp_path / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    def forbidden(*a, **k): raise AssertionError('No private input, networking or shadow fallback before real proof')
    monkeypatch.setattr(profile, 'configuration_gate', forbidden)
    monkeypatch.setattr(profile, 'verify_all_model_shadow', forbidden)
    with pytest.raises(ValueError, match='fingerprints pending'): profile.verify_member_input_shadow(tmp_path)


@pytest.mark.parametrize('field,value', [
    ('models_orders_prefixes_thresholds_history_or_guards_changed', True),
    ('partial_metric_reuse_model_selection_or_full_Goal_acceptance', True),
    ('output_prefix', 'outputs/old'), ('shadow_preregistration_commit', '0' * 40),
    ('policy_parity_is_not_full_development_parity_or_speedup_guarantee', False),
])
def test_config_never_converts_partial_failure_into_new_science(tmp_path,field,value):
    cfg = json.loads((ROOT / profile.CONFIG).read_text()); cfg[field] = value
    p = tmp_path / profile.CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): profile.configuration(tmp_path, {}, SimpleNamespace(model='relative_quarantine',seed=1027))


def test_original_worker_and_full_probes_are_composed_without_reduced_scope():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_member_input_memo.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source
    assert 'original.supervise(args, cfg, safety)' in source
    assert 'original_gate(args, cfg, safety)' in source
    assert 'verify_member_input_shadow(ROOT)' in source
    assert 'execution_profile.json' in source


def test_actual_all11_complete_input_cache_parity_is_not_full_development_or_goal():
    proof = profile.verify_member_input_shadow(ROOT)
    assert proof['actual_ALL11_member_input_and_current_refresh_index_parity_checked']
    assert proof['member_input_shadow_result_sha256'] == 'cc9f3c704a860f62f56bbfe927c830b7e1c3bdd2967b405174549e40bdd33e33'
    assert not proof['full_dev_or_speedup_or_Goal_accepted']
