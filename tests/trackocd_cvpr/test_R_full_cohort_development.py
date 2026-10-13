import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.R_development_cohort import VARIANTS, SEEDS, cohort_sources, verified_cohort
from src.trackocd_cvpr.R_development_probes import numeric_probes
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.batched_evidence import TimelessStaticEvidence
from tests.trackocd_cvpr.test_fit_R_training import track

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all12_complete_before_privateGT_with_real_source_endpoint_checks():
    assert len(cohort_sources()) == 36 and len(set(cohort_sources())) == 36
    cohort = verified_cohort(ROOT)
    assert len(cohort) == 12 and set(cohort) == {(v, s) for v in VARIANTS for s in SEEDS}
    for (_, seed), (_, r) in cohort.items():
        assert r['seed'] == seed and r['completed_optimizer_steps'] == 200
    # This is metadata/own endpoint fingerprinting, NOT dev-label read or acceptance.


def test_missing_endpoint_cohort_blocks_before_private_data(tmp_path):
    with pytest.raises(FileNotFoundError): verified_cohort(tmp_path)


def test_actual_numeric_future_and_timeless_order_time_probes_no_fit_or_GT():
    rows = [track(1, None, 1), track(2, None, 2)]
    for model, timeless in ((RawPreservingEvidence(), False), (TimelessStaticEvidence(), True)):
        model.eval().requires_grad_(False)
        r = numeric_probes(model, rows, lambda: None, timeless=timeless)
        assert r['future_rows_mutated'] == 2 and r['actual_first_prefix_future_invariant']
        assert not r['temporal_information_benefit_or_scientific_acceptance']
        if timeless:
            assert r['timeless_exact_order_and_time_invariance_required_and_passed']
            assert r['order_embedding_l2_change'] == r['time_embedding_l2_change'] == 0


def test_registered_full_matrix_has_no_seed_prefix_or_checkpoint_selector():
    c = json.loads((ROOT / 'configs/trackocd_cvpr/R_development_all12_v1.json').read_text())
    assert c['models'] == list(VARIANTS) and c['seeds'] == list(SEEDS)
    assert c['checkpoint_step'] == 200 and c['optimizer_steps'] == 0
    assert c['prefixes'] == [1, 2, 4, 8, 16, 'all_available']
    assert not c['select_model_or_checkpoint'] and not c['all_R_C_M_scientific_qualified_Pred_or_Goal_acceptance']


def test_full_R_supervisor_forwards_config_not_legacy_firstpair(tmp_path, monkeypatch):
    from scripts.trackocd_cvpr import evaluate_R_development as script
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'resources_ok', lambda *args: None)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda cmd, **kw: (commands.append(cmd), Child())[1])
    assert script.supervise(SimpleNamespace(preregistration_commit='source', git_proxy=''),
        {'output_directory': 'new_R_full', 'full_matrix': True}, {}) == 0
    assert commands[0][commands[0].index('--config') + 1] == script.FULL_CONFIG
