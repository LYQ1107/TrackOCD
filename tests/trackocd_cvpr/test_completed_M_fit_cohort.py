import hashlib
import json
from pathlib import Path
from src.trackocd_cvpr.M_development_cohort import verified_cohort, VARIANTS, SEEDS

ROOT = Path(__file__).resolve().parents[2]


def test_actual_last_M_fit_and_all9_endpoint_sources_initializations_normal_terminals():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_pooled_weighted_s1029'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'a3334abec084c2723ab46ea98449916beae30f31ded462b8fec10da9dfb5027b'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    assert (r['variant'], r['seed'], s['optimizer_steps'], s['trainable_parameters']) == ('pooled_weighted', 1029, 200, 161)
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '123dca39f04c5bd008a39152e9d706dc9407586f60cc878971d3c929f027e5f7', 'reopen_exact': True}
    assert s['actual_optimizer_family_groups'] == {'Known': 576, 'policy_pseudo': 114}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert s['actual_optimizer_target_bins'] == {'fractional': 20, 'one': 434, 'zero': 236}
    assert s['episodes'][-1]['external_strict_version_lineage']['strict_latest_version_status_counts']['VERIFIED_BAD_MIXED_PSEUDO_HISTORY'] == 1
    cohort = verified_cohort(ROOT)
    assert set(cohort) == {(v, seed) for v in VARIANTS for seed in SEEDS}
    initial = {}
    for (variant, seed), (path, receipt) in cohort.items():
        assert path.is_file() and receipt['summary']['optimizer_steps'] == 200
        before = receipt['summary']['model_parameter_sha256_before']
        assert seed not in initial or initial[seed] == before
        initial[seed] = before
    assert len(set(initial.values())) == 3
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
