import hashlib
import json
from pathlib import Path
from src.trackocd_cvpr.M_all_model_fingerprint_development_profile import verify_all_model_shadow

ROOT = Path(__file__).resolve().parents[2]


def test_actual_whole_all11_normal_cpu_shadow_qualifies_same_full_development_profile():
    proof = verify_all_model_shadow(ROOT)
    assert proof['actual_entire_ALL11_weighted_M_support_fingerprint_shadow_checked']
    assert not proof['full_development_parity_speedup_or_partial_metric_reuse_claim']
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_all_model_fingerprint_shadow_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '9b552ebd1f3919d2f697df034aa3843d4a98fa39b08ea8be53716a52b52f595f'
    assert hashlib.sha256((out / 'execution_profile.json').read_bytes()).hexdigest() == 'e219dccd1108c7ede224dbd3ce9de9f1c02e8dbdbdffad67ecd66bc105234688'
    r = json.loads((out / 'result.json').read_text()); e = json.loads((out / 'execution_profile.json').read_text())
    assert len(r['cases']) == len(e['ordered_all11_scope_counters']) == 11
    assert r['optimizer_steps'] == 0 and r['resources']['training_GPU_hours'] == 0
    assert not r['resources']['CUDA_initialized'] and not e['full72_development_speedup_purity_or_Goal_accepted']
    for row in e['ordered_all11_scope_counters']:
        assert row['fingerprint_hits'] > 0
        assert row['Raw_statistics_bytes'] <= 33554432 and row['fingerprint_bytes'] <= 33554432
