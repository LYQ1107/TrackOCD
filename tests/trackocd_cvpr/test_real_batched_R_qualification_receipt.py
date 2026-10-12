import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/fit_R_batched_qualification_v1'


def test_actual_fullsupport_trained_R_batch_parity_without_optimizer_or_semantic_expansion():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '83a9dfc9a3cb8ace64a87af6e08b4d0486cbe92deae2e69dfed5b223718dcda9'
    r = json.loads(p.read_text())
    assert r['live_remote_registration_proof']['sha'] == '14d06436782151aa8ea4cdd3869a88f0c5f5d708'
    assert r['support_members'] == 1149 and r['support_observations'] == 22066
    assert r['forward_and_statistics_max_absolute_error'] <= 1e-5
    assert r['parameter_gradients_max_absolute_error'] <= 2e-4
    assert r['model_parameter_sha256_unchanged'] == 'c7f17ae79be3a57abf3d87f546ab3b8291aec4a93c4e098de75b4dc811574ca3'
    assert r['optimizer_steps'] == 0 and not r['dev_final_labels_pixels_originalGT_DINOweights_read']
    assert not r['trained_scientific_mechanism_or_Goal_A_B_C_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
