import hashlib
import json
from pathlib import Path
from src.trackocd_cvpr.M_terminal_replica_recovery import ORIGINAL, REPLICA, verify_replica_result

ROOT = Path(__file__).resolve().parents[2]


def test_actual_completed_replica_has_exact_private_optimizer_rng_endpoint_history_and_new_normal_wait():
    out = ROOT / REPLICA
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'ab7d8a292ac15be45ccb427e1f5afba69123e0dbb1a100dd95df1e04f8cc5a4f'
    r = json.loads((out / 'result.json').read_text())
    assert r['execution_transport_replica'] == verify_replica_result(ROOT, r)
    assert r['summary']['optimizer_steps'] == 200
    assert r['endpoint']['sha256'] == 'd3ead779d3d1d6df86f22dde6242b5daa0cd972438324eb82b04b8928f9759b6'
    assert r['private_prediction_history']['sha256'] == '51d96c16389c57102a2d50da81c368ea08789d25224fd119d7275fd956088f60'
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['last_persisted_completed_optimizer_steps'] == 200
    assert not (ROOT / (ORIGINAL + '_supervisor_attempt_1.json')).exists()
