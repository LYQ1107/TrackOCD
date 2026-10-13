import hashlib
import json
from pathlib import Path
from src.trackocd_cvpr.M_cached_development_profile import verify_actual_shadow

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all11_policy_controller_shadow_matches_full_events_scores_history_and_normal_wait():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_raw_cache_controller_shadow_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '3cff459de63e60b6b9a7797c73af427c08f34879dbaab6244f10b77410bf8156'
    r = json.loads((out / 'result.json').read_text())
    assert len(r['cases']) == 11 and r['physical_tracks_per_case'] == 54 and r['observations_per_replay'] == 1268
    for case in r['cases'].values():
        assert case['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
        assert len(case['private_trace_sha256']) == 64 and case['retained_derived_cache_bytes'] <= 67108864
        assert case['timing_is_shared_host_sequential_pair_not_isolated_speedup_claim']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['wall_seconds_including_registration'] <= 3000
    q = verify_actual_shadow(ROOT)
    assert q['actual_all11_policy_controller_shadow_checked']
    assert not q['original_full72_Mean_parity_or_partial_metric_reuse_claim']
    assert r['optimizer_steps'] == 0 and not r['full72development_or_Pred_purity_or_Goal_accepted']
