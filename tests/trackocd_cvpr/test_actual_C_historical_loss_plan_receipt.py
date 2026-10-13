import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all9_history_reconstruction_matches_200_updates_without_new_training():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_historical_loss_plan_audit_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '53522917edfe26ec097a511bd88ab28d65db751d034be2b0f9726a9c2ca6b9a3'
    r = json.loads((out / 'result.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert r['live_remote_registration_proof']['sha'] == 'f35c77926e55dc79698808e7c1a870047d23a55b'
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['wall_seconds_including_registration'] <= 3000
    expected = {'Top1_conditioned_s1027': (359, 110, 0), 'Top1_conditioned_s1028': (274, 88, 12),
        'Top1_conditioned_s1029': (210, 163, 22), 'Top8_conditioned_s1027': (538, 82, 24),
        'Top8_conditioned_s1028': (381, 56, 41), 'Top8_conditioned_s1029': (266, 111, 67),
        'Top8_pool_context_s1027': (538, 83, 24), 'Top8_pool_context_s1028': (381, 56, 41),
        'Top8_pool_context_s1029': (266, 103, 80)}
    assert set(r['trials']) == set(expected)
    for name, counts in expected.items():
        row = r['trials'][name]; hist = row['update_positive_plan_groups_by_family_and_kind']
        assert tuple(hist.get(key, 0) for key in ('Known:KNOWN', 'policy_pseudo:NEW', 'policy_pseudo:EXISTING')) == counts
        assert hist.get('Known:WAIT', 0) == hist.get('policy_pseudo:WAIT', 0) == 0
        assert row['historical_optimizer_frames_matched'] == 200 and row['new_optimizer_steps'] == 0
        assert row['all_original_episode_summaries_update_calendars_and_retained_histories_match']
        assert not row['historical_logits_gradients_or_rowwise_numeric_trace_recovered']
        assert len(row['episodes']) == 4
        for episode in row['episodes']:
            assert (episode['optimizer_frames_matched'], episode['metadata_frames_matched'], episode['physical_observations_matched']) == (50, 374, 1268)
    assert r['new_optimizer_steps'] == 0 and not r['new_inference_collapse_cause_development_Pred_or_Goal_accepted']
    assert not r['policy_scope']['development_final_or_other_video_payload_bodies_opened']
    assert not r['resources']['CUDA_initialized'] and r['resources']['training_GPU_hours'] == 0
