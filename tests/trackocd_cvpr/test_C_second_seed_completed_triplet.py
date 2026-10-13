import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_C_pooled_second_seed_full_replay_normal_terminal_original_failure_preserved():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_registration_retry1_Top8_pool_context_s1028'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '7dff3f6168077f2078fe99353835e840cef26a4406cd6b8bbe883bc10c60e2d7'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top8_pool_context' and r['training_seed'] == 1028
    assert r['actual_full_ID_rename_reverse_stream_equal']
    assert r['actual_future_probe'] == {'future_observations_mutated': 39, 'verified_same_prior_events': True}
    for row in r['orders'].values():
        assert (row['observations'], row['metadata_frames'], row['empty_frames']) == (8753, 2562, 116)
        assert row['physical_actions'] == {'KNOWN': 8753}
        end = row['prefix_snapshots']['all_available']
        assert end['Known_micro_accuracy'] == pytest.approx(282 / 302)
        assert end['physical_tracks'] == 472 and end['declared_pseudo_queries'] == 119
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 119
        assert end['declared_pseudo_anonymous_commit_queries'] == end['certified_cross_video_reuse_micro_lower_bound'] == 0
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    old = out.with_name('C_GT_causal_dev_v1_Top8_pool_context_s1028_supervisor_attempt_1.json')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == 'e6315ce9eb14c5b583c8ecfe9d3e46a6a7eb8abb13e72e342e1c78a9292a987c'
    assert not json.loads(old.read_text())['successful_owned_execution']
