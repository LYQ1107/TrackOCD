import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_transport_retry1_Raw_Top1'


def test_actual_Raw_causal_full_dev_and_invariance_complete_without_selection():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == 'e63831f1f6485ced2f09c164099eafc1674f589c7b322b05503366788d454b89'
    r = json.loads((OUT / 'result.json').read_text())
    assert r['model'] == 'Raw_Top1' and r['training_seed'] is None
    assert r['development_scope']['physical_tracks'] == 472 and r['optimizer_steps'] == 0
    assert r['actual_full_ID_rename_reverse_stream_equal']
    assert r['actual_future_probe'] == {'future_observations_mutated': 39, 'verified_same_prior_events': True}
    for order, known_correct, certified_reuse in (('1027', 12, 13), ('1028', 6, 8), ('1029', 4, 13)):
        row = r['orders'][order]; end = row['prefix_snapshots']['all_available']
        assert (row['metadata_frames'], row['empty_frames'], row['observations']) == (2562, 116, 8753)
        assert end['Known_queries'] == 302 and end['declared_pseudo_queries'] == 119
        assert end['Known_micro_accuracy'] == pytest.approx(known_correct / 302)
        assert end['certified_cross_video_reuse_micro_lower_bound'] == pytest.approx(certified_reuse / 119)
        assert end['all_physical_commit_track_coverage'] == 1
        assert row['first_Known_anonymous_write_canonical_group'] == 0
        assert row['strict_route_reassignment_history']['verified_non_pseudo_or_mixed_history'] >= 146
        assert not row['unknown_declared_background_or_Novel']
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_actual_RawTop8_all_order_results_equal_RawTop1_not_a_K_expansion_gain():
    out = OUT.with_name('C_GT_causal_dev_v1_Raw_Top8')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'e6fae9929d29358da18005fe52833c96de3a8abe61f50d8160665fcba9d99320'
    r = json.loads((out / 'result.json').read_text()); first = json.loads((OUT / 'result.json').read_text())
    assert r['model'] == 'Raw_Top8' and r['training_seed'] is None
    assert r['orders'] == first['orders']
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe'] == first['actual_future_probe']
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
