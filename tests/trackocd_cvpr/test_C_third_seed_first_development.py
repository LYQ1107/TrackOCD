import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_third_C_seed_Top1_allorders_small_anonymous_coverage_without_certified_reuse():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top1_conditioned_s1029'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'f8890293a7d37436c8c06a0cee05c190f1fef759adbf46d24261b5229b088645'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top1_conditioned' and r['training_seed'] == 1029
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe']['verified_same_prior_events']
    assert r['actual_future_probe']['future_observations_mutated'] == 39
    for row in r['orders'].values():
        assert row['physical_actions'] == {'EXISTING': 147, 'KNOWN': 8593, 'NEW': 13}
        end = row['prefix_snapshots']['all_available']
        assert end['Known_micro_accuracy'] == pytest.approx(281 / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 117
        assert end['declared_pseudo_anonymous_commit_queries'] == 2
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['physical_tracks'] == 472 and end['declared_pseudo_queries'] == 119
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
