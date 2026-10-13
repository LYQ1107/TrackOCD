import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_third_C_seed_Top8_opposite_anonymous_collapse_not_a_novel_win():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_conditioned_s1029'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '6f29e226715ac307022955ddfbff9f3066e73acef1baa45612e24019c349045a'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top8_conditioned' and r['training_seed'] == 1029
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe']['verified_same_prior_events']
    for seed, correct in (('1027', 9), ('1028', 7), ('1029', 4)):
        row = r['orders'][seed]; end = row['prefix_snapshots']['all_available']
        assert row['physical_actions'].get('KNOWN', 0) == 0
        assert sum(row['physical_actions'].values()) == 8753
        assert end['Known_micro_accuracy'] == end['Known_macro_accuracy'] == 0
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 0
        assert end['declared_pseudo_anonymous_commit_queries'] == 119
        assert end['certified_cross_video_reuse_micro_lower_bound'] == pytest.approx(correct / 119)
        assert end['physical_tracks'] == 472 and end['all_physical_commit_track_coverage'] == 1
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
