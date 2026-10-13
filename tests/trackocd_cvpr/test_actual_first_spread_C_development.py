import json
from pathlib import Path
import pytest
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.spread_C_complete_fit_cohort import verified_cohort
from src.trackocd_cvpr.spread_C_development_complete_matrix import verified_trial, verified_matrix, aggregate_matrix, trial_sources

ROOT = Path(__file__).resolve().parents[2]
HASHES = dict(zip(trial_sources('baseline',1027), (
    '46dd78f69dffa7e773c0ab914793f8c9ddfa32987ef459f0a3b597d0f1d34e73',
    'e8e0782ea78c4bb4e124932cf58719f279862e7cb8efeb13fbe9dfad51450314',
    '9e8cc0807e2cde5c791ab29ceeda6a38ff00a4bd1c7385534ca346d9ad4e0bc0')))


def test_one_actual_full_normal_baseline_dev_not_repaired_arm_or_full_matrix_acceptance():
    for p,h in HASHES.items():
        assert sha256_file(ROOT/p) == h
    r = verified_trial(ROOT,'baseline',1027)
    assert r['actual_future_probe'] == {'future_observations_mutated':39, 'verified_same_prior_events':True}
    for row in r['orders'].values():
        end = row['prefix_snapshots']['all_available']
        assert row['physical_actions'] == {'KNOWN':8753}
        assert end['Known_micro_accuracy'] == 282/302
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 119
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['all_physical_commit_track_coverage'] == 1
    with pytest.raises(ValueError,match='ALL6'):
        aggregate_matrix({('baseline',1027):r})


def test_missing_full_matrix_or_trial_refuses_without_dev_or_weight_reads(tmp_path):
    with pytest.raises(ValueError,match='Complete original'):
        verified_trial(tmp_path,'baseline',1027)
    with pytest.raises(ValueError,match='Complete original'):
        verified_matrix(tmp_path)


@pytest.mark.parametrize('change',['no_probe','vacuous_future','failed_terminal','selected_model'])
def test_public_partial_or_relaxed_completion_refused_before_private_inputs(tmp_path,monkeypatch,change):
    import src.trackocd_cvpr.spread_C_development_complete_matrix as module
    public_cohort = verified_cohort(ROOT)
    monkeypatch.setattr(module,'verified_cohort',lambda root:public_cohort)
    for path in HASHES:
        p = tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/path).read_bytes())
    path = trial_sources('baseline',1027)[2 if change == 'failed_terminal' else 1]
    p = tmp_path/path; r = json.loads(p.read_text())
    if change == 'no_probe':
        r['actual_full_ID_rename_reverse_stream_equal'] = False
    elif change == 'vacuous_future':
        r['actual_future_probe']['future_observations_mutated'] = 0
    elif change == 'failed_terminal':
        r['successful_owned_execution'] = False
    else:
        r['model_or_checkpoint_selection'] = True
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Actual complete causal'):
        verified_trial(tmp_path,'baseline',1027)
