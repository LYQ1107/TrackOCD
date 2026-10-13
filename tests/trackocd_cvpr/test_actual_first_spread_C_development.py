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


def test_actual_seed1027_repair_full_normal_result_has_no_aggregate_discovery_gain():
    paths = trial_sources('Known_set_rejection',1027)
    hashes = ('c6a2cd83ddbc5bcec3f0ba73c5d06fda4eb10d67570ba7b7c3038ebf1d2d84db',
              'd987a86db32d2e933f9eb91250592839829569457d4104b4f7bb11c08373acef',
              '23d4aaf1ec267fc184585967f4c446f704316da5c1199bfc69dd2085527bb1ea')
    assert all(sha256_file(ROOT/p) == h for p,h in zip(paths,hashes))
    base = verified_trial(ROOT,'baseline',1027)
    repair = verified_trial(ROOT,'Known_set_rejection',1027)
    assert repair['orders'] == base['orders']
    assert repair['evaluated_model_parameter_sha256'] != base['evaluated_model_parameter_sha256']
    with pytest.raises(ValueError,match='ALL6'):
        aggregate_matrix({('baseline',1027):base,('Known_set_rejection',1027):repair})


def test_public_pair_summary_is_exact_original_aggregate_not_trace_parity_or_full6_acceptance():
    p = ROOT/'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_full_dev_pair_s1027_public_summary_v1.json'
    r = json.loads(p.read_text())
    assert r['completed_arm_seed_trials'] == 2 and r['planned_arm_seed_trials'] == 6
    assert r['incomplete_full6_aggregation_refused'] and r['all_three_order_public_aggregate_rows_equal']
    assert not r['numeric_scores_parameters_or_full_private_prediction_traces_equal_claimed']
    assert not r['new_private_weights_GT_or_data_loaded']
    assert not r['full6matrix_Goal_or_calibration_accepted']
    for path,h in r['input_sha256'].items():
        assert sha256_file(ROOT/path) == h
    for arm,field in (('baseline','all_available_baseline'),('Known_set_rejection','all_available_repair')):
        actual = verified_trial(ROOT,arm,1027)
        for order,row in actual['orders'].items():
            assert r['orders'][order][field] == row['prefix_snapshots']['all_available']
            assert all(v == 0 for v in r['orders'][order]['all_available_repair_minus_baseline'].values())


def test_actual_baseline1028_full_normal_same_public_Known_only_outcomes_not_complete_matrix():
    paths = trial_sources('baseline',1028)
    hashes = ('d92a785856b3c4877f269854cfc88745609ab55abab6869253a7e75fdbfd67f6',
              '8b574f7887b66394d1dc4286e5a65d72380b66f7a1757af043e116a09d199fc9',
              '39a9adba6b9d0150bc24197b449dfd2ecaf3c3a5b438376ee697d061cd5f787a')
    assert all(sha256_file(ROOT/p) == h for p,h in zip(paths,hashes))
    r = verified_trial(ROOT,'baseline',1028)
    previous = verified_trial(ROOT,'baseline',1027)
    assert r['evaluated_model_parameter_sha256'] != previous['evaluated_model_parameter_sha256']
    assert r['orders'] == previous['orders']
    assert r['actual_future_probe'] == {'future_observations_mutated':39,'verified_same_prior_events':True}
    table = {('baseline',1027):previous, ('baseline',1028):r,
             ('Known_set_rejection',1027):verified_trial(ROOT,'Known_set_rejection',1027)}
    with pytest.raises(ValueError,match='ALL6'):
        aggregate_matrix(table)


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
