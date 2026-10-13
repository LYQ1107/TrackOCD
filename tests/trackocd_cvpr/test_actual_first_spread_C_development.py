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


@pytest.mark.parametrize('seed,hashes', [
    (1027, ('c6a2cd83ddbc5bcec3f0ba73c5d06fda4eb10d67570ba7b7c3038ebf1d2d84db',
            'd987a86db32d2e933f9eb91250592839829569457d4104b4f7bb11c08373acef',
            '23d4aaf1ec267fc184585967f4c446f704316da5c1199bfc69dd2085527bb1ea')),
    (1028, ('ee3e59c48f4f8263577bce41b691ed3e7fe76702d33c70a252c3a1636e7f2b2c',
            '0119f5bf46e077b5aadbd9c4db225af1ab13a37bec4283bdeae335b0abd852e3',
            'a56b5d6df269f5a6043ee1f51e8dca81fc9db5731b4b376a61459f53428e8e4c')),
])
def test_actual_repairs_full_normal_results_have_no_aggregate_discovery_gain(seed, hashes):
    paths = trial_sources('Known_set_rejection',seed)
    assert all(sha256_file(ROOT/p) == h for p,h in zip(paths,hashes))
    base = verified_trial(ROOT,'baseline',seed)
    repair = verified_trial(ROOT,'Known_set_rejection',seed)
    assert repair['orders'] == base['orders']
    assert repair['evaluated_model_parameter_sha256'] != base['evaluated_model_parameter_sha256']
    with pytest.raises(ValueError,match='ALL6'):
        aggregate_matrix({('baseline',seed):base,('Known_set_rejection',seed):repair})


@pytest.mark.parametrize('seed', [1027,1028])
def test_public_pair_summary_is_exact_original_aggregate_not_trace_parity_or_full6_acceptance(seed):
    p = ROOT/f'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_full_dev_pair_s{seed}_public_summary_v1.json'
    r = json.loads(p.read_text())
    field = 'completed_arm_seed_trials' if seed == 1027 else 'completed_arm_seed_trials_in_this_pair'
    assert r[field] == 2 and r['planned_arm_seed_trials'] == 6
    if seed == 1028:
        assert r['completed_protocol_trials_at_publication'] == 4
        partial = {(a,s):verified_trial(ROOT,a,s) for s in (1027,1028) for a in ('baseline','Known_set_rejection')}
        with pytest.raises(ValueError,match='ALL6'):
            aggregate_matrix(partial)
    assert r['incomplete_full6_aggregation_refused'] and r['all_three_order_public_aggregate_rows_equal']
    assert not r['numeric_scores_parameters_or_full_private_prediction_traces_equal_claimed']
    assert not r['new_private_weights_GT_or_data_loaded']
    assert not r['full6matrix_Goal_or_calibration_accepted']
    for path,h in r['input_sha256'].items():
        assert sha256_file(ROOT/path) == h
    for arm,field in (('baseline','all_available_baseline'),('Known_set_rejection','all_available_repair')):
        actual = verified_trial(ROOT,arm,seed)
        for order,row in actual['orders'].items():
            assert r['orders'][order][field] == row['prefix_snapshots']['all_available']
            assert all(v == 0 for v in r['orders'][order]['all_available_repair_minus_baseline'].values())


@pytest.mark.parametrize('seed,hashes', [
    (1028, ('d92a785856b3c4877f269854cfc88745609ab55abab6869253a7e75fdbfd67f6',
            '8b574f7887b66394d1dc4286e5a65d72380b66f7a1757af043e116a09d199fc9',
            '39a9adba6b9d0150bc24197b449dfd2ecaf3c3a5b438376ee697d061cd5f787a')),
    (1029, ('196a9d15dce7919d45f545337092c447b02169d4aa2420aef4dcc9200eb35b4b',
            '51263a725782c87c0e34b7c0e36fb4497fd9dc0f906e9bef71477706840bfdcf',
            'a2131207ee725cdc290ce2c2370f19b15fa5c9c3c6225c1d8719f96a6966013b')),
])
def test_actual_baselines_full_normal_same_public_Known_only_outcomes_not_complete_matrix(seed, hashes):
    paths = trial_sources('baseline',seed)
    assert all(sha256_file(ROOT/p) == h for p,h in zip(paths,hashes))
    r = verified_trial(ROOT,'baseline',seed)
    previous = verified_trial(ROOT,'baseline',1027)
    assert r['evaluated_model_parameter_sha256'] != previous['evaluated_model_parameter_sha256']
    assert r['orders'] == previous['orders']
    assert r['actual_future_probe'] == {'future_observations_mutated':39,'verified_same_prior_events':True}
    table = {('baseline',1027):previous, ('baseline',seed):r,
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
