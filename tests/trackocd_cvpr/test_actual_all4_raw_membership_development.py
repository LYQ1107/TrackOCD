import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.raw_membership_formal_cohort import TRIALS
from src.trackocd_cvpr import raw_membership_development_receipts as receipts
from src.trackocd_cvpr.raw_membership_control_comparison import complete_comparison

ROOT = Path(__file__).resolve().parents[2]


def forbid_private_stat(monkeypatch):
    original = Path.stat
    def guard(path, *args, **kwargs):
        if path.suffix in ('.pt', '.pth'):raise AssertionError('Public ALL4 comparison touched private model')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', guard)


def test_actual_ONE_fullpolicy_normal_complete_full_orders_probes_no_fake_seeds():
    p = receipts.verified_trial(ROOT, 'full_policy', None)
    assert p['calendar_seed'] is None and p['all3_full_orders_and_complete_probes_verified']
    assert not p['full_deployment_calibration_fair_match_control_winner_or_Goal_accepted']


def test_actual_all4_original_trials_same_full_diagnostic_corpus_and_orders_without_private_load(monkeypatch):
    forbid_private_stat(monkeypatch); p = receipts.verified_matrix(ROOT)
    assert p['all4_actual_complete_dev_trials_verified'] and len(p['trials']) == 4
    assert p['same_actual_initial_Raw_diagnostic_universe_and_video_orders']
    assert not p['fullpolicy_is_three_independent_training_seeds']
    assert not p['deployment_calibration_latency_coverage_winner_or_Goal_accepted']


def test_actual_complete_all4_comparison_keeps_raw_three_calendars_one_solver_and_all6_C_separate(monkeypatch):
    forbid_private_stat(monkeypatch); p = complete_comparison(ROOT)
    assert p['all4_proof']['all4_actual_complete_dev_trials_verified']
    assert set(p['calendar_matched']['per_calendar_seed_order_mean']) == {'1027', '1028', '1029'}
    assert not p['ONE_deterministic_fullpolicy']['independent_training_seed_runs']
    assert len(p['independent_conditional_reliability_per_endpoint']) == 4
    assert set(p['ALL6_previous_paired_C_comparison']['arms']) == {'baseline', 'Known_set_rejection'}
    assert p['calendar_matched_to_C_means_same_actual_frame_updates_not_capacity_targets_or_gradient_exposure']
    assert not p['fullpolicy_is_calendar_or_optimizer_matched_to_C']
    assert not p['control_seed_order_prefix_bin_threshold_or_checkpoint_selected']
    assert not p['full_semantic_GT_Hscore_CT_R_M_Pred_dense_lockedfinal_or_Goal_accepted']
    for k in ('fixed_Raw_zero_fit_order_mean',):
        assert set(p[k]) == {'1', '2', '4', '8', '16', 'all_available'}


def test_actual_published_aggregate_exactly_reduces_original_complete_public_trials(monkeypatch):
    forbid_private_stat(monkeypatch)
    path = ROOT/'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_all4_full_dev_comparison_v1.json'
    report = json.loads(path.read_text()); actual = complete_comparison(ROOT)
    assert {k:report[k] for k in actual} == actual
    terminals = [json.loads((ROOT/receipts.trial_sources(c,s)[2]).read_text()) for c,s in TRIALS]
    resources = report['actual_all4_evaluation_resources']
    assert resources['total_owned_wall_seconds'] == sum(t['wall_seconds_including_worker_imports_network'] for t in terminals)
    assert resources['max_sampled_owned_RSS_bytes'] == max(t['peak_sampled_owned_worker_RSS_bytes'] for t in terminals)
    assert resources['all_owned_normal_exits'] and resources['training_GPU_hours'] == 0


@pytest.mark.parametrize('mutation,expected', [
    ('initial_task', 'Same complete group initial Raw'), ('source', 'Same actual frozen numerical/source'),
    ('clock', 'Same full physical diagnostic universe'), ('ordinal', 'Same actual video ordinal schedules'),
])
def test_cross_trial_source_corpus_clock_and_actual_schedule_drift_not_silently_aggregated(tmp_path, monkeypatch, mutation, expected):
    # Each actual trial is qualified separately first. Stub ONLY individual
    # gates below to isolate the matrix-level equality checks over public copies.
    proofs = {(c,s):receipts.verified_trial(ROOT,c,s) for c,s in TRIALS}
    for c,s in TRIALS:
        for p in receipts.trial_sources(c,s):
            f = tmp_path/p; f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes((ROOT/p).read_bytes())
    p = tmp_path/receipts.trial_sources('full_policy',None)[0 if mutation == 'source' else 1]
    r = json.loads(p.read_text())
    if mutation == 'source':r['source_sha256'][next(iter(r['source_sha256']))] = '0'*64
    elif mutation == 'clock':r['conditional_reliability']['full_clock']['physical_observations'] -= 1
    elif mutation == 'ordinal':r['orders']['1029']['private_video_ordinal_order_sha256'] = '0'*64
    else:r['conditional_reliability']['complete_group_observation_task']['initial_fixed_Raw_task_score']['micro_NLL'] += .001
    p.write_text(json.dumps(r)); monkeypatch.setattr(receipts,'verified_trial',lambda root,c,s:proofs[(c,s)])
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match=expected):receipts.verified_matrix(tmp_path)
