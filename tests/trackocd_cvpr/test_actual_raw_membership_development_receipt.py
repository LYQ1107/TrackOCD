import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import raw_membership_development_receipts as receipts
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort

ROOT = Path(__file__).resolve().parents[2]
CONTROL, SEED = 'calendar_matched', 1027


def test_actual_first_complete_raw_membership_dev_has_all_clocks_and_probes_not_all4_or_Goal():
    proof = receipts.verified_trial(ROOT, CONTROL, SEED)
    assert proof['all3_full_orders_and_complete_probes_verified']
    assert not proof['private_weights_or_GT_loaded_by_verifier']
    assert not proof['full_deployment_calibration_fair_match_control_winner_or_Goal_accepted']


def test_actual_second_complete_raw_membership_dev_retains_distinct_fixed_endpoint_and_same_clock():
    a = receipts.verified_trial(ROOT, CONTROL, SEED)
    b = receipts.verified_trial(ROOT, CONTROL, 1028)
    assert a['result_sha256'] != b['result_sha256']
    assert b['all3_full_orders_and_complete_probes_verified']
    assert not b['private_weights_or_GT_loaded_by_verifier']


def test_actual_third_complete_calendar_control_all_full_probes_without_fullpolicy_substitution():
    r = receipts.verified_trial(ROOT, CONTROL, 1029)
    assert r['calendar_seed'] == 1029 and r['all3_full_orders_and_complete_probes_verified']
    assert not r['full_deployment_calibration_fair_match_control_winner_or_Goal_accepted']


def forbid_private_stat(monkeypatch):
    original = Path.stat
    def guard(path, *args, **kwargs):
        if path.suffix in ('.pt', '.pth'): raise AssertionError('Public receipt verifier touched private model')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', guard)


def test_missing_normal_completion_never_grants_private_followup(tmp_path, monkeypatch):
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError, match='Actual complete public'):receipts.verified_trial(tmp_path, CONTROL, SEED)


@pytest.mark.parametrize('mutation,expected', [
    ('guard', 'bounded normal'), ('orders', 'ALL3 full orders'), ('future', 'ALL3 full orders'),
    ('model', 'Same actual fixed'), ('masked', 'physical track-prefix denominators'),
    ('bins', 'fixed probability-bin'), ('scope', 'Full uncapped'), ('selection', 'bounded normal'),
    ('negative', 'diagnostic clock'), ('partial_order', 'complete original causal order'),
    ('target_denominator', 'physical/Known/pseudo outcome denominator'),
])
def test_false_partial_selected_or_unbounded_public_receipts_reject_without_weights(tmp_path, monkeypatch, mutation, expected):
    formal = verified_cohort(ROOT)  # Only genuine existing public fit proof.
    for p in receipts.trial_sources(CONTROL, SEED):
        f = tmp_path/p; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes((ROOT/p).read_bytes())
    paths = receipts.trial_sources(CONTROL, SEED); result_path = tmp_path/paths[1]
    r = json.loads(result_path.read_text())
    if mutation == 'guard':
        p = tmp_path/paths[2]; t = json.loads(p.read_text()); t['guard_stop_reason'] = 'owned_job_wall'; p.write_text(json.dumps(t))
    elif mutation == 'orders':r['orders'].pop('1029')
    elif mutation == 'future':r['actual_future_probe']['future_observations_mutated'] = 0
    elif mutation == 'model':r['evaluated_model_parameter_sha256'] = '0'*64
    elif mutation == 'masked':r['conditional_reliability']['fixed_observed_prefix_track_tasks']['all_available']['masked_tracks'] = 0
    elif mutation == 'bins':r['conditional_reliability']['complete_group_observation_task']['fixed_formal_endpoint_task_score']['bins'].pop()
    elif mutation == 'scope':r['development_scope']['physical_tracks'] = 421
    elif mutation == 'selection':r['dev_model_seed_threshold_bin_or_checkpoint_selection'] = True
    elif mutation == 'negative':r['conditional_reliability']['unknown_conflicting_groups_as_negative_BG_Novel'] = True
    elif mutation == 'partial_order':r['orders']['1027']['observations'] = 100
    else:r['orders']['1027']['prefix_snapshots']['all_available']['declared_pseudo_queries'] = 80
    result_path.write_text(json.dumps(r))
    monkeypatch.setattr(receipts, 'verified_cohort', lambda root:formal)
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError, match=expected):receipts.verified_trial(tmp_path, CONTROL, SEED)


def test_one_complete_trial_never_substitutes_all4_matrix(tmp_path, monkeypatch):
    real = receipts.verified_trial; first = real(ROOT, CONTROL, SEED); calls = []
    def route(root, control, seed):
        calls.append((control, seed))
        if (control, seed) == (CONTROL, SEED):return first
        return real(root, control, seed)  # Genuine missing second public triple.
    monkeypatch.setattr(receipts, 'verified_trial', route); forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError, match='Actual complete public'):receipts.verified_matrix(tmp_path)
    assert calls == [('calendar_matched', 1027), ('calendar_matched', 1028)]
