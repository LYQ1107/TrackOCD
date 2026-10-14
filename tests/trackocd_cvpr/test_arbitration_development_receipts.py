import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import arbitration_development_receipts as receipts
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort

ROOT = Path(__file__).resolve().parents[2]


def forbid_private_stat(monkeypatch):
    original = Path.stat
    def guard(p,*args,**kwargs):
        if p.suffix in ('.pt','.pth'):raise AssertionError('Public dev receipt qualifier touched private model')
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',guard)


@pytest.mark.parametrize('model',['initial_Raw','calendar1027'])
def test_actual_complete_dev_observer_all3_references_and_full_probes_not_full5_or_intervention_win(monkeypatch,model):
    forbid_private_stat(monkeypatch); p = receipts.verified_trial(ROOT,model)
    assert p['model'] == model
    assert p['all3_full_orders_each_original_reference_and_full_probes_checked']
    assert not p['private_models_GT_or_row_tables_read']
    assert not p['full_intervention_causal_gain_or_Goal_accepted']


def test_missing_full_normal_receipt_refuses_without_private_followup(tmp_path,monkeypatch):
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Complete original full-dev'):receipts.verified_trial(tmp_path,'initial_Raw')


@pytest.mark.parametrize('mutation,message',[
    ('guard','Actual bounded normal'),('orders','Actual bounded normal'),('probe','Actual bounded normal'),
    ('write','Actual bounded normal'),('selection','Actual bounded normal'),('model','Actual fixed formal'),
    ('scope','Complete lawful physical'),('reference','Each complete full-order'),('masked','All eligible/masked'),
    ('partition','Support provenance'),('shadow','Support provenance'),
])
def test_partial_false_selected_or_written_shadow_dev_receipts_refuse_private_inputs(tmp_path,monkeypatch,mutation,message):
    formal = verified_cohort(ROOT)
    paths = receipts.trial_sources('initial_Raw')
    for p in paths:
        f = tmp_path/p; f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes((ROOT/p).read_bytes())
    p = tmp_path/paths[1]; r = json.loads(p.read_text()); order = r['orders']['1027']
    if mutation == 'guard':
        q = tmp_path/paths[2]; t = json.loads(q.read_text()); t['guard_stop_reason'] = 'owned_job_wall'; q.write_text(json.dumps(t))
    elif mutation == 'orders':r['orders'].pop('1029')
    elif mutation == 'probe':r['actual_future_probe']['future_observations_mutated'] = 0
    elif mutation == 'write':r['shadow_or_GT_used_for_policy_memory_write'] = True
    elif mutation == 'selection':r['dev_head_seed_order_threshold_selected'] = True
    elif mutation == 'model':r['evaluated_parameter_sha256'] = '0'*64
    elif mutation == 'scope':r['development_scope']['physical_tracks'] = 421
    elif mutation == 'reference':order['actual_full_normal_event_commit_state_equal'] = False
    elif mutation == 'masked':order['measured']['external_strata']['masked']['physical_observations'] = 0
    elif mutation == 'partition':order['measured']['external_strata']['Known']['Existing_provenance_partition_only_own_groups'] += 1
    else:order['measured']['external_strata']['policy_pseudo']['same_history_remove_own_correct_Known_groups'] = 1
    p.write_text(json.dumps(r)); monkeypatch.setattr(receipts,'verified_cohort',lambda root:formal)
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match=message):receipts.verified_trial(tmp_path,'initial_Raw')


def test_one_model_never_substitutes_full5_window(tmp_path,monkeypatch):
    real = receipts.verified_trial; first = real(ROOT,'initial_Raw'); called = []
    def route(root,model):
        called.append(model)
        return first if model == 'initial_Raw' else real(root,model)
    monkeypatch.setattr(receipts,'verified_trial',route); forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Complete original full-dev'):receipts.verified_matrix(tmp_path)
    assert called == ['initial_Raw','calendar1027']
