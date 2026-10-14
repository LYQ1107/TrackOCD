import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import continuity_policy_receipts as receipts
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort

ROOT=Path(__file__).resolve().parents[2]


def no_private_stat(monkeypatch):
    original=Path.stat
    def guard(p,*a,**k):
        if p.suffix in ('.pt','.pth','.npz'):raise AssertionError('Public policy qualifier touched private input')
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'stat',guard)


def test_actual_first_complete_policy_trial_retains_all4arms_probes_origins_and_full_denominators(monkeypatch):
    no_private_stat(monkeypatch);r=receipts.verified_trial(ROOT,'initial_Raw')
    assert r['all4arms_each_full3orders_original_baseline_and_full_probes_verified']
    assert r['origin_masked_scored_reuse_all_physical_unknown_latency_denominators_verified']
    assert not r['private_GT_weights_or_prediction_rows_read'] and not r['unseen_dev_joint_Pred_final_or_Goal_accepted']


@pytest.mark.parametrize('mutation,message',[
    ('guard','Actual bounded'),('arms','Actual bounded'),('orders','Each complete'),('probe','Each complete'),
    ('endpoint','Original frozen'),('scope','Complete lawful'),('baseline','Complete clock'),('commits','Complete clock'),
    ('origins','Actual full origin'),('denominator','All physical'),('mask','All physical'),('latency','Full censored'),
])
def test_partial_changed_or_falsely_certified_policy_receipts_refuse_before_private_followup(tmp_path,monkeypatch,mutation,message):
    formal=verified_cohort(ROOT);paths=receipts.trial_sources('initial_Raw')
    for p in paths:
        q=tmp_path/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
    p=tmp_path/paths[1];r=json.loads(p.read_text());arm=r['arms']['full_raw'];m=arm['orders']['1027']['measured']
    if mutation=='guard':
        q=tmp_path/paths[2];t=json.loads(q.read_text());t['guard_stop_reason']='owned_job_wall';q.write_text(json.dumps(t))
    elif mutation=='arms':r['arms'].pop('exclude_query_raw')
    elif mutation=='orders':arm['orders'].pop('1029')
    elif mutation=='probe':arm['future_mutations']=0
    elif mutation=='endpoint':r['formal_parameter_sha256']='0'*64
    elif mutation=='scope':r['policy_scope']['physical_tracks']=44
    elif mutation=='baseline':arm['orders']['1027']['actual_original_baseline_parity']=False
    elif mutation=='commits':m['actual_commits']-=1
    elif mutation=='origins':m['origin_aware_audit']['physical_origin_observations']['SCORED_CANDIDATE']-=1
    elif mutation=='denominator':m['origin_aware_audit']['primary_scored_reuse_prefix_snapshots']['all_available']['physical_tracks']=44
    elif mutation=='mask':m['origin_aware_audit']['primary_scored_reuse_prefix_snapshots']['all_available']['certified_reuse_excludes_routing_continuation']=False
    else:m['origin_aware_audit']['first_commit_latency']['unresolved_tracks_retained']+=1
    p.write_text(json.dumps(r));monkeypatch.setattr(receipts,'verified_cohort',lambda root:formal);no_private_stat(monkeypatch)
    with pytest.raises(ValueError,match=message):receipts.verified_trial(tmp_path,'initial_Raw')


def test_missing_other_fixed_model_never_substitutes_first_trial_or_opens_private_data(monkeypatch):
    no_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Complete original'):receipts.verified_trial(ROOT,'calendar1027')
