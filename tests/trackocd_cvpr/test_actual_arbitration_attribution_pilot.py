import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import arbitration_attribution_qualification as qualification
from src.trackocd_cvpr.arbitration_attribution_protocol import OUT
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort

ROOT = Path(__file__).resolve().parents[2]
PATHS = (OUT+'/launch.json',OUT+'/result.json',OUT+'_supervisor_attempt_1.json')


def forbid_private_stat(monkeypatch):
    original = Path.stat
    def guard(p,*args,**kwargs):
        if p.suffix in ('.pt','.pth'):raise AssertionError('Public pilot qualifier touched private model')
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',guard)


def test_actual_normal_all5_policy_observer_is_not_unseen_cause_intervention_rollout_or_private_authority(monkeypatch):
    forbid_private_stat(monkeypatch); r = qualification.verify_actual_policy_pilot(ROOT)
    assert r['actual_normal_all5_policy_pilot_verified']
    assert not r['private_GT_models_or_row_tables_read_by_qualifier']
    assert not r['one_step_policy_resubstitution_is_unseen_dev_or_intervention_stream_win']
    assert not r['new_private_access_automatic_expansion_or_Goal_acceptance']


def test_partial_pilot_never_substitutes_complete_observer_or_grants_private_followup(tmp_path,monkeypatch):
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Complete original all5'):qualification.verify_actual_policy_pilot(tmp_path)


@pytest.mark.parametrize('mutation,message',[
    ('guard','Actual bounded normal'),('model_subset','Actual bounded normal'),('intervention','Actual bounded normal'),
    ('expanded_role','Actual bounded normal'),('model','Each complete actual'),('probe','Each complete actual'),
    ('clock','Each complete actual'),('masked','All policy physical'),('partition','Existing provenance partition'),
    ('shadow','Existing provenance partition'),('schedule','Same actual complete policy video schedule'),
])
def test_false_partial_or_selected_or_written_shadow_pilot_refuses_without_private_access(tmp_path,monkeypatch,mutation,message):
    formal = verified_cohort(ROOT)
    for p in PATHS:
        f = tmp_path/p; f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes((ROOT/p).read_bytes())
    p = tmp_path/PATHS[1]; r = json.loads(p.read_text()); m = r['models']['calendar1027']; s = m['measured']['external_strata']
    if mutation == 'guard':
        q = tmp_path/PATHS[2]; t = json.loads(q.read_text()); t['guard_stop_reason'] = 'owned_job_wall'; q.write_text(json.dumps(t))
    elif mutation == 'model_subset':r['models'].pop('initial_Raw')
    elif mutation == 'intervention':r['shadow_or_GT_used_for_policy_memory_write'] = True
    elif mutation == 'expanded_role':r['dev_final_Val_Test_pixels_or_new_nonKnown_semantics_read'] = True
    elif mutation == 'model':m['formal_parameter_sha256'] = '0'*64
    elif mutation == 'probe':m['future_observations_mutated'] = 0
    elif mutation == 'clock':m['measured']['full_clock']['physical_observations'] = 100
    elif mutation == 'masked':s['masked']['physical_observations'] = 0
    elif mutation == 'partition':s['Known']['Existing_provenance_partition_only_own_groups'] += 1
    elif mutation == 'shadow':s['policy_pseudo']['same_history_remove_own_correct_Known_groups'] = 1
    else:m['measured']['video_ordinal_schedule_sha256'] = '0'*64
    p.write_text(json.dumps(r)); monkeypatch.setattr(qualification,'verified_cohort',lambda root:formal)
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError,match=message):qualification.verify_actual_policy_pilot(tmp_path)
