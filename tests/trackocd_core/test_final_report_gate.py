import pytest
import json
import hashlib
from pathlib import Path
from scripts.trackocd_core.report_core_training import require_complete,table,require_terminal,terminal_provenance,RUNNING_PROTOCOL_STATUS


def test_final_report_gate_rejects_partial_or_aliased_independent_cases():
    rep={'cases':[{}]*420};policy={'cases':[{}]*1800}
    partial={'status':'RUNNING'}
    with pytest.raises(ValueError,match='incomplete'):require_complete(rep,policy,partial)
    val={'status':'COMPLETE_REAL_FULL_LIMITED_MASA_FROZEN_SEMANTIC_EVALUATION',
         'unique_sealed_executions':840,'logical_cases_including_explicit_aliases':2040,
         'cases':[{'standard':{'old_denominator':4413,'new_denominator':819}}]*2040,'paired_comparisons':[{}]*780}
    require_complete(rep,policy,val)
    val['unique_sealed_executions']=2040
    with pytest.raises(ValueError,match='aliased'):require_complete(rep,policy,val)
    val['unique_sealed_executions']=840;val['cases'][0]['standard']['old_denominator']=1400
    with pytest.raises(ValueError,match='GT penalties'):require_complete(rep,policy,val)


def test_final_table_uses_actual_mean_H_not_H_of_mean_old_new():
    row={k+'_mean':.5 for k in ('old_acc','new_acc','correct_commit_ct','false_merge_rate','effective_commit_coverage')}
    row['h_score_mean']=0.
    assert '| method | 50.00 | 50.00 | 0.00 |' in table([('method',row)],'Method')


@pytest.mark.parametrize('codes,error', [([None],None), ([-9],None), ([True],None), ([0],'failed'), (None,None)])
def test_final_report_requires_real_successful_terminal_not_live_or_failed(codes,error):
    with pytest.raises(ValueError,match='terminal receipt'):
        require_terminal({'owned_child_returncodes':codes,'error':error},'owned_child_returncodes')


def test_orphan_terminal_requires_exact_group_and_unknown_exit_status():
    receipt={'error':None,'owned_child_returncodes':[0],
        'adopted_original_workers_terminal_proof':[{'group':'g','orphan_exit_status':'UNAVAILABLE_NOT_A_CHILD','atomic_terminal_and_every_seal_verified':True}]}
    require_terminal(receipt,'owned_child_returncodes',['g'])
    with pytest.raises(ValueError,match='Every adopted worker'):
        require_terminal(receipt,'owned_child_returncodes',['g','h'])
    receipt['adopted_original_workers_terminal_proof'][0]['orphan_exit_status']=0
    with pytest.raises(ValueError,match='Orphan exit status unknown'):
        require_terminal(receipt,'owned_child_returncodes',['g'])
    receipt['legacy_live_workers_left_untouched']=[{'pid':42}]
    with pytest.raises(ValueError,match='still live'):
        require_terminal(receipt,'owned_child_returncodes',['g'])


def test_original_scientific_registration_separate_from_scheduling_recovery(tmp_path,monkeypatch):
    def put(name,data):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    attempt='a'*32;evaluator='b'*32;recovery_name=f'recovery_{attempt}_started.json'
    identity={'source':'unchanged'};source=b'registered operational recovery';sha=hashlib.sha256(source).hexdigest()
    put('plan.json',{'groups':{'group':['first','second']},'jobs':{'first':{},'second':{}}})
    put(recovery_name,{'identity':identity,'plan_sha256':'plan','preregistration_commit':'recovery',
        'source_sha256':{'recovery.py':sha},'prior_assignments':{'old':{'preregistration_commit':'original','identity':identity,'plan_sha256':'plan'}},
        'adopted_current_host_workers':[{'group':'group'}]})
    put(f'supervisor_{attempt}.json',{'error':None,'owned_child_returncodes':[], 'completed_groups':['group'],
        'adopted_original_workers_terminal_proof':[{'group':'group','orphan_exit_status':'UNAVAILABLE_NOT_A_CHILD','atomic_terminal_and_every_seal_verified':True}]})
    put(f'evaluator_assignment_{evaluator}.json',{'preregistration_commit':'collector','source_sha256':{'eval':'sha'}})
    put(f'evaluator_supervisor_{evaluator}.json',{'error':None,'owned_returncodes':[0]})
    for key,reg in [('first','original'),('second','recovery')]:
        put(f'cases/{key}/case_metrics.json',{'status':'COMPLETE_FULL_GT_POSTHOC_SEMANTIC_METRICS','input_identity':identity,
            'evaluator_source_sha256':{'eval':'sha'},'sealed_ledger_sha256':key,'evaluator_preregistration_commit':reg})
    monkeypatch.setattr('scripts.trackocd_core.report_core_training.subprocess.check_output',lambda *a,**k:source)
    inference={'preregistration_commit':'recovery','identity':identity,'plan_sha256':'plan','scheduling_recovery_receipt':recovery_name,
        'records':[{'job':{'id':key},'ledger':{'sha256':key}} for key in ('first','second')]}
    val={'preregistration_commit':'collector','source_sha256':{'eval':'sha'}}
    result=terminal_provenance(tmp_path,inference,val)
    assert result['original_frozen_inference_registration_commits']==['original']
    assert result['scheduling_recovery_registration_commit']=='recovery'
    assert result['actual_case_evaluator_registration_commits']==['original','recovery']
    assert len(result['actual_successful_terminal_receipts_sha256'])==4
    (tmp_path/f'evaluator_supervisor_{evaluator}.json').unlink()
    with pytest.raises(ValueError,match='evaluator terminal'):
        terminal_provenance(tmp_path,inference,val)


def test_pending_doc_matches_exact_safe_final_replacement_marker():
    root=Path(__file__).resolve().parents[2]
    doc=(root/'docs/trackocd_core/LIMITED_MASA_EVALUATION.md').read_text()
    if '## Completed actual full semantic evaluation' not in doc:
        assert doc.count(RUNNING_PROTOCOL_STATUS)==1
    else:
        result=json.loads((root/'outputs/trackocd_core/CORE_TRAINING_FINAL_RESULT.json').read_text())
        assert RUNNING_PROTOCOL_STATUS not in doc
        assert result['limited_MASA_unique_executions']==840 and result['limited_MASA_logical_cases']==2040
