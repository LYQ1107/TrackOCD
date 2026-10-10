import pytest
from scripts.trackocd_core.report_core_training import require_complete,table


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
