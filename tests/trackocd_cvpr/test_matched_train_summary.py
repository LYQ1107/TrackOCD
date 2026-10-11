import copy
import pytest
from scripts.trackocd_cvpr.write_matched_train_summary import summarize


def fixture_result():
    cases = []
    for rep in ('raw', 'a1', 'a2'):
        for seed in (1027, 1028, 1029):
            for prefix in (1, 2, 4, 8, 16):
                sides = {}
                for name, count in (('predicted', 16933), ('GT_controlled', 258)):
                    sides[name] = {'action_counts': {'WAIT': count},
                        'runtime': {'commitment_coverage': 0.0},
                        'evaluation': {'standard': {'old_denominator': 249, 'new_denominator': 9,
                            'old_acc': 0.0, 'new_acc': 0.0, 'h_score': 0.0,
                            'non_wait_coverage': 0.0, 'prediction_coverage': 1.0},
                            'persistent': {'commit_ct_denominator': 5, 'correct_commit_ct': 0.0,
                                           'effective_commit_coverage': 0.0}}}
                cases.append({'representation': rep, 'seed': seed, 'prefix': prefix, **sides})
    return {'cases': cases}


def test_actual_values_and_coverage_types_preserved():
    result = fixture_result()
    result['cases'][0]['predicted']['evaluation']['standard']['h_score'] = 0.125
    rows = summarize(result)
    assert len(rows) == 90 and rows[0]['H'] == 0.125
    assert rows[0]['reference_prediction_coverage'] == 1.0
    assert rows[0]['reference_non_wait_coverage'] == 0.0
    assert rows[0]['WAIT'] == 16933 and not rows[0]['main_acceptance']


@pytest.mark.parametrize('mutation', ('denominator', 'missing', 'duplicate', 'input_count'))
def test_real_summary_input_mutations_refused(mutation):
    result = copy.deepcopy(fixture_result())
    if mutation == 'denominator':
        result['cases'][0]['predicted']['evaluation']['standard']['old_denominator'] = 107
    elif mutation == 'missing':
        result['cases'].pop()
    elif mutation == 'duplicate':
        result['cases'][0] = result['cases'][1]
    else:
        result['cases'][0]['predicted']['action_counts']['WAIT'] = 16932
    with pytest.raises(ValueError):
        summarize(result)
