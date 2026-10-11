import json
from pathlib import Path
import pytest
from scripts.trackocd_core.enrich_final_handoff import per_seed_four_orders
from scripts.trackocd_core.report_final_handoff import mean_std

OUT=Path(__file__).resolve().parents[2]/'outputs/trackocd_core'


def test_each_seed_four_order_means_and_std_are_actual_not_h_of_means():
    v=json.loads((OUT/'LIMITED_MASA_EVALUATION_RESULT.json').read_text());rows=per_seed_four_orders(v)
    assert len(rows)==510 and sum(r['video_orders'] for r in rows)==2040
    for r in rows:
        cases=[c for c in v['cases'] if (c['method'],c['prefix'],c['coverage_target'],c['seed'])==(r['method'],r['prefix'],r['coverage_target'],r['seed'])]
        m,s=mean_std([c['standard']['h_score'] for c in cases])
        assert r['h_score_four_order_mean']==pytest.approx(m)
        assert r['h_score_four_order_std_ddof0']==pytest.approx(s)
    assert all(r['h_score_four_order_mean']==0 for r in rows if r['method']=='FULL')


def test_partial_order_group_cannot_become_published_seed_statistic():
    v=json.loads((OUT/'LIMITED_MASA_EVALUATION_RESULT.json').read_text());v['cases']=v['cases'][1:]
    with pytest.raises(ValueError):per_seed_four_orders(v)
