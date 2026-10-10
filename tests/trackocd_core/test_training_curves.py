import json
from pathlib import Path
from scripts.trackocd_core.report_training_curves import trace_rows
from src.trackocd_v2.io import sha256_file


def test_trace_rows_keep_real_loss_and_different_policy_objectives():
    fit={'model':'D2_RISK_AWARE','seed':1027,'representation':'A2_EVIDENCE',
        'trace':[{'epoch':1,'episodes':[{'loss':1.},{'loss':3.}]}]}
    r=trace_rows('policy',fit)
    assert r[0]['actual_training_loss']==2. and r[0]['step_or_epoch']==1 and r[0]['phase']=='T3'


def test_actual_33_fit_curves_bind_sealed_training_not_invented_test_scores():
    root=Path(__file__).resolve().parents[2];path=root/'outputs/trackocd_core/TRAINING_CURVES_RESULT.json'
    result=json.loads(path.read_text())
    assert result['actual_fits']==33 and result['actual_trace_rows']==15360
    assert result['D1_D2_have_different_objectives'] and result['loss_is_not_heldout_metric_or_scientific_PASS']
    assert not result['new_training'] and not result['Val_or_Test_read']
    for r in result['receipt_lineages'].values():assert sha256_file(root/r['path'])==r['sha256']
    for p,r in result['artifacts'].items():assert sha256_file(root/p)==r['sha256']
