import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file


def test_first_complete_case_is_not_partial_video_or_best_configuration_claim():
    root=Path(__file__).resolve().parents[2]
    d=json.loads((root/'outputs/trackocd_core/LIMITED_MASA_FIRST_FULLSTREAM_CASE_RESULT.json').read_text())
    assert (d['method'],d['order'],d['prefix'],d['seed'])==('B0_frame_snapshot_vote','main',1,None)
    assert d['full988videos'] and d['all304561physical_IDs'] and not d['full_matrix_complete']
    assert (d['standard']['old_denominator'],d['standard']['new_denominator'],d['persistent']['commit_ct_denominator'])==(4413,819,527)
    assert d['source_sha256']==sha256_file(root/'scripts/trackocd_core/export_first_fullstream_case.py')
    assert d['GT_unresolved_includes_physically_missing_not_only_policy_WAIT']
    assert d['persistent']['outcome_counts']['missed_predicted_opportunity']==409 and d['runtime']['wait_rate']==0
    assert not d['GT_or_metrics_in_inference'] and not d['Val_tuning'] and not d['TAO_Test_access']
