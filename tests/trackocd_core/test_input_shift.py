import ast
from pathlib import Path
from scripts.trackocd_core.report_input_shift import length_profile


def test_shift_lengths_are_true_clipped_prefixes_and_keep_singletons():
    r=length_profile([1,2,4,16])
    assert r['tracks']==4 and r['singletons']==1 and r['observations']==23 and r['mean_available_observations']==5.75
    assert r['total_observations_at_cap']=={'1':4,'2':7,'4':11,'8':15,'16':23}


def test_shift_source_is_postseal_descriptive_not_model_or_label_repair():
    root=Path(__file__).resolve().parents[2];s=(root/'scripts/trackocd_core/report_input_shift.py').read_text();ast.parse(s)
    assert s.index('seal=json.loads')<s.index('gt=json.loads(gt_path.read_text())')
    assert 'posthoc_description_after_first_Val_metrics_not_preregistered_causal_hypothesis' in s
    assert "'models_or_operating_points_modified':False" in s and "'GT_to_live_inference':False" in s
    assert '256*2**20' in s and 'remaining_RAM_reservation_bytes' in s
    assert 'a[\'visual\']' not in s and 'optimizer' not in s
