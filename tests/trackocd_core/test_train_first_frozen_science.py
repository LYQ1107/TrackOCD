import json
from pathlib import Path
import numpy as np
from src.trackocd_v2.io import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def test_actual_policy1800_cases_fixed_denominators_and_coverage_comparability():
    r=json.loads((ROOT/'outputs/trackocd_core/TRAIN_FIRST_POLICY_HELDOUT_RESULT.json').read_text())
    assert len(r['cases'])==1800 and len(r['aggregate'])==150
    assert not r['GT_or_mapping_in_policy_state'] and not r['val_or_test_access'] and not r['heldout_operating_point_tuning']
    for n,sha in r['source_sha256'].items():assert sha256_file(ROOT/n)==sha
    for c in r['cases']:
        assert c['standard']['old_denominator']==207 and c['standard']['new_denominator']==16
        assert c['persistent']['fixed_gt_cross_video_reuse_opportunities']=={'main':9,'seed1027':7,'seed1028':7,'seed1029':9}[c['order']]
        assert sum(c['persistent']['outcome_counts'][k] for k in ('pure_correct_reuse','false_split_new','wrong_known_assignment','contaminated_token_reuse','unverified_token_reuse','wrong_category_merge','same_video_only_unsupported_existing','missed_predicted_opportunity','wait'))==c['persistent']['commit_ct_denominator']
    assert all(p['comparable_within_registered_tolerance']==(max(p['actual_reuse_opportunity_coverage_gap'],p['actual_total_commitment_coverage_gap'])<=.05) for p in r['paired_comparisons'])
    assert sum(p['comparable_within_registered_tolerance'] for p in r['paired_comparisons'])==377
    path=ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy_evaluation/sealed_predictions.parquet'
    assert sha256_file(path)==r['private_ledger']['sha256']


def test_actual_role_separated_queries_and_no_scientific_success_by_known_inflation():
    r=json.loads((ROOT/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json').read_text())
    assert len(r['analyses'])==50 and not r['model_optimizer_or_selection'] and not r['new_threshold_trial']
    for a in r['analyses']:
        novel=a['roles']['pseudo_novel'];assert novel['query_tracks']==16 and novel['queries_with_cross_video_positive']==16
        assert novel['unsupported_queries']==0 and 'Known' in novel['gallery']
    for prefix in (1,2,4,8,16):
        raw=next(a['roles']['pseudo_novel']['recall_at_1_category_macro'] for a in r['analyses'] if a['representation']=='A0_RAW' and a['prefix']==prefix)
        learned=np.mean([a['roles']['pseudo_novel']['recall_at_1_category_macro'] for a in r['analyses'] if a['representation']=='A1_SELECTED' and a['prefix']==prefix])
        assert learned<raw  # Actual negative outcome kept,not a claimed desired target.


def test_actual_model_freeze_hashes_all_completed_stages_not_goal_complete():
    path=ROOT/'outputs/trackocd_core/core_training/train_first_v1/model_freeze.json';r=json.loads(path.read_text())
    assert r['status']=='FROZEN_TRAIN_ONLY_MODELS_AND_OPERATING_POINTS' and not r['scientific_PASS_claimed'] and not r['final_goal_complete']
    assert not r['TAO_Test_access'] and not r['Val_Novel_training_or_tuning']
    for n,sha in r['protected_sha256'].items():assert sha256_file(ROOT/n)==sha
    assert r['historical_M1_status']=='BLOCKED_FRONTEND_QUALITY' and 'LIMITED' in r['native_MASA_status']
