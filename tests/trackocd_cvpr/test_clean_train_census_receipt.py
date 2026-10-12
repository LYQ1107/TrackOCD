"""Actual aggregate/source receipts only: do not decode private GT index."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/clean_train_gt_census_v1'


def test_actual_census_hashes_and_remote_registered_source_binding():
    result = json.loads((OUT / 'result.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == 'b60f9bcd769f7d3b45ef5b01590c0821f863918b83455f165bbd1ec60b18cdd0'
    assert hashlib.sha256((OUT / 'launch.json').read_bytes()).hexdigest() == '5628576d5c67d13795029537a554c1924dd7c0346629dc501823dfea781bd9d2'
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '7887151617bf7a4433e088594240c98674806bc2'
    assert result['source_sha256'] == launch['source_sha256']
    for name, digest in launch['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)).hexdigest() == digest
    assert result['private_index'] == {'bytes': 318880, 'sha256': '94d1e1e1bba5645fc7848aa5f56912fa2d890d05d01385390379f5195d35a2a7'}


def test_all_actual_denominators_reconcile_without_caps_or_short_track_removal():
    c = json.loads((OUT / 'result.json').read_text())['census']
    counts = c['counts']
    assert counts['all_annotation_records_scanned'] == 54639
    assert counts['all_annotation_records_scanned'] == (counts['allowed_annotation_records'] +
        counts['excluded_heldout_records_byte_skipped'] + counts['other_unlisted_records_byte_skipped'])
    assert counts['allowed_annotation_records'] == counts['retained_Known_annotation_records'] + counts['allowed_nonKnown_records_not_retained'] == 47362
    assert counts['retained_Known_annotation_records'] == 38680
    assert counts['excluded_heldout_records_byte_skipped'] == 5608
    assert counts['other_unlisted_records_byte_skipped'] == 1669
    assert c['allowlisted_videos'] == c['videos_with_retained_Known_tracks'] == 429
    hist = {int(n): count for n, count in c['track_observation_count_histogram'].items()}
    assert sum(hist.values()) == c['retained_Known_tracks'] == 1962
    assert sum(n * count for n, count in hist.items()) == 38680
    assert hist[1] == 93 and max(hist) == 40
    assert c['track_cap_applied'] is False and c['short_tracks_retained'] is True
    support = c['Known_category_video_support_counts_sorted']
    video_hist = {int(n): count for n, count in c['video_positive_Known_category_count_histogram'].items()}
    assert len(support) == c['positive_supported_Known_categories'] == 48
    assert sum(support) == sum(n * count for n, count in video_hist.items())
    assert sum(video_hist.values()) == 429
    assert sum(n >= 5 for n in support) == 18
    assert sum(n >= 7 for n in support) == 13
    assert sum(n == 1 for n in support) == 20
    assert max(support) == 376


def test_actual_access_and_science_bounds_remain_explicit():
    result = json.loads((OUT / 'result.json').read_text())
    for key in ('category_names_Novel_vocabulary_nested_video_GT_protected_target_values_decoded',
                'Val_Test_pixels_features_models_read', 'labels_in_forward_or_live_memory',
                'historical_access_incident_erased'):
        assert result[key] is False
    assert result['census']['positive_annotations_certify_visual_category_absence'] is False
    assert result['census']['new_protocol_frozen'] is False
    assert result['optimization_steps'] == result['resources']['training_GPU_hours'] == 0
    resources = result['resources']
    assert resources['wall_seconds'] < 120
    assert resources['guarded_process_RSS_bytes'] < 268435456
    assert resources['peak_sampled_smaps_rollup_Rss_bytes_through_result_snapshot'] < 268435456
    assert resources['resource_ru_maxrss_bytes_including_preexec_history'] > resources['guarded_process_RSS_bytes']


def test_failed_pre_read_attempt_and_actual_user_host_resume_preserved():
    failure = json.loads((ROOT / 'outputs/trackocd_cvpr/safety/clean_train_gt_census_network_gate_failure_v1.json').read_text())
    assert failure['complete'] is False and failure['git_exit_code'] == 128
    assert failure['GT_values_decoded_in_this_attempt'] is False
    assert failure['script_fresh_remote_gate_not_bypassed'] is True
    resume = json.loads((ROOT / 'outputs/trackocd_cvpr/safety/clean_train_gt_goal_resume_receipt_v1.json').read_text())
    assert resume['first_goal_tool_status_after_scope_approval'] == 'blocked'
    assert resume['subsequent_goal_tool_status'] == 'active'
    assert resume['full_original_objective_unchanged'] is True
    assert resume['replacement_goal_created'] is False
    assert resume['Goal_A_B_C_accepted'] is False
