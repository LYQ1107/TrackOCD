"""Metadata receipt regressions, not fresh-fold/scientific acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/train_protocol_inventory_v1'


def test_completed_inventory_exact_launch_binding_without_new_protocol_claim():
    path = OUT / 'result.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '5117b414c9fe49a35604c1221176f8dab2b0910890dd76650333592e4bc77615'
    result = json.loads(path.read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert result['preregistration_commit'] == launch['preregistration_commit'] == 'f832f4c74e32e2ad549d5a5b336613b4bf60ec8e'
    for name, digest in launch['source_sha256'].items():
        source = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(source).hexdigest() == result['source_sha256'][name] == digest
    assert result['metadata_scope']['old_final_heldout_labels_decoded'] is False
    assert result['metadata_scope']['feature_arrays_decoded'] is False
    assert result['GT_forward_or_model_execution'] is result['Val_or_Test_or_new_GT_annotation_access'] is False
    assert result['summary']['category_selection_performed'] is result['summary']['new_fold_or_training_protocol_frozen'] is False
    assert result['resources']['training_GPU_hours'] == result['optimization_steps'] == 0
    assert result['resources']['peak_RSS_bytes'] < 1024 ** 3


def test_existing_video_cache_is_not_misreported_as_full_reference_or_matching():
    summary = json.loads((OUT / 'result.json').read_text())['summary']
    assert summary['categories'] == 15
    assert summary['permitted_reference_pool']['selected_capped_GT_tracks'] == 1867
    assert summary['permitted_reference_pool']['videos'] == 421
    support = summary['category_video_support_threshold_counts']
    assert support['permitted_reference_pool']['5'] == 15
    assert support['reference_video_intersection_with_existing_predicted_cache']['2'] == 10
    assert support['reference_video_intersection_with_existing_predicted_cache']['5'] == 3
    assert summary['full_GT_census'] is False
    assert summary['predicted_cache_intersection_is_not_matching_or_predicted_track_denominator'] is True
