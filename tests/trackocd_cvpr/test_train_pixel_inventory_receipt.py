"""Actual path/stat receipt checks; no real pixels decoded during tests."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/train_pixel_inventory_v1'


def test_pixel_inventory_receipt_is_bound_to_actual_registered_source():
    path = OUT / 'result.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'e0c0ad8777f13d5002d4bff29cef70fa633d12767bb6b374a6ba87afd66171ef'
    result = json.loads(path.read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '8e499515e3ce49b843529a2f097d9d733c29e2ab'
    for name, digest in launch['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == result['source_sha256'][name] == digest
    assert result['annotation_or_category_arrays_JSON_decoded'] is result['model_executed'] is result['Val_or_Test_access'] is False
    assert result['metadata_scope']['old_final_heldout_labels_decoded'] is False
    assert result['resources']['training_GPU_hours'] == result['optimization_steps'] == 0


def test_available_pixels_not_misreported_as_dense_valid_features_or_folds():
    result = json.loads((OUT / 'result.json').read_text())
    pixels = result['pixel_inventory']
    assert pixels['selected_safe_reference_videos'] == pixels['complete_annotated_image_videos'] == 421
    assert pixels['annotated_images'] == pixels['present_images'] == 15275
    assert pixels['present_bytes'] == 1798339000
    assert pixels['missing_images'] == pixels['invalid_existing_files'] == 0
    assert pixels['pixel_bytes_hashed_or_decoded'] is pixels['dense_frame_coverage_certified'] is pixels['asset_download_or_deletion'] is False
    assert result['reference_support']['new_fold_or_training_protocol_frozen'] is False
    assert result['reference_support']['full_GT_census'] is False
