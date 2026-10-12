"""Actual stat-only selected pixels; no decoded/common-feature proof."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v12_images'


def test_actual_all_selected_image_paths_exist_GT_keys_align_and_empty_frames_not_BG():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['selected_videos'] == 407 and public['selected_image_metadata_rows'] == 14798
    assert public['GT_observations_cross_checked'] == 45450 and public['GT_referenced_images'] == 14017
    assert public['selected_metadata_images_without_GT_reference'] == 781
    for role, videos, images in (('fit_known', 257, 9345), ('policy_train_pseudo', 10, 374),
                                ('development_pseudo', 72, 2562), ('independent_final_pseudo', 68, 2517)):
        inventory = public['role_pixel_inventory'][role]
        assert inventory['complete_annotated_image_videos'] == videos
        assert inventory['annotated_images'] == inventory['present_images'] == images
        assert inventory['missing_images'] == inventory['invalid_existing_files'] == 0
    assert not public['annotation_category_video_GT_values_decoded']
    assert not public['pixel_bytes_hashed_or_decoded'] and not public['features_loaded_or_extracted']
    assert not public['dense_frame_coverage_proven'] and not public['new_protocol_frozen']
    assert result['selected_Train_path_metadata_source_read_without_target_decoding']
    assert result['live_remote_registration_proof']['sha'] == '62c6233fc18c050c651367e8b4c55ea057be2e69'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_2.json').read_text())
    assert receipt['successful_owned_execution'] and receipt['guard_stop_reason'] is None
    assert receipt['peak_sampled_worker_threads'] == 1 and receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
