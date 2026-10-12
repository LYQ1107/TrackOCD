"""Approved geometry-only annotated GT census, not semantic/dense evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v10_physical_GT'


def test_actual_physical_census_scope_counts_and_original_geometry_authority():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['allowlisted_videos'] == public['represented_videos'] == 407
    assert public['physical_tracks'] == 2222 and public['selected_image_metadata_rows'] == 14798
    assert public['counts'] == dict(all_annotation_records_byte_scanned=54639,
        retained_physical_observations=45450, old_heldout_records_byte_skipped=5608,
        other_unlisted_records_byte_skipped=3581)
    assert public['single_observation_tracks'] == 95
    assert not public['category_values_names_Novel_vocabulary_video_GT_arrays_decoded']
    assert not public['image_paths_pixels_features_models_read']
    assert not public['dense_frame_coverage_proven'] and not public['new_protocol_frozen']
    assert result['original_Train_annotation_geometry_read_under_new_explicit_authority']
    assert result['original_annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read'] is None
    assert result['optimization_training_steps'] == 0
    assert result['live_remote_registration_proof']['sha'] == 'decedf4b74bb738b42cb21cf128079cbc72289de'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['owned_worker_exit_code'] == 0 and receipt['successful_owned_execution']
    assert receipt['guard_stop_reason'] is None and receipt['peak_sampled_worker_threads'] == 1
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
    # No private geometry file is opened or uploaded by public receipt tests.
