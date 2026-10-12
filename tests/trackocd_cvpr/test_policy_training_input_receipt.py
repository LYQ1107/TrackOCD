"""Actual owned private-input qualification, not policy/memory training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'


def test_actual_full_policy_input_owned_completion_and_unexpanded_label_scope():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == 'bdc762881ad0656115ac118e8f9c6d538d749803147ee7e181ce2d5fb5719993'
    result = json.loads((OUT / 'result.json').read_text())
    scope = result['policy_scope']; frames = result['actual_frame_adapter']
    assert result['live_remote_registration_proof']['sha'] == '04d2865fc700fddc855f4d5353c5e36fb9c9e792'
    assert (scope['videos'], scope['physical_tracks'], scope['observations'], scope['metadata_frames']) == (10, 54, 1268, 374)
    assert scope['existing_Known_target_counts'] == {'Known': 34, 'policy_pseudo': 10}
    assert scope['physical_label_strata'] == {'Known': 34, 'policy_pseudo': 10, 'no_retained_semantic_truth': 10}
    assert scope['masked_loss_tracks'] == 10 and not scope['masked_tracks_dropped_or_declared_background']
    assert not scope['development_final_or_other_video_semantic_values_decoded']
    assert not scope['development_final_or_other_video_payload_bodies_opened']
    assert frames['metadata_frames'] == 374 and frames['empty_frames'] == 1
    assert frames['physical_observations'] == frames['canonical_visible_groups'] == 1268
    assert frames['bijective_ID_rename_and_reversed_delivery_numeric_inputs_equal']
    assert result['optimizer_steps'] == 0 and not result['model_or_memory_training_or_scientific_acceptance']
    assert not result['original_annotations_pixels_weights_or_protected_semantics_read']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['training_GPU_hours'] == 0
