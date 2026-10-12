import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_human_permission_and_fixed_data_IO_label_boundary():
    approval = json.loads((ROOT / 'configs/trackocd_cvpr/formal_research_scope_approval_v1.json').read_text())
    cfg = json.loads((ROOT / 'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json').read_text())
    assert approval['user_message_verbatim'] == '同意，先冻结协议，再有限缓存与训练'
    assert approval['bound_private_assignment_sha256'] == cfg['input_sha256']['private_global_assignment']
    assert sum(r['classes'] for r in cfg['roles'].values()) == 23
    assert sum(r['videos'] for r in cfg['roles'].values()) == 407
    physical = cfg['annotated_physical_scope']
    assert sum(physical['role_track_denominators'].values()) == physical['tracks'] == 2222
    assert physical['images_without_GT_reference'] == 781 and physical['no_short_low_quality_unmatched_or_empty_frame_drop']
    assert cfg['training_labels']['eligible_supervised_track_count'] == 1158 + 34 + 10
    assert not cfg['training_labels']['semantic_labels_GT_purity_roles_IDs_or_futurelength_in_numeric_forward']
    assert not cfg['training_labels']['GT_repair_of_live_prediction_owned_memory']
    assert cfg['GT_control_feature_IO']['maximum_observations_per_track'] is None
    assert not cfg['execution_gates']['training_started_or_Goal_A_B_C_achieved']
    assert not cfg['execution_gates']['full_cache_automatic_start_from_this_manifest']
    assert cfg['execution_gates']['small_pilot_cache_qualification_first']
    assert not approval['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
