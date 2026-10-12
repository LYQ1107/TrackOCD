import json
from pathlib import Path


def test_frozen_R_window_requires_actual_seeded_matched_trials_not_smoke_or_three_parameter_mechanisms():
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/R_GT_scientific_window_v1.json').read_text())
    assert c['training_seed_replicates'] == [1027, 1028, 1029]
    assert len(c['variants']) == 4 and c['trainable_parameters_each'] == 50113
    assert c['training']['steps_per_variant_and_seed'] == 200
    assert c['training']['checkpoint_selection'] == 'Fixed200 endpoint, no bestloss/dev/final selection'
    assert c['training']['tracks_without_fit5_semantic_targets'] == 243
    assert not c['independent_final_or_Val_Test_for_model_selection']
    assert not c['automatic_launch_or_Goal_completion_from_manifest']
    assert c['maximum_evidence_driven_scientific_repairs'] == 2
    assert 'ALL current support member embeddings' in c['training']['Raw_teacher']
