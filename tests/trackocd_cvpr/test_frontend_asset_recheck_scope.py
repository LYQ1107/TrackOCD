import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readonly_frontend_recheck_reports_exact_existing_assets_not_a_new_primary_or_model_smoke():
    r = json.loads((ROOT / 'outputs/trackocd_cvpr/cvpr5/image_only_frontend_asset_recheck_v1.json').read_text())
    assert r['checkpoint']['bytes'] == 558882875
    assert r['checkpoint']['sha256'] == '441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c'
    assert r['checkpoint']['matches_existing_pinned_LFS_spec']
    assert r['checkpoint']['whole_file_hashed_without_deserialization']
    assert not r['checkpoint']['model_loaded_or_tensor_keys_or_checkpoint_class_metadata_decoded']
    assert r['pinned_sources']['native_files'] == 14 and r['pinned_sources']['AMG_files'] == 4
    assert not r['isolated_runtime_installed_metadata']['runtime_import_CUDA_or_model_compatibility_inferred']
    assert r['new_recovered_upstream_source_weights_environment_dataset_or_cache_bytes'] == 0
    assert not r['model_inference_training_GT_annotation_pixel_or_Test_read']
    assert not r['strong_legal_primary_Pred_or_scientific_Goal_accepted']
    assert not r['current_source_contract_facts']['actual_forbidden_category_metadata_presence_or_historical_scope_violation_proved']
