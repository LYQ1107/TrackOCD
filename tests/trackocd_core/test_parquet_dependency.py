from scripts.trackocd_core.check_parquet_dependency import run_smoke


def test_synthetic_parquet_and_fp16_roundtrip_is_not_real_cache_validation(tmp_path):
    result = run_smoke(tmp_path)
    assert result['status'] == 'PASS_SYNTHETIC_CPU_IO_ONLY'
    assert (result['tracks'], result['observations'], result['dimension']) == (2, 5, 768)
    assert result['parquet_nested_row_roundtrip_pass']
    assert result['float16_npy_memory_map_roundtrip_pass']
    assert result['same_local_id_different_video_retained']
    assert not result['real_nas_shards_validated']
    assert not result['legacy_sharded_reader_imported']
    assert not result['feature_contract_compatibility_proved']
    assert not result['test_annotation_opened']
    assert not result['training_started']
