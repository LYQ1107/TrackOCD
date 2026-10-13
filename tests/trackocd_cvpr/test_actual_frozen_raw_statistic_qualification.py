import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_complete_semantic_free_prefix_qualification_not_integration_or_method_gain():
    p = ROOT / 'outputs/trackocd_cvpr/cvpr4/qualified_frozen_raw_statistics_v1/result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '35743611e5eb395bd4091fac50b9e815515a6404c2f95ad5328c594580144f57'
    r = json.loads(p.read_text()); q = r['numeric_qualification']
    assert r['live_remote_registration_proof']['sha'] == '5db8bb3a9ef8ce57d82dd0ef95fcb8fa38f154bd'
    assert (q['videos'], q['physical_tracks'], q['metadata_frames'], q['empty_frames'],
            q['physical_observations'], q['canonical_visible_prefixes_checked']) == (72, 472, 2562, 116, 8753, 8753)
    assert q['exact_CPU_original_cached_miss_hit_and_duplicate_scope_tensor_bytes']
    assert q['retained_derived_cache_bytes'] <= q['derived_cache_max_bytes'] == 67108864
    assert not q['semantic_class_final_values_Known_index_or_GT_labels_decoded']
    assert not q['model_parameters_or_prediction_memory_or_GT_feedback_or_optimizer']
    assert not q['full_controller_shadow_integration_speedup_or_method_gain']
    assert r['resources']['peak_guarded_RSS_bytes'] <= 536870912
    assert r['resources']['wall_seconds_excluding_registration'] <= 600
    assert r['resources']['training_GPU_hours'] == 0 and not r['resources']['CUDA_initialized']
