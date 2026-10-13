"""Numeric qualification on qualified dev cache only; no semantic projection."""
import json
import torch
from .GT_video_cache import policy_dev_video_ids, read_verified_shard
from .policy_training_data import policy_current_frames
from .causal_observation_router import CurrentObservationRouter, visible_bytes
from .prefix_member_bank import CommittedRawSupport
from .reliability_support_bridge import committed_reliability_inputs
from .frozen_raw_statistics_cache import FrozenRawStatisticsCache
from src.trackocd_v2.io import sha256_file


def scoped_numeric_video_rows(root, cfg):
    frozen = json.loads((root / cfg['frozen_protocol']).read_text())
    assignment = root / cfg['private_assignment']
    if assignment.is_symlink() or sha256_file(assignment) != frozen['input_sha256']['private_global_assignment']:
        raise ValueError('Exact immutable ADMINISTRATIVE role assignment')
    # This reader skips ALL class/final values. No private Known semantic index.
    roles = policy_dev_video_ids(assignment)
    directory = root / cfg['policy_dev_cache_directory']
    public = json.loads((directory / 'result.json').read_text())
    terminal = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
    if (sha256_file(directory / 'result.json') != 'b6747cbc8d4802620cc56f2a2af25b6e172e664ab6903e79d04cda4ab2a9e0be'
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None
            or public['development_or_final_semantic_values_decoded'] or public['independent_final_or_protected_pixels_read']):
        raise ValueError('Qualified frozen geometry-only completed input')
    index_path = directory / 'private_shard_index.json'
    if (index_path.is_symlink() or sha256_file(index_path) != '2064817b09c7a27621f34c6182b236ab59f8c92bf920062f902e9edf3e55ee25'):
        raise ValueError('Exact immutable administrative routing before JSON decode')
    index = json.loads(index_path.read_text())
    if (len(index) != 82 or len({row['video_id'] for row in index}) != 82
            or {row['video_id'] for row in index} != set(roles)
            or sorted(row['ordinal'] for row in index) != list(range(82))):
        raise ValueError('Complete administrative 82-video universe')
    selected = tuple(row for row in sorted(index, key=lambda r: r['ordinal']) if roles[row['video_id']] == 'development_pseudo')
    if len(selected) != 72: raise ValueError('Exact authorized72 development videos')
    return directory, frozen, public, selected


def qualify_numeric_prefixes(root, cfg, guard):
    directory, frozen, public, selected = scoped_numeric_video_rows(root, cfg)
    by_ordinal = {row['ordinal']: row for row in public['shards']}
    cache = FrozenRawStatisticsCache(); frames = empty = observations = groups = tracks = 0
    with torch.inference_mode():
        for row in selected:
            guard(); arrays, record = read_verified_shard(directory, row['ordinal'])
            expected = {'opaque_video_id': row['video_id'], 'frozen_protocol_sha256': sha256_file(root / cfg['frozen_protocol']),
                'physical_index_sha256': frozen['input_sha256']['private_physical_GT'],
                'image_plan_sha256': frozen['input_sha256']['private_image_plan'],
                'checkpoint_sha256': frozen['GT_control_feature_IO']['checkpoint_sha256']}
            if (row['source_directory'] != cfg['policy_dev_cache_directory']
                    or any(record['lineage'].get(k) != v for k, v in expected.items())
                    or record['payload']['sha256'] != row['payload_sha256']
                    or record['payload']['sha256'] != by_ordinal[row['ordinal']]['private_payload_sha256']):
                raise ValueError('Exact private/public frozen shard lineage, no alternate input')
            router = CurrentObservationRouter(max_tracks=472, max_observations=8753)
            tracks += len(arrays['track_id'])
            for frame, current in policy_current_frames(arrays):
                guard(); batch = router.ingest(frame, current)
                for view in batch.model_views:
                    payload = visible_bytes(view)
                    # Two identical numeric rows still remain separate support
                    # scopes; no alias/video dedup or model output selection.
                    old = committed_reliability_inputs((CommittedRawSupport('S:0', (payload, payload)),), 'S:0')
                    new = cache.inputs((payload, payload)); again = cache.inputs((payload, payload))
                    if any(a.numpy().tobytes() != b.numpy().tobytes() or a.numpy().tobytes() != c.numpy().tobytes()
                           for a, b, c in zip(old, new, again)):
                        raise ValueError('Actual frozen Raw statistics bytes differ: whole qualification fails')
                    groups += 1
                frames += 1; empty += int(not current); observations += len(current)
    if (len(selected), tracks, frames, empty, observations) != (72, 472, 2562, 116, 8753):
        raise ValueError('Complete fixed input/prefix clock, no short or empty drop')
    return {'videos': 72, 'physical_tracks': tracks, 'metadata_frames': frames, 'empty_frames': empty,
        'physical_observations': observations, 'canonical_visible_prefixes_checked': groups,
        'exact_CPU_original_cached_miss_hit_and_duplicate_scope_tensor_bytes': True,
        'derived_cache_max_bytes': 67108864, 'retained_derived_cache_bytes': cache.retained_bytes,
        'semantic_class_final_values_Known_index_or_GT_labels_decoded': False,
        'model_parameters_or_prediction_memory_or_GT_feedback_or_optimizer': False,
        'full_controller_shadow_integration_speedup_or_method_gain': False}
