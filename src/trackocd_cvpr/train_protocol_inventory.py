"""External Train-label feasibility summaries; never a model-facing API.

Only four previously authorized old Train partitions are decoded. This is an
inventory of selected/capped references, NOT new folds or full annotation truth.
No visual vector, checkpoint, old final-heldout label or semantic name is read.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from src.trackocd_v2.io import sha256_file

ALLOWED_PARTITIONS = ('representation_fit', 'prototype', 'policy_train', 'development')
LABEL_COLUMNS = ('key', 'category_id', 'partition', 'simulation_role')
INDEX_COLUMNS = ('key', 'video_id', 'observation_count')


def read_registered_known_metadata(cache_root, manifest_sha256):
    import pyarrow.parquet as pq
    root = Path(cache_root)
    path = root / 'manifest.json'
    if sha256_file(path) != manifest_sha256:
        raise ValueError('Registered historical Train manifest changed')
    manifest = json.loads(path.read_text())
    if manifest['source_role'] != 'TAO Train Known GT main training, not predicted/end-to-end':
        raise ValueError('Only declared Train Known reference metadata')
    for name in ('train_labels.parquet', 'index.parquet'):
        payload = root / name
        record = manifest['payloads'][name]
        if payload.is_symlink() or payload.stat().st_size != record['bytes'] or sha256_file(payload) != record['sha256']:
            raise ValueError('Registered metadata payload changed')
    labels = pq.read_table(root / 'train_labels.parquet', columns=list(LABEL_COLUMNS),
                           filters=[('partition', 'in', list(ALLOWED_PARTITIONS))]).to_pylist()
    if any(row['partition'] not in ALLOWED_PARTITIONS for row in labels):
        raise ValueError('Forbidden Train-label partition')
    if len({row['key'] for row in labels}) != len(labels):
        raise ValueError('Duplicate permitted label key')
    eligible = {row['category_id'] for row in labels if row['partition'] == 'representation_fit'}
    selected = [row for row in labels if row['category_id'] in eligible and row['simulation_role'] == 'known']
    if not selected:
        raise ValueError('Empty previously supervised Known pool')
    keys = [row['key'] for row in selected]
    index = pq.read_table(root / 'index.parquet', columns=list(INDEX_COLUMNS),
                          filters=[('key', 'in', keys)]).to_pylist()
    by_key = {row['key']: row for row in index}
    if len(by_key) != len(index) or set(by_key) != set(keys):
        raise ValueError('Missing/duplicate safe metadata join')
    return [{**row, **by_key[row['key']]} for row in selected], {
        'manifest_sha256': manifest_sha256,
        'metadata_payload_sha256': {name: manifest['payloads'][name]['sha256']
                                    for name in ('train_labels.parquet', 'index.parquet')},
        'authorized_filtered_label_rows': len(labels),
        'eligible_old_representation_fit_categories': len(eligible),
        'label_partitions_decoded': list(ALLOWED_PARTITIONS),
        'label_columns_decoded': list(LABEL_COLUMNS), 'index_columns_decoded': list(INDEX_COLUMNS),
        'old_final_heldout_labels_decoded': False, 'feature_arrays_decoded': False}


def support_summary(rows, predicted_cache_videos):
    """No score-based class selection; sorted ordinals anonymize aggregate output."""
    if not rows or len({row['key'] for row in rows}) != len(rows):
        raise ValueError('Nonempty unique metadata rows required')
    grouped = defaultdict(list)
    for row in rows:
        if (row['partition'] not in ALLOWED_PARTITIONS or row['simulation_role'] != 'known'
                or type(row['category_id']) is not int or type(row['video_id']) is not int
                or type(row['observation_count']) is not int or not 1 <= row['observation_count'] <= 16):
            raise ValueError('Only safe selected/capped Known metadata rows')
        grouped[row['category_id']].append(row)

    def statistics(subset):
        counts = Counter(row['video_id'] for row in subset)
        choose2 = lambda n: n * (n - 1) // 2
        return {'selected_capped_GT_tracks': len(subset), 'videos': len(counts),
                'cached_observations': sum(row['observation_count'] for row in subset),
                'singleton_references': sum(row['observation_count'] == 1 for row in subset),
                'cross_video_reference_pairs': choose2(len(subset)) - sum(choose2(n) for n in counts.values())}

    per_category = []
    for ordinal, category in enumerate(sorted(grouped), 1):
        subset = grouped[category]
        intersection = [row for row in subset if row['video_id'] in predicted_cache_videos]
        per_category.append({'anonymous_category_ordinal': ordinal, 'permitted_reference_pool': statistics(subset),
                             'reference_video_intersection_with_existing_predicted_cache': statistics(intersection)})
    return {'categories': len(grouped), 'permitted_reference_pool': statistics(rows),
            'per_category': per_category,
            'category_video_support_threshold_counts': {
                domain: {str(threshold): sum(row[domain]['videos'] >= threshold for row in per_category)
                         for threshold in (2, 3, 5, 7)}
                for domain in ('permitted_reference_pool', 'reference_video_intersection_with_existing_predicted_cache')},
            'category_selection_performed': False, 'new_fold_or_training_protocol_frozen': False,
            'predicted_cache_intersection_is_not_matching_or_predicted_track_denominator': True,
            'full_GT_census': False}
