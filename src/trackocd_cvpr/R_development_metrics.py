"""External Train-development labels and certified cross-video retrieval.

No label/role/ID to model.forward. No new raw annotation / nonKnown label
decoding, no final/Val/Test selector, no inference of unknown as background.
Actual calls need prior frozen source/hash/authorization and cache gates.
"""
import mmap
from math import comb
from collections import Counter, defaultdict
import numpy as np
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value


def project_dev_labels(assignment_path, known_path, *, expected_fit_classes=5, expected_dev_classes=8,
                       expected_videos=72, expected_known_tracks=302, expected_pseudo_tracks=119):
    with open(assignment_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        assignment = object_ranges(data, *top_ranges(data)['assignment'])
        class_roles = object_ranges(data, *assignment['classes'])
        video_roles = object_ranges(data, *assignment['video_ids'])
        fit = [integer_value(data, b) for b in array_ranges(data, *class_roles['fit_known'])]
        dev = [integer_value(data, b) for b in array_ranges(data, *class_roles['development_pseudo'])]
        videos = [integer_value(data, b) for b in array_ranges(data, *video_roles['development_pseudo'])]
    if (len(fit) != expected_fit_classes or len(dev) != expected_dev_classes or len(videos) != expected_videos
            or len(set(fit + dev)) != len(fit + dev) or len(set(videos)) != len(videos)):
        raise ValueError('Exact disjoint existingKnown class / development video scope')
    labels = {}; counts = Counter(); selected = set(videos)
    with open(known_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for bounds in array_ranges(data, *top_ranges(data)['tracks']):
            fields = object_ranges(data, *bounds)
            video = integer_value(data, fields['video_id'])
            if video not in selected:
                continue  # before category, trackID or support-count value decode
            category = integer_value(data, fields['category_id'])
            kind = 'Known' if category in fit else 'declared_pseudoNovel' if category in dev else 'outside_eligible_Known'
            track = integer_value(data, fields['track_id']); key = (video, track)
            if key in labels: raise ValueError('Unique development external physical key')
            labels[key] = (kind, category if kind != 'outside_eligible_Known' else None)
            counts[kind] += 1
    if counts['Known'] != expected_known_tracks or counts['declared_pseudoNovel'] != expected_pseudo_tracks:
        raise ValueError('Fixed development supervised-label denominators, no subset relaxation')
    return labels, {'counts': dict(counts), 'fit_policy_final_or_other_video_target_values_decoded': False,
                    'new_nonKnown_semantic_values_or_vocabulary_decoded': False,
                    'external_evaluator_only_not_model_or_loss_input': True}


def reconcile_dev_labels(video_track_keys, projected, *, expected_physical=472, expected_unlabeled=49, expected_outside=2):
    keys = tuple(video_track_keys)
    if len(keys) != expected_physical or len(set(keys)) != len(keys) or not set(projected) <= set(keys):
        raise ValueError('Every physical track retained, all target keys join')
    result = [projected.get(key, ('no_retained_semantic_truth', None)) for key in keys]
    counts = Counter(kind for kind, _ in result)
    if counts['no_retained_semantic_truth'] != expected_unlabeled or counts['outside_eligible_Known'] != expected_outside:
        raise ValueError('Unknown/otherKnown retained, not invented Novel/BG truth')
    return result, dict(counts)


def _certified_topk(similarity, correct, k):
    """Uniform exact-score tie expectation; missing truth never certified TP.

    Unknown gallery objects are NOT asserted false/background. This is a
    certified lower bound from existing semantic truth, explicitly labeled.
    """
    if not len(similarity): return 0.
    k = min(k, len(similarity)); cutoff = np.partition(similarity, -k)[-k]
    strict = similarity > cutoff; boundary = similarity == cutoff
    if np.any(correct[strict]): return 1.
    slots = k - int(strict.sum()); total = int(boundary.sum()); positive = int(correct[boundary].sum())
    if not positive: return 0.
    return 1. - (comb(total - positive, slots) / comb(total, slots) if total - positive >= slots else 0.)


def certified_cross_video_retrieval(embeddings, videos, external_labels, query_kind='declared_pseudoNovel'):
    """All physical gallery tracks, no samevideo, no target/role model features.

    Include all declared query targets, even no valid cross-video positive.
    Return lower-bound micro/macro R1/R5, unscorable top ties and denominators.
    This is not purity, H-score, physical MOT or final experimental acceptance.
    """
    embeddings = np.asarray(embeddings, np.float64); videos = tuple(videos); labels = tuple(external_labels)
    n = len(videos)
    if (embeddings.shape != (n, 768) or len(labels) != n or not np.isfinite(embeddings).all()
            or np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12)):
        raise ValueError('Complete aligned finite common768 embeddings / external evaluator bindings')
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = normalized @ normalized.T
    values = defaultdict(list); class_values = defaultdict(lambda: defaultdict(list))
    no_positive = unknown_top = queries = 0
    for i, (kind, category) in enumerate(labels):
        if kind != query_kind: continue
        if category is None: raise ValueError('Declared semantic queries require retained lawful truth')
        gallery = np.asarray([j for j in range(n) if videos[j] != videos[i]], np.int64)
        similarities = scores[i, gallery]
        correct = np.asarray([labels[j][1] is not None and labels[j][1] == category for j in gallery], bool)
        if not correct.any(): no_positive += 1
        if len(gallery):
            top = gallery[similarities == similarities.max()]
            unknown_top += int(any(labels[j][1] is None for j in top))
        for k in (1, 5):
            value = _certified_topk(similarities, correct, k)
            values[k].append(value); class_values[category][k].append(value)
        queries += 1
    if not queries: raise ValueError('No frozen declared query population')
    return {'physical_gallery_tracks': n, 'declared_query_tracks': queries, 'declared_query_classes': len(class_values),
            'queries_without_verified_cross_video_positive_retained': no_positive,
            'queries_with_semantic_unknown_in_top_tie': unknown_top,
            'certified_lower_bound_micro_R1': float(np.mean(values[1])),
            'certified_lower_bound_micro_R5': float(np.mean(values[5])),
            'certified_lower_bound_macro_R1': float(np.mean([np.mean(v[1]) for v in class_values.values()])),
            'certified_lower_bound_macro_R5': float(np.mean([np.mean(v[5]) for v in class_values.values()])),
            'samevideo_excluded': True, 'unknown_gallery_declared_background_or_labeled_Novel': False,
            'exact_tie_uniform_expectation_no_ID_ordering': True,
            'not_complete_semantic_GT_purity_Hscore_or_final_acceptance': True}
