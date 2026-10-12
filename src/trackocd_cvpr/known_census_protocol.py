"""External positive-Known protocol hypotheses, not model-facing labels."""
from collections import defaultdict
import random

from .disjoint_video_witness import allocate_witness

ROLES = ('fit_known', 'policy_train_pseudo', 'development_pseudo', 'independent_final_pseudo')


def known_presence(index, allowed, known):
    if (set(index) != {'schema_version', 'not_model_facing', 'tracks', 'video_positive_Known_categories'}
            or index['schema_version'] != 'trackocd.cvpr.private-known-positive-census.v1'
            or index['not_model_facing'] is not True or not isinstance(index['tracks'], list)):
        raise ValueError('Exact private positive Known census schema required')
    presence, seen, observations = defaultdict(set), set(), 0
    for row in index['tracks']:
        if (set(row) != {'video_id', 'track_id', 'category_id', 'annotation_count', 'distinct_annotated_images'}
                or any(type(value) is not int or value < 0 for value in row.values())
                or row['video_id'] not in allowed or row['category_id'] not in known
                or row['annotation_count'] == 0 or row['annotation_count'] != row['distinct_annotated_images']):
            raise ValueError('Allowlisted Known track census only; no repair or dropping')
        key = (row['video_id'], row['track_id'])
        if key in seen:
            raise ValueError('Duplicate census track')
        seen.add(key)
        presence[row['video_id']].add(row['category_id'])
        observations += row['annotation_count']
    declared = index['video_positive_Known_categories']
    expected = {str(v): sorted(cs) for v, cs in presence.items()}
    if (declared != expected or any(type(c) is not int for cs in declared.values() for c in cs)
            or not presence):
        raise ValueError('Private positive presence must exactly reconcile tracks')
    return dict(presence), {'tracks': len(seen), 'observations': observations, 'videos': len(presence),
                            'positive_Known_classes': len(set().union(*presence.values()))}


def role_candidates(presence, classes):
    if set(classes) != set(ROLES):
        raise ValueError('Exact four external roles required')
    all_classes = set()
    for cs in classes.values():
        if not isinstance(cs, (set, frozenset)) or not cs or all_classes & cs:
            raise ValueError('Nonempty class-disjoint roles required')
        all_classes |= cs
    f, p, d, h = (classes[role] for role in ROLES)
    forbidden = {'fit_known': p | d | h, 'policy_train_pseudo': d | h,
                 'development_pseudo': p | h, 'independent_final_pseudo': p | d}
    return {role: {v for v, seen in presence.items() if not seen & denied}
            for role, denied in forbidden.items()}


def fixed_hypotheses(presence, *, seed=2093001, folds=4, policy_classes=2,
                     minimum_fit=3, minimum_eval=2, eligibility_video_support=0):
    if (type(seed) is not int or folds != 4 or policy_classes != 2
            or minimum_fit != 3 or minimum_eval != 2
            or type(eligibility_video_support) is not int or eligibility_video_support not in (0, 3)):
        raise ValueError('Only the two declared fixed hypotheses; no adaptive seed/threshold search')
    by_category = defaultdict(set)
    for video, categories in presence.items():
        if type(video) is not int or video < 0 or not isinstance(categories, (set, frozenset)) or not categories:
            raise ValueError('Valid Known positive presence required')
        for category in categories:
            if type(category) is not int or category < 0:
                raise ValueError('Known numeric membership only')
            by_category[category].add(video)
    pool = sorted(c for c, videos in by_category.items() if len(videos) >= eligibility_video_support)
    if len(pool) < 8:
        raise ValueError('Declared class-pool capacity fails; do not trim roles or quotas')
    shuffled = pool.copy()
    random.Random(seed).shuffle(shuffled)
    groups = [set(shuffled[n::folds]) for n in range(folds)]
    outputs, private = [], []
    for fold in range(folds):
        final, dev = groups[fold], groups[(fold + 1) % folds]
        remaining = [c for c in shuffled if c not in final | dev]
        if len(remaining) < policy_classes + 2:
            raise ValueError('Fixed role pool insufficient; preserve failure')
        classes = dict(zip(ROLES, (set(remaining[:-policy_classes]), set(remaining[-policy_classes:]), dev, final)))
        candidates = role_candidates(presence, classes)
        minima = {role: minimum_fit if role == 'fit_known' else minimum_eval for role in ROLES}
        public, witness = allocate_witness(presence, classes, candidates, minima)
        public.update(fold=fold, candidate_scope='Exclude observed positive Known conflicts for declared pseudo roles; Known fit classes may occur in evaluation videos')
        outputs.append(public)
        private.append({'fold': fold, 'classes': {role: sorted(cs) for role, cs in classes.items()},
                        'candidate_video_ids': {role: sorted(vs) for role, vs in candidates.items()},
                        'partial_or_full_video_witness': witness,
                        'not_a_frozen_protocol': True})
    return {'fixed_seed': seed, 'eligibility_video_support': eligibility_video_support,
            'positive_supported_Known_classes': len(by_category), 'class_pool': len(pool),
            'classes_outside_pool': len(by_category) - len(pool),
            'outside_pool_not_declared_background': True,
            'every_pool_class_final_pseudo_exactly_once': True,
            'all_folds_retained_no_successful_fold_or_metric_selection': True,
            'observed_positive_scope_only_not_complete_visual_absence': True,
            'new_protocol_frozen': False, 'folds': outputs}, private
