"""External deterministic co-occurrence feasibility, NOT a frozen new split.

Only selected/capped permitted reference metadata is used. Candidate video
sets overlap and absent annotations are unknown; PASS is necessary-only.
"""
from collections import defaultdict
import random

from src.trackocd_cvpr.train_protocol_inventory import ALLOWED_PARTITIONS


def cyclic_role_feasibility(rows, *, seed=2093001, folds=4, policy_pseudo_classes=2,
                            minimum_fit_candidate_videos=3, minimum_eval_candidate_videos=2):
    if not rows or len({row['key'] for row in rows}) != len(rows):
        raise ValueError('Nonempty unique permitted reference metadata')
    for row in rows:
        if (row['partition'] not in ALLOWED_PARTITIONS or row['simulation_role'] != 'known'
                or type(row['category_id']) is not int or type(row['video_id']) is not int):
            raise ValueError('Authorized Known Train metadata only')
    if (type(seed) is not int or type(folds) is not int or folds < 3
            or type(policy_pseudo_classes) is not int or policy_pseudo_classes < 1
            or min(minimum_fit_candidate_videos, minimum_eval_candidate_videos) < 2):
        raise ValueError('Bounded fixed-role metadata feasibility parameters')
    categories = sorted({row['category_id'] for row in rows})
    ordinal = {category: index + 1 for index, category in enumerate(categories)}
    shuffled = categories.copy()
    random.Random(seed).shuffle(shuffled)
    groups = [set(shuffled[index::folds]) for index in range(folds)]
    if any(not group for group in groups):
        raise ValueError('Nonempty distinct pseudo-category groups required')
    presence, by_category = defaultdict(set), defaultdict(set)
    for row in rows:
        presence[row['video_id']].add(row['category_id'])
        by_category[row['category_id']].add(row['video_id'])
    output = []
    for fold in range(folds):
        final_pseudo = groups[fold]
        dev_pseudo = groups[(fold + 1) % folds]
        remaining = [category for category in shuffled if category not in final_pseudo | dev_pseudo]
        if len(remaining) < policy_pseudo_classes + 2:
            raise ValueError('Need distinct fitting and supervised-policy pseudo categories')
        policy_pseudo = set(remaining[-policy_pseudo_classes:])
        fit_known = set(remaining[:-policy_pseudo_classes])
        roles = {'fit_known': fit_known, 'policy_train_pseudo': policy_pseudo,
                 'development_pseudo': dev_pseudo, 'independent_final_pseudo': final_pseudo}
        forbidden = {'fit_known': policy_pseudo | dev_pseudo | final_pseudo,
                     'policy_train_pseudo': dev_pseudo | final_pseudo,
                     'development_pseudo': policy_pseudo | final_pseudo,
                     'independent_final_pseudo': policy_pseudo | dev_pseudo}
        candidates = {role: {video for video, seen in presence.items() if not seen & excluded}
                      for role, excluded in forbidden.items()}
        summaries = {}
        deficit_count = 0
        for role, members in roles.items():
            minimum = minimum_fit_candidate_videos if role == 'fit_known' else minimum_eval_candidate_videos
            per_class = []
            for category in sorted(members):
                count = len(by_category[category] & candidates[role])
                deficit_count += count < minimum
                per_class.append({'anonymous_category_ordinal': ordinal[category],
                                  'candidate_videos': count, 'minimum_necessary_videos': minimum,
                                  'necessary_support_sufficient': count >= minimum})
            summaries[role] = {'categories': len(members), 'candidate_videos': len(candidates[role]),
                               'observed_conflict_excluded_videos': len(presence) - len(candidates[role]),
                               'per_class': per_class}
        output.append({'fold': fold, 'roles': summaries, 'necessary_support_deficits': deficit_count,
                       'necessary_support_only_PASS': deficit_count == 0,
                       'global_video_disjoint_assignment_proven': False,
                       'complete_category_absence_in_video_proven': False})
    return {'fixed_seed': seed, 'folds': output, 'all_source_categories_final_pseudo_once': True,
            'class_or_seed_selected_from_metric_or_support_results': False,
            'new_training_protocol_frozen': False, 'selected_reference_metadata_only': True,
            'necessary_condition_does_not_certify_full_GT_or_global_video_allocation': True}
