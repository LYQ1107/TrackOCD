"""External Known-only support audit; no model labels or protocol freeze.

Extend a verified global witness to every uniquely legal, positive-own-role
video. Classes remain fixed; no ranking, metric, random tie break or GT repair.
Ambiguous/unusable videos are counted and retained outside the assignment.
"""
from collections import defaultdict
from .known_census_protocol import ROLES, role_candidates
from .joint_known_protocol import verify_global_assignment


def verified_global_wrapper_assignment(value):
    if (type(value) is not dict or set(value) != {'not_model_facing', 'not_a_frozen_protocol', 'assignment'}
            or value['not_model_facing'] is not True or value['not_a_frozen_protocol'] is not True):
        raise ValueError('Exact external non-frozen global assignment wrapper')
    return value['assignment']


def audit_global_denominators(presence, index, wrapper):
    """Every retained Known row in assigned videos, not only own-role targets.

    This cannot certify full GT denominators: non-Known GT was never retained
    by the authorized census. No new labels/boxes are inferred or opened here.
    """
    if (type(wrapper) is not dict or set(wrapper) != {'not_model_facing', 'not_a_frozen_protocol',
                                                    'assignment', 'role_class_support'}
            or wrapper['not_model_facing'] is not True or wrapper['not_a_frozen_protocol'] is not True):
        raise ValueError('Exact external expanded non-frozen wrapper')
    assignment = wrapper['assignment']
    verify_global_assignment(presence, assignment)
    classes = {r: set(cs) for r, cs in assignment['classes'].items()}
    pool = set().union(*classes.values())
    by_video = {v: role for role, vs in assignment['video_ids'].items() for v in vs}
    result = {role: {kind: {'tracks': 0, 'observations': 0} for kind in
        ('fit_Known', 'own_role_pseudo', 'other_role_pseudo_conflict', 'outside_eligible_Known')}
        for role in ROLES}
    unassigned_tracks = unassigned_observations = 0
    for row in index['tracks']:
        role = by_video.get(row['video_id'])
        if role is None:
            unassigned_tracks += 1; unassigned_observations += row['annotation_count']
            continue
        c = row['category_id']
        if c in classes[ROLES[0]]:
            kind = 'fit_Known'
        elif c not in pool:
            kind = 'outside_eligible_Known'
        elif c in classes[role]:
            kind = 'own_role_pseudo'
        else:
            kind = 'other_role_pseudo_conflict'
        result[role][kind]['tracks'] += 1
        result[role][kind]['observations'] += row['annotation_count']
    if any(result[r]['other_role_pseudo_conflict']['tracks'] for r in ROLES):
        raise ValueError('Cross-role positive conflicts in full retained Known census')
    for role in ROLES:
        result[role]['all_retained_Known'] = {
            k: sum(row[k] for row in result[role].values()) for k in ('tracks', 'observations')}
    total_tracks = sum(row['all_retained_Known']['tracks'] for row in result.values()) + unassigned_tracks
    total_obs = sum(row['all_retained_Known']['observations'] for row in result.values()) + unassigned_observations
    if total_tracks != len(index['tracks']) or total_obs != sum(row['annotation_count'] for row in index['tracks']):
        raise ValueError('Every original Known row must reconcile')
    return {'status': 'VERIFIED_RETAINED_KNOWN_DENOMINATORS_FULL_GT_STILL_UNPROVEN',
            'integer_necessary_conditions': {'class_pool': len(pool)},
            'role_denominators': result,
            'unassigned_retained_Known': {'tracks': unassigned_tracks, 'observations': unassigned_observations},
            'reconciled_Known_tracks': total_tracks, 'reconciled_Known_observations': total_obs,
            'full_GT_denominator_proven': False, 'nonKnown_GT_for_assigned_videos_retained': False,
            'outside_eligible_Known_declared_background': False,
            'Known_rows_are_not_all_physical_ground_truth': True,
            'new_protocol_frozen': False, 'training_GPU_hours': 0,
            'pixel_feature_availability_proven': False,
            'model_scores_or_names_used': False}, {'not_model_facing': True,
                'not_a_frozen_protocol': True, 'assignment_unchanged': True}


def _support(index, assignment):
    output, private = {}, {}
    for role in ROLES:
        classes, videos = set(assignment['classes'][role]), set(assignment['video_ids'][role])
        rows = [row for row in index['tracks'] if row['category_id'] in classes and row['video_id'] in videos]
        by_class = defaultdict(list)
        for row in rows:
            by_class[row['category_id']].append(row)
        per_class = []
        for c in sorted(classes):
            tracks = by_class[c]
            supported = {row['video_id'] for row in tracks}
            observations = sum(row['annotation_count'] for row in tracks)
            cross = sum(bool(supported - {row['video_id']}) for row in tracks)
            per_class.append({'category_id': c, 'tracks': len(tracks), 'observations': observations,
                              'positive_videos': len(supported), 'possible_cross_video_query_tracks': cross})
        def bounds(key):
            values = [row[key] for row in per_class]
            return {'min': min(values), 'max': max(values), 'sum': sum(values)}
        output[role] = {'classes': len(classes), 'assigned_videos': len(videos),
                        'tracks': bounds('tracks'), 'observations': bounds('observations'),
                        'positive_videos_per_class': bounds('positive_videos'),
                        'possible_cross_video_query_tracks': bounds('possible_cross_video_query_tracks'),
                        'single_annotation_tracks': sum(row['annotation_count'] == 1 for row in rows)}
        private[role] = per_class
    return output, private


def audit_global_coverage(presence, index, assignment):
    verify_global_assignment(presence, assignment)
    classes = {role: set(assignment['classes'][role]) for role in ROLES}
    candidates = role_candidates(presence, classes)
    prior = set(v for vs in assignment['video_ids'].values() for v in vs)
    expanded = {'classes': {r: list(assignment['classes'][r]) for r in ROLES},
                'video_ids': {r: list(assignment['video_ids'][r]) for r in ROLES}}
    added = {r: 0 for r in ROLES}
    ambiguous = no_role = 0
    for video in sorted(set(presence) - prior):
        eligible = [r for r in ROLES if video in candidates[r] and presence[video] & classes[r]]
        if len(eligible) == 1:
            role = eligible[0]
            expanded['video_ids'][role].append(video)
            added[role] += 1
        elif eligible:
            ambiguous += 1
        else:
            no_role += 1
    expanded['video_ids'] = {r: sorted(vs) for r, vs in expanded['video_ids'].items()}
    verify_global_assignment(presence, expanded)
    before, _ = _support(index, assignment)
    after, detail = _support(index, expanded)
    return {'status': 'VERIFIED_FIXED_CLASS_COVERAGE_EXTENSION_NOT_FROZEN_PROTOCOL',
            'integer_necessary_conditions': {'class_pool': sum(len(cs) for cs in classes.values())},
            'all_original_videos_retained_as_candidates': len(presence),
            'base_assigned_videos': len(prior), 'added_unique_role_videos': added,
            'expanded_assigned_videos': len(prior) + sum(added.values()),
            'unassigned_ambiguous_positive_role_videos': ambiguous,
            'unassigned_no_legal_positive_own_role_videos': no_role,
            'base_support': before, 'expanded_support': after,
            'class_assignment_changed': False, 'model_scores_or_names_used': False,
            'numeric_ID_or_difficulty_tie_break': False, 'new_protocol_frozen': False,
            'possible_queries_are_annotated_track_potential_not_decoded_features_or_online_results': True,
            'complete_visual_category_absence_proven': False, 'training_GPU_hours': 0}, {
                'not_model_facing': True, 'assignment': expanded, 'role_class_support': detail,
                'not_a_frozen_protocol': True}
