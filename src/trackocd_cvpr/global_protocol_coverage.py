"""External Known-only support audit; no model labels or protocol freeze.

Extend a verified global witness to every uniquely legal, positive-own-role
video. Classes remain fixed; no ranking, metric, random tie break or GT repair.
Ambiguous/unusable videos are counted and retained outside the assignment.
"""
from collections import defaultdict
from .known_census_protocol import ROLES, role_candidates
from .joint_known_protocol import verify_global_assignment


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
