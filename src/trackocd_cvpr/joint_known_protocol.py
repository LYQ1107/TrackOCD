"""Finite structural joint class/video allocation, external Known bookkeeping.

No model scores, names, extra category pool, frequency exceptions or GT repair.
Integer necessary-condition certificate precedes one bounded feasibility MILP.
"""
from collections import defaultdict
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .known_census_protocol import ROLES, role_candidates


def category_support(presence):
    if not presence or len(presence) > 500:
        raise ValueError('Nonempty bounded positive Known presence required')
    support = defaultdict(set)
    for video, categories in presence.items():
        if (type(video) is not int or video < 0 or not isinstance(categories, (set, frozenset))
                or not categories or any(type(c) is not int or c < 0 for c in categories)):
            raise ValueError('Positive Known administrative IDs only')
        for category in categories:
            support[category].add(video)
    if len(support) > 48:
        raise ValueError('Bounded48positive Known census; no silent class dropping')
    pool = sorted(c for c, videos in support.items() if len(videos) >= 3)
    if len(pool) < 8:
        raise ValueError('Original uniformly eligible class pool cannot fill four roles')
    return support, pool


def forced_group_certificate(presence, *, union_support=False):
    """If |V_X minus V_Y|<2, X must share Y's final-pseudo group.

In Y's final fold every non-final role excludes positive Y, and every role
requires >=2videos. Equality is transitive, so a component larger than any
balanced group cannot fit. This is a scoped integer proof, not MILP status.
"""
    support, pool = category_support(presence)
    parent, edges = {c: c for c in pool}, []
    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for first in pool:
        for second in pool:
            if first == second:
                continue
            count = len(support[first] - support[second])
            if count < 2:
                edges.append({'category_X': first, 'category_Y': second,
                              'X_videos_without_Y': count})
                a, b = root(first), root(second)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for c in pool:
        groups[root(c)].append(c)
    components = sorted(groups.values(), key=lambda cs: (-len(cs), cs))
    direct_sizes = [len(cs) for cs in components]
    union_witnesses = []
    if union_support:
        # Every member of an already-forced component is final together. In
        # that fold every other role excludes their entire positive union.
        # One merge per round; at most pool-1 merges, no solver/seed search.
        while True:
            changed = False
            for group in components:
                denied = set().union(*(support[c] for c in group))
                for other in components:
                    if other == group:
                        continue
                    forcing = next((c for c in other if len(support[c] - denied) < 2), None)
                    if forcing is None:
                        continue
                    union_witnesses.append({'category_X': forcing, 'already_forced_group': group.copy(),
                                            'X_videos_without_group_union': len(support[forcing] - denied)})
                    components = sorted([cs for cs in components if cs != group and cs != other]
                                        + [sorted(group + other)], key=lambda cs: (-len(cs), cs))
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                break
    sizes = [len(cs) for cs in components]
    balanced_max = (len(pool) + 3) // 4
    # Even without balanced groups, final component + nonempty development +
    # two policy classes + at least two fit classes needs at least S+5classes.
    unbalanced_max = len(pool) - 5
    public = {'class_pool': len(pool), 'forced_equality_edges': len(edges),
            'forced_component_sizes_sorted': sizes, 'balanced_group_max_classes': balanced_max,
            'balanced_family_integer_capacity_contradiction': sizes[0] > balanced_max,
            'nonempty_four_role_unbalanced_capacity_contradiction': sizes[0] > unbalanced_max,
            'unbalanced_final_capacity_ceiling_with_dev1_policy2_fit2': unbalanced_max,
            'scope': 'Same eligible Known pool, every class final-pseudo once, >=2videos in every role; Y-positive videos excluded from every non-final role in Y final fold; not arbitrary other protocol/data'}
    private = {
                'components': components, 'directed_support_witnesses': edges,
                'not_model_facing': True}
    if union_support:
        public.update(union_group_closure_enabled=True, direct_component_sizes_before_union=direct_sizes,
                      union_support_forced_merges=len(union_witnesses))
        private['union_forced_group_witnesses'] = union_witnesses
    return public, private


def verify_assignment(presence, assignment, *, balanced=True):
    """Independent combinatorial verification, not just solver residuals."""
    support, pool = category_support(presence)
    if set(assignment) != {'groups', 'folds'} or len(assignment['groups']) != 4 or len(assignment['folds']) != 4:
        raise ValueError('Complete four-group/four-fold witness required')
    if any(not isinstance(cs, list) or any(type(c) is not int or c < 0 for c in cs) for cs in assignment['groups']):
        raise ValueError('Exact integer category memberships required')
    groups = [set(cs) for cs in assignment['groups']]
    if (sorted(c for cs in groups for c in cs) != pool
            or any(len(cs) != len(assignment['groups'][i]) for i, cs in enumerate(groups))
            or any(not (len(pool) // 4 if balanced else 1) <= len(cs) <= ((len(pool) + 3) // 4 if balanced else len(pool) - 5) for cs in groups)):
        raise ValueError('Every eligible class exactly once in declared final-group capacity')
    for f, fold in enumerate(assignment['folds']):
        if set(fold) != {'classes', 'video_ids'} or set(fold['classes']) != set(ROLES) or set(fold['video_ids']) != set(ROLES):
            raise ValueError('Exact independent witness roles required')
        if any(not isinstance(cs, list) or any(type(c) is not int or c < 0 for c in cs) for cs in fold['classes'].values()):
            raise ValueError('Exact integer role memberships required')
        classes = {role: set(cs) for role, cs in fold['classes'].items()}
        if (any(len(classes[role]) != len(fold['classes'][role]) for role in ROLES)
                or classes['independent_final_pseudo'] != groups[f]
                or classes['development_pseudo'] != groups[(f + 1) % 4]
                or len(classes['policy_train_pseudo']) != 2 or len(classes['fit_known']) < 2
                or sorted(c for cs in classes.values() for c in cs) != pool):
            raise ValueError('Cyclic class-role isolation or cardinality violated')
        candidates = role_candidates(presence, classes)
        seen = set()
        for role in ROLES:
            videos = set(fold['video_ids'][role])
            if (any(type(v) is not int for v in videos) or len(videos) != len(fold['video_ids'][role])
                    or seen & videos or not videos <= candidates[role]):
                raise ValueError('Video overlap, unknown video or positive class conflict')
            seen |= videos
            minimum = 3 if role == 'fit_known' else 2
            if any(len(videos & support[c]) < minimum for c in classes[role]):
                raise ValueError('Insufficient actual distinct-video support')
    return True


def build_joint(presence, *, balanced=True):
    support, pool = category_support(presence)
    videos = sorted(presence)
    n, m = len(pool), len(videos)
    group = lambda c, g: c * 4 + g
    role = lambda c, f, r: n * 4 + (c * 4 + f) * 4 + r
    video = lambda v, f, r: n * 20 + (v * 4 + f) * 4 + r
    variables = n * 20 + m * 16
    rows, cols, values, lower, upper = [], [], [], [], []
    def add(items, lb, ub):
        row = len(lower)
        for col, value in items:
            rows.append(row); cols.append(col); values.append(value)
        lower.append(lb); upper.append(ub)
    for c in range(n):
        add([(group(c, g), 1) for g in range(4)], 1, 1)
        for f in range(4):
            add([(role(c, f, r), 1) for r in range(4)], 1, 1)
            add([(role(c, f, 3), 1), (group(c, f), -1)], 0, 0)
            add([(role(c, f, 2), 1), (group(c, (f + 1) % 4), -1)], 0, 0)
            for r in range(4):
                add([(video(v, f, r), 1) for v, vid in enumerate(videos) if vid in support[pool[c]]]
                    + [(role(c, f, r), -(3 if r == 0 else 2))], 0, np.inf)
    for g in range(4):
        add([(group(c, g), 1) for c in range(n)], n // 4 if balanced else 1, (n + 3) // 4 if balanced else n - 5)
    forbidden = ((1, 2, 3), (2, 3), (1, 3), (1, 2))
    for f in range(4):
        add([(role(c, f, 1), 1) for c in range(n)], 2, 2)
        add([(role(c, f, 0), 1) for c in range(n)], 2, np.inf)
        for v, vid in enumerate(videos):
            add([(video(v, f, r), 1) for r in range(4)], 0, 1)
            for c, category in enumerate(pool):
                if category in presence[vid]:
                    for r in range(4):
                        add([(video(v, f, r), 1)] + [(role(c, f, q), 1) for q in forbidden[r]], -np.inf, 1)
    if variables > 8960 or len(lower) > 60000 or len(values) > 250000:
        raise ValueError('Declared sparse formulation resource bound')
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), variables)).tocsc()
    return matrix, np.array(lower), np.array(upper), pool, videos


def joint_feasibility(presence, *, time_limit=30, node_limit=5000, solve=milp, balanced=True, union_support=False):
    if type(time_limit) not in (int, float) or not 0 < time_limit <= 30 or type(node_limit) is not int or not 1 <= node_limit <= 5000:
        raise ValueError('One bounded registered solver attempt only')
    if type(balanced) is not bool or type(union_support) is not bool:
        raise ValueError('Explicit boolean structural family flags required')
    certificate, private_proof = forced_group_certificate(presence, union_support=union_support)
    common = {'integer_necessary_conditions': certificate,
              'model_scores_or_names_used': False, 'frequency_always_Known_exception': False,
              'complete_visual_category_absence_proven': False, 'new_protocol_frozen': False,
              'training_GPU_hours': 0}
    contradiction = certificate['balanced_family_integer_capacity_contradiction'] if balanced else certificate['nonempty_four_role_unbalanced_capacity_contradiction']
    if contradiction:
        return {**common, 'status': 'INTEGER_CAPACITY_CONTRADICTION_FOR_DECLARED_JOINT_FAMILY',
                'solver_executed': False, 'feasible_joint_assignment_verified': False}, {'proof': private_proof}
    matrix, lower, upper, pool, videos = build_joint(presence, balanced=balanced)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = solve(c=np.zeros(matrix.shape[1]), integrality=np.ones(matrix.shape[1]), bounds=Bounds(0, 1),
                       constraints=LinearConstraint(matrix, lower, upper), options={
                           'time_limit': time_limit, 'node_limit': node_limit, 'mip_rel_gap': 0,
                           'presolve': True, 'threads': 1, 'parallel': False, 'random_seed': 2093001})
    public = {**common, 'solver_executed': True, 'solver_status': int(result.status),
              'solver_message': str(result.message), 'variables': matrix.shape[1], 'constraints': matrix.shape[0],
              'nonzeros': matrix.nnz, 'forwarded_option_warnings': [str(w.message) for w in captured],
              'solver_reported_scoped_infeasibility_not_formal_global_data_proof': result.status == 2,
              'feasible_joint_assignment_verified': False}
    if result.x is None:
        public['status'] = 'SOLVER_SCOPED_INFEASIBLE' if result.status == 2 else 'INCONCLUSIVE_NO_VERIFIED_ASSIGNMENT'
        return public, {'proof': private_proof}
    if result.status not in (0, 1):
        raise ValueError('Unexpected solver status with nonempty witness; preserve failure')
    vector = np.asarray(result.x)
    if (vector.shape != (matrix.shape[1],) or not np.isfinite(vector).all()
            or np.max(np.abs(vector - np.rint(vector))) > 1e-7
            or np.min(vector) < -1e-7 or np.max(vector) > 1 + 1e-7):
        raise ValueError('Solver vector is not a verified binary witness')
    x = np.rint(vector).astype(int)
    n = len(pool)
    groups = [[pool[c] for c in range(n) if x[c * 4 + g]] for g in range(4)]
    folds = []
    for f in range(4):
        folds.append({'classes': {name: [pool[c] for c in range(n) if x[n * 4 + (c * 4 + f) * 4 + r]] for r, name in enumerate(ROLES)},
                      'video_ids': {name: [vid for v, vid in enumerate(videos) if x[n * 20 + (v * 4 + f) * 4 + r]] for r, name in enumerate(ROLES)}})
    assignment = {'groups': groups, 'folds': folds}
    verify_assignment(presence, assignment, balanced=balanced)
    public.update(status='VERIFIED_POSITIVE_KNOWN_JOINT_ASSIGNMENT_NOT_FROZEN_PROTOCOL', feasible_joint_assignment_verified=True)
    return public, {'proof': private_proof, 'assignment': assignment}
