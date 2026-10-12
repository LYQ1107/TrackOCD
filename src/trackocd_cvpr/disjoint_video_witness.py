"""External protocol bookkeeping: sufficient disjoint-video certificate.

One distinct video per required category slot is deliberately restrictive.
Failure is NOT proof that a more permissive multi-category video assignment
is impossible. No labels or administrative IDs from this module enter models.
"""
from collections import defaultdict


def _validate_inputs(presence, role_classes, role_candidates, minima, max_slots):
    if (not presence or not role_classes or set(role_classes) != set(role_candidates)
            or set(role_classes) != set(minima) or type(max_slots) is not int
            or not 1 <= max_slots <= 512):
        raise ValueError('Bounded complete external protocol inputs required')
    seen_classes = set()
    for video, classes in presence.items():
        if (type(video) is not int or video < 0 or not isinstance(classes, (set, frozenset))
                or not classes or any(type(c) is not int or c < 0 for c in classes)):
            raise ValueError('Positive Known administrative presence only')
    for role, classes in role_classes.items():
        if (not isinstance(role, str) or not role or not isinstance(classes, (set, frozenset))
                or not classes or seen_classes & classes
                or any(type(c) is not int or c < 0 for c in classes)
                or type(minima[role]) is not int or not 2 <= minima[role] <= 16
                or not isinstance(role_candidates[role], (set, frozenset))
                or any(type(v) is not int or v < 0 for v in role_candidates[role])
                or not role_candidates[role] <= presence.keys()):
            raise ValueError('Disjoint nonempty category roles and valid video candidates required')
        seen_classes |= classes
    required = sum(len(classes) * minima[role] for role, classes in role_classes.items())
    if required > max_slots:
        raise ValueError('Certificate slot bound; do not silently trim categories')
    return required


def allocate_witness(presence, role_classes, role_candidates, minima, *, max_slots=512):
    required = _validate_inputs(presence, role_classes, role_candidates, minima, max_slots)
    eligible, per_role, deficits = {}, {}, 0
    for role in sorted(role_classes):
        support_counts = []
        for category in sorted(role_classes[role]):
            videos = tuple(sorted(v for v in role_candidates[role] if category in presence[v]))
            support_counts.append(len(videos))
            deficits += len(videos) < minima[role]
            for repeat in range(minima[role]):
                eligible[(role, category, repeat)] = videos
        per_role[role] = {'classes': len(role_classes[role]), 'minimum_distinct_videos_per_class': minima[role],
                          'candidate_support_counts_sorted': sorted(support_counts)}
    # Scarcity first is deterministic bookkeeping, not model-score selection.
    order = sorted(eligible, key=lambda slot: (len(eligible[slot]), slot))
    assigned_video = {}

    def augment(slot, visited):
        for video in eligible[slot]:
            if video in visited:
                continue
            visited.add(video)
            previous = assigned_video.get(video)
            if previous is None or augment(previous, visited):
                assigned_video[video] = slot
                return True
        return False

    matched = sum(augment(slot, set()) for slot in order)
    private = {video: {'role': slot[0], 'category_id': slot[1], 'required_slot': slot[2]}
               for video, slot in sorted(assigned_video.items())}
    if matched == required:
        verify_witness(presence, role_classes, role_candidates, minima, private)
    public = {'required_category_video_slots': required, 'matched_slots': matched,
              'individual_necessary_support_deficits': deficits, 'roles': per_role,
              'globally_video_disjoint_sufficient_witness_found': matched == required,
              'failure_proves_general_assignment_impossible': False,
              'unmatched_slot_count_is_not_general_infeasibility_lower_bound': True,
              'multiple_classes_per_video_can_make_general_assignment_easier': True,
              'complete_visual_category_absence_proven': False,
              'candidate_semantic_scope_requires_separate_registered_role_filter': True,
              'new_protocol_frozen': False, 'model_executed': False,
              'training_GPU_hours': 0}
    return public, private


def verify_witness(presence, role_classes, role_candidates, minima, witness):
    """Verify actually assigned videos, membership and each class quota."""
    _validate_inputs(presence, role_classes, role_candidates, minima, 512)
    supports, slots = defaultdict(set), set()
    for video, row in witness.items():
        if type(video) is not int or video < 0 or not isinstance(row, dict) or set(row) != {'role', 'category_id', 'required_slot'}:
            raise ValueError('Exact administrative certificate fields required')
        role, category, repeat = row['role'], row['category_id'], row['required_slot']
        if (role not in role_classes or type(category) is not int or category not in role_classes[role]
                or video not in role_candidates[role] or category not in presence[video]
                or type(repeat) is not int or not 0 <= repeat < minima[role]):
            raise ValueError('Certificate role/presence/candidate/slot mismatch')
        slot = (role, category, repeat)
        if slot in slots:
            raise ValueError('Duplicate required slot')
        slots.add(slot)
        supports[(role, category)].add(video)
    expected = {(role, category, repeat) for role, classes in role_classes.items()
                for category in classes for repeat in range(minima[role])}
    if slots != expected or any(len(supports[(role, category)]) < minima[role]
                                for role, classes in role_classes.items() for category in classes):
        raise ValueError('Incomplete global disjoint certificate; no quota relaxation')
    return True
