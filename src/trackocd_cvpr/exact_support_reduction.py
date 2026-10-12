"""Current-weight equal-member support reduction, no chunk prototype averaging.

This is a bounded numerical prerequisite, not a trainer or scalability result.
All member graphs remain live until backward; chunk traversal is NOT a claim
of bounded autograd activation memory. A run still needs a real RSS/GPU guard.
No pruning, adapted-prototype cache, label, identity or evaluator input.
"""
import torch
from .raw_preserving_evidence import RawPreservingEvidence, unit
from .differentiable_support_bank import VisiblePrefix

MAX_GROUPS = 4096
MAX_OBSERVATIONS = 65536
MAX_CHUNK_OBSERVATIONS = 16384


def reencode_exact_supports(model, candidate_groups, known_flags, binding_counts, *, chunk_observations=MAX_CHUNK_OBSERVATIONS):
    """Sum original unit members, then normalize ONCE per full candidate.

    Never average normalized chunk prototypes: that changes relative member
    weights and can damage gradients. Chunks do not split a temporal member.
    Known flags are candidate kinds, not query labels; bindings only counts.
    Returns Raw/adapted full prototypes, same five statistics, and count proof.
    """
    n = len(candidate_groups)
    if (not isinstance(model, RawPreservingEvidence) or n > MAX_GROUPS
            or type(chunk_observations) is not int or not 1 <= chunk_observations <= MAX_CHUNK_OBSERVATIONS):
        raise ValueError('Registered bounded current evidence model and traversal size')
    if (known_flags.shape != (n,) or binding_counts.shape != (n,)
            or known_flags.dtype != torch.float32 or binding_counts.dtype != torch.float32
            or known_flags.device != binding_counts.device
            or not torch.isfinite(known_flags).all() or not torch.isfinite(binding_counts).all()
            or torch.any((known_flags != 0) & (known_flags != 1))
            or torch.any(binding_counts < 1) or torch.any(binding_counts != binding_counts.floor())):
        raise ValueError('Numeric candidate kind / binding count only')
    groups = sum(len(group) for group in candidate_groups)
    if groups > MAX_GROUPS or any(not group for group in candidate_groups):
        raise ValueError('Nonempty complete member sets within explicit group bound')
    observations = 0
    for group in candidate_groups:
        for prefix in group:
            if type(prefix) is not VisiblePrefix or prefix.visual.device != known_flags.device:
                raise ValueError('Numeric visible prefixes on the statistics device')
            count = len(prefix.quality)
            if not 1 <= count <= chunk_observations:
                raise ValueError('Whole temporal member must fit a chunk; no observation truncation')
            observations += count
    if observations > MAX_OBSERVATIONS:
        raise ValueError('Whole-call observation bound; reject, never prune/detach')
    raw_rows, adapted_rows, stats = [], [], []
    chunks = 0
    for index, group in enumerate(candidate_groups):
        raw_sum = adapted_sum = quality_sum = None
        chunk_sum = 0
        for prefix in group:
            count = len(prefix.quality)
            if chunk_sum == 0 or chunk_sum + count > chunk_observations:
                chunks += 1; chunk_sum = 0
            chunk_sum += count
            evidence = model(prefix.visual.clone(), prefix.boxes.clone(), prefix.quality.clone(), prefix.elapsed.clone())
            raw_sum = evidence.raw if raw_sum is None else raw_sum + evidence.raw
            adapted_sum = evidence.embedding if adapted_sum is None else adapted_sum + evidence.embedding
            quality_sum = evidence.quality_mean if quality_sum is None else quality_sum + evidence.quality_mean
        raw, adapted = unit(raw_sum), unit(adapted_sum)
        dispersion = (1 - ((raw_sum / len(group)) * raw).sum()).clamp(0, 1)
        raw_rows.append(raw); adapted_rows.append(adapted)
        stats.append(torch.stack((known_flags[index], dispersion, raw.new_tensor(len(group)),
                                  quality_sum / len(group), binding_counts[index])))
    proof = {'groups_reencoded': groups, 'observations_reencoded': observations, 'whole_member_chunks': chunks,
             'equal_member_reduction': True, 'adapted_gradients_not_detached': True,
             'chunk_traversal_not_activation_memory_bound': True, 'not_trainer_or_real_scale_qualification': True}
    if not n:
        return known_flags.new_empty((0, 768)), known_flags.new_empty((0, 768)), known_flags.new_empty((0, 5)), proof
    return torch.stack(raw_rows), torch.stack(adapted_rows), torch.stack(stats), proof
