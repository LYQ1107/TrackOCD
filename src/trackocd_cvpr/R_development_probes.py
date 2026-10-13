"""Read-only, qualified observed-prefix numeric probes, not scientific gains.

No loader/GT labels, no dataset writes. Caller supplies already lawful tracks
and a fixed frozen model. These are one-track observed-input probes, NOT full
online controller/cadence proofs or a guarantee of temporal benefit.
"""
import numpy as np
import torch
from .fit_R_training import FitTrack
from .R_development_evaluation import embed_complete_role


def numeric_probes(model, tracks, guard, *, timeless):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen endpoint only, no training in probes')
    selected = next((t for t in tracks if t.count() >= 2), None)
    if selected is None:
        return {'status': 'NO_FUTURE_OBSERVATIONS_NOT_PROOF', 'future_rows_mutated': 0}
    first, end = map(int, selected.arrays['offsets'][selected.position:selected.position + 2])
    arrays = dict(selected.arrays); arrays['visual'] = arrays['visual'].copy()
    original_future = arrays['visual'][first + 1:end].copy()
    arrays['visual'][first + 1:end] = 0; arrays['visual'][first + 1:end, 0] = 1
    changes = int(np.any(arrays['visual'][first + 1:end] != original_future, axis=1).sum())
    if not changes: raise ValueError('Actual future probe must change numeric data')
    changed = FitTrack(selected.video, arrays, selected.position, None)
    raw, adapted, actual = embed_complete_role(model, [selected], 1, guard)
    new_raw, new_adapted, new_actual = embed_complete_role(model, [changed], 1, guard)
    if not np.array_equal(raw, new_raw) or not np.array_equal(adapted, new_adapted) or actual != new_actual:
        raise ValueError('Sealed visible prefix depends on actual future payload')
    visible = selected.visible(min(4, selected.count()))
    reversed_visual, reversed_boxes, reversed_quality = (a.flip(0) for a in (visible.visual, visible.boxes, visible.quality))
    compressed = torch.arange(len(visible.elapsed), dtype=torch.int64)
    with torch.inference_mode():
        original = model(visible.visual, visible.boxes, visible.quality, visible.elapsed)
        ordered = model(reversed_visual, reversed_boxes, reversed_quality, visible.elapsed)
        timed = model(visible.visual, visible.boxes, visible.quality, compressed)
    if timeless:
        if (not torch.equal(original.embedding, ordered.embedding) or not torch.equal(original.raw, ordered.raw)
                or not torch.equal(original.embedding, timed.embedding) or not torch.equal(original.raw, timed.raw)):
            raise ValueError('Actual timeless endpoint consumes order/time despite registered contract')
    return {'status': 'ACTUAL_ONE_TRACK_OBSERVED_INPUT_PROBES_NOT_FULL_ONLINE_PROOF',
        'future_rows_mutated': changes, 'actual_first_prefix_future_invariant': True,
        'order_observations_in_probe': len(visible.elapsed),
        'order_numeric_rows_changed': not torch.equal(visible.visual, reversed_visual) or not torch.equal(visible.boxes, reversed_boxes),
        'legal_compressed_time_values_changed': not torch.equal(visible.elapsed, compressed),
        'timeless_exact_order_and_time_invariance_required_and_passed': bool(timeless),
        'order_embedding_l2_change': float((original.embedding - ordered.embedding).norm()),
        'time_embedding_l2_change': float((original.embedding - timed.embedding).norm()),
        'temporal_information_benefit_or_scientific_acceptance': False}
