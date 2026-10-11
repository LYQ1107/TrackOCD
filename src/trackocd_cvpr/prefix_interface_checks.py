"""Actual invisible-tail mutations of the common observed-prefix interface."""
import numpy as np
from src.trackocd_core.features import prefix_view


def check_prefix_tails(visual, geometry, quality, frames, offsets):
    checks = mutated = 0
    for begin, end in zip(offsets, offsets[1:]):
        arrays = [np.array(a[int(begin):int(end)], copy=True) for a in (visual, geometry, quality, frames)]
        for cap in (1, 2, 4, 8, 16):
            n = min(cap, int(end - begin))
            original = prefix_view(*arrays, n)
            changed = [a.copy() for a in arrays]
            for a in changed[:3]:
                a[n:] = np.nan
            changed[3][n:] = -999
            altered = prefix_view(*changed, n)
            for a, b in zip((original.visual, original.boxes_normalized_xyxy, original.quality, original.elapsed_frames),
                            (altered.visual, altered.boxes_normalized_xyxy, altered.quality, altered.elapsed_frames)):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(original.weighted_mean(), altered.weighted_mean())
            checks += 1
            mutated += n < end - begin
    return {'prefix_checks': checks, 'checks_with_real_tail_mutation': mutated,
            'interface_only_not_dense_online_or_end_to_end_ID_invariance': True}
