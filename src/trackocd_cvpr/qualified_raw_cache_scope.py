"""Explicit process-local CPU Raw reduction overlay, no controller edits.

Only registered read-only workers may use after numeric qualification. The
original numeric input function remains the fallback for non-CPU; no learned
embedding, prediction/GT metadata, tensor references or member eviction.
"""
from contextlib import contextmanager
from .prefix_member_bank import CommittedRawSupport
from .reliability_support_bridge import committed_reliability_inputs as original_inputs
from .frozen_raw_statistics_cache import FrozenRawStatisticsCache


@contextmanager
def qualified_raw_cache_scope():
    from . import frame_atomic_state_ema as ema
    from . import versioned_reliability_memory as reliability
    if ema.committed_reliability_inputs is not original_inputs or reliability.committed_reliability_inputs is not original_inputs:
        raise ValueError('Exact original input bindings; no nested/unknown runtime patch')
    cache = FrozenRawStatisticsCache()
    def inputs(snapshot, selected_token, *, device='cpu'):
        if device != 'cpu': return original_inputs(snapshot, selected_token, device=device)
        snapshot = tuple(snapshot)
        if (len(snapshot) > 4096 or type(selected_token) is not str
                or any(type(row) is not CommittedRawSupport for row in snapshot)
                or len({row.token for row in snapshot}) != len(snapshot)):
            raise ValueError('Same original immutable snapshot/binding contract')
        groups = next((row.payloads for row in snapshot if row.token == selected_token), None)
        if groups is None: raise ValueError('Selected support absent, no alternate binding')
        return cache.inputs(groups)
    ema.committed_reliability_inputs = reliability.committed_reliability_inputs = inputs
    try:
        yield cache
    finally:
        ema.committed_reliability_inputs = reliability.committed_reliability_inputs = original_inputs
