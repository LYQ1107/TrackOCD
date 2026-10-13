"""Source-only exact immutable SHA memo + frozen Raw reduction overlay.

Only own-process FrameAtomicStateEMA incremental support hashes are memoized.
Keys are the COMPLETE ordered byte segments, compared exactly, not a digest,
ID, token, role or label. Actual SHA256 still computes every cache miss. This
does not skip any state/version refresh or memoize learned/EMA prototypes.
Two derived LRUs share the old 64MiB logical ceiling, 32MiB each. Real RSS,
member/history/clock bounds remain separate and mandatory in a real worker.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
from .frozen_raw_statistics_cache import FrozenRawStatisticsCache
from .prefix_member_bank import CommittedRawSupport
from .reliability_support_bridge import committed_reliability_inputs as original_inputs


class ExactSegmentSHA256Cache:
    def __init__(self, *, max_bytes=33554432):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive bounded derived fingerprint cache')
        self._limit, self._rows, self._bytes = max_bytes, OrderedDict(), 0
        self.hits = self.misses = 0

    @property
    def retained_bytes(self): return self._bytes

    def digest(self, pieces):
        key = tuple(pieces)
        if any(type(piece) is not bytes for piece in key):
            raise ValueError('Complete immutable ordered byte segments only')
        if key in self._rows:
            self.hits += 1; self._rows.move_to_end(key)
            return self._rows[key]
        self.misses += 1
        h = hashlib.sha256()
        for piece in key: h.update(piece)
        result = h.digest()
        size = sum(map(len, key)) + 8 * len(key) + len(result) + 16
        if size > self._limit: return result
        while self._rows and self._bytes + size > self._limit:
            old, value = self._rows.popitem(last=False)
            self._bytes -= sum(map(len, old)) + 8 * len(old) + len(value) + 16
        self._rows[key] = result; self._bytes += size
        return result


class _DeferredSHA256:
    def __init__(self, cache): self._cache, self._pieces = cache, []
    def update(self, value):
        if type(value) is not bytes: raise ValueError('Immutable support hash updates only')
        self._pieces.append(value)
    def digest(self): return self._cache.digest(self._pieces)
    def hexdigest(self): return self.digest().hex()
    def copy(self):
        other = type(self)(self._cache); other._pieces = list(self._pieces)
        return other


class _SHA256Proxy:
    def __init__(self, cache): self._cache = cache
    def sha256(self, data=b'', **kwargs):
        # Initialized numeric/latent hashes remain original, not cached.
        if type(data) is not bytes or data or kwargs:
            return hashlib.sha256(data, **kwargs)
        return _DeferredSHA256(self._cache)


@contextmanager
def qualified_fingerprint_cache_scope():
    from . import frame_atomic_state_ema as ema
    from . import versioned_reliability_memory as reliability
    if (ema.hashlib is not hashlib or ema.committed_reliability_inputs is not original_inputs
            or reliability.committed_reliability_inputs is not original_inputs):
        raise ValueError('Exact original own-process bindings, no nested/unknown overlay')
    raw = FrozenRawStatisticsCache(max_bytes=33554432)
    fingerprints = ExactSegmentSHA256Cache(max_bytes=33554432)
    def inputs(snapshot, selected_token, *, device='cpu'):
        if device != 'cpu': return original_inputs(snapshot, selected_token, device=device)
        snapshot = tuple(snapshot)
        if (len(snapshot) > 4096 or type(selected_token) is not str
                or any(type(row) is not CommittedRawSupport for row in snapshot)
                or len({row.token for row in snapshot}) != len(snapshot)):
            raise ValueError('Same complete immutable snapshot/binding contract')
        groups = next((row.payloads for row in snapshot if row.token == selected_token), None)
        if groups is None: raise ValueError('Selected whole support absent')
        return raw.inputs(groups)
    ema.hashlib = _SHA256Proxy(fingerprints)  # module-local alias, NOT global hashlib
    ema.committed_reliability_inputs = reliability.committed_reliability_inputs = inputs
    try:
        yield raw, fingerprints
    finally:
        ema.hashlib = hashlib
        ema.committed_reliability_inputs = reliability.committed_reliability_inputs = original_inputs
