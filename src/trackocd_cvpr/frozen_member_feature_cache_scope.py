"""Read-only numeric-input memo prototype; actual ALL11 parity still required.

Never cache model logits/prototypes, skip refreshes or change numerical kernels.
Raw32MiB + SHA15.5MiB + member-input16MiB + temporary index0.5MiB share the
prior64MiB logical ceiling. Real allocator/RSS remains separately guarded.
The temporary index borrows current immutable bank bytes only during refresh;
it is cleared afterwards and never copies/retains old payloads between frames.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import struct
import numpy as np
import torch
from . import member_reliability as member
from .prefix_member_bank import CommittedRawSupport
from .reliability_support_bridge import committed_reliability_inputs as original_inputs
from .frozen_raw_statistics_cache import FrozenRawStatisticsCache
from .qualified_fingerprint_cache_scope import ExactSegmentSHA256Cache, _SHA256Proxy
from .qualified_M_fingerprint_cache_scope import _SupportOnlySHA256Proxy

ORIGINAL_FEATURES = member.member_features


class FrozenMemberFeaturesCache:
    def __init__(self, *, max_bytes=16777216):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive complete-input byte ceiling')
        self.limit, self.rows, self.retained_bytes = max_bytes, OrderedDict(), 0
        self.hits = self.misses = self.fallbacks = 0

    def features(self, raw, quality, counts):
        tensors = (raw, quality, counts)
        eligible = (not torch.is_grad_enabled()
            and all(type(t) is torch.Tensor and t.device.type == 'cpu'
                and t.dtype == torch.float32 and t.is_contiguous() and not t.requires_grad for t in tensors)
            and raw.ndim == 2 and raw.shape[1] == 768 and 1 <= len(raw) <= 512
            and quality.shape == counts.shape == (len(raw),))
        if not eligible:
            self.fallbacks += 1
            return ORIGINAL_FEATURES(raw, quality, counts)
        key = (struct.pack('<Q', len(raw)), *(t.numpy().tobytes() for t in tensors))
        if key in self.rows:
            self.hits += 1
            self.rows.move_to_end(key)
            saved = self.rows[key]
        else:
            self.misses += 1
            features, vectors = ORIGINAL_FEATURES(raw, quality, counts)
            saved = (features.numpy().tobytes(), vectors.numpy().tobytes())
            size = sum(map(len, key)) + sum(map(len, saved)) + 256
            if size <= self.limit:
                while self.rows and self.retained_bytes + size > self.limit:
                    old_key, old_value = self.rows.popitem(last=False)
                    self.retained_bytes -= sum(map(len, old_key)) + sum(map(len, old_value)) + 256
                self.rows[key] = saved
                self.retained_bytes += size
        # Fresh tensors on hits AND misses; mutations cannot contaminate memo.
        n = len(raw)
        return tuple(torch.from_numpy(np.frombuffer(b, dtype=np.float32).copy().reshape(n, width))
                     for b, width in zip(saved, (8, 768)))


class RefreshSnapshotIndex:
    def __init__(self, *, max_bytes=524288):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Bounded temporary lookup bookkeeping')
        self.limit = max_bytes
        self.active = False
        self.snapshot = None
        self.mapping = {}
        self.retained_bytes = self.peak_bytes = self.hits = self.misses = self.fallbacks = 0

    def clear(self):
        self.snapshot = None
        self.mapping = {}
        self.retained_bytes = 0

    @contextmanager
    def transaction(self):
        if self.active:
            raise ValueError('No nested unknown refresh overlay')
        self.clear()
        self.active = True
        try:
            yield
        finally:
            self.clear()
            self.active = False

    def select(self, snapshot, token):
        if not self.active or type(token) is not str or type(snapshot) is not tuple:
            self.fallbacks += 1
            return None
        if self.snapshot is snapshot:
            self.hits += 1
            if token not in self.mapping:
                raise ValueError('Selected whole support absent')
            return snapshot[self.mapping[token]].payloads
        if len(snapshot) > 4096 or any(type(row) is not CommittedRawSupport for row in snapshot):
            raise ValueError('Same complete immutable snapshot contract')
        if len({row.token for row in snapshot}) != len(snapshot):
            raise ValueError('Same unique binding contract')
        if any(type(row.token) is not str or type(row.payloads) is not tuple
                or any(type(p) is not bytes for p in row.payloads) for row in snapshot):
            self.fallbacks += 1
            return None
        size = sum(len(row.token.encode('utf8', errors='surrogatepass')) + 96 + 8 * len(row.payloads) for row in snapshot)
        if size > self.limit:
            self.fallbacks += 1
            return None
        self.misses += 1
        self.snapshot = snapshot
        self.mapping = {row.token: i for i, row in enumerate(snapshot)}
        self.retained_bytes = size
        self.peak_bytes = max(self.peak_bytes, size)
        if token not in self.mapping:
            raise ValueError('Selected whole support absent')
        return snapshot[self.mapping[token]].payloads


@contextmanager
def frozen_member_feature_cache_scope():
    from . import frame_atomic_state_ema as ema, versioned_reliability_memory as reliability
    if (member.member_features is not ORIGINAL_FEATURES or ema.hashlib is not hashlib
            or reliability.hashlib is not hashlib or ema.committed_reliability_inputs is not original_inputs
            or reliability.committed_reliability_inputs is not original_inputs):
        raise ValueError('Original own-process bindings, no nested/unknown overlay')
    raw = FrozenRawStatisticsCache(max_bytes=33554432)
    sha = ExactSegmentSHA256Cache(max_bytes=16252928)
    features = FrozenMemberFeaturesCache()
    index = RefreshSnapshotIndex()
    refresh_M, refresh_EMA = reliability.VersionedReliabilityMemory.refresh, ema.FrameAtomicStateEMA.refresh

    def inputs(snapshot, selected_token, *, device='cpu'):
        if device != 'cpu':
            return original_inputs(snapshot, selected_token, device=device)
        groups = index.select(snapshot, selected_token)
        if groups is None:
            snapshot = tuple(snapshot)
            if (len(snapshot) > 4096 or type(selected_token) is not str
                    or any(type(row) is not CommittedRawSupport for row in snapshot)
                    or len({row.token for row in snapshot}) != len(snapshot)):
                raise ValueError('Same complete immutable snapshot/binding contract')
            groups = next((row.payloads for row in snapshot if row.token == selected_token), None)
            if groups is None:
                raise ValueError('Selected whole support absent')
        return raw.inputs(groups)

    def M_refresh(self, *args, **kwargs):
        with index.transaction():
            return refresh_M(self, *args, **kwargs)

    def EMA_refresh(self, *args, **kwargs):
        with index.transaction():
            return refresh_EMA(self, *args, **kwargs)

    ema.hashlib, reliability.hashlib = _SHA256Proxy(sha), _SupportOnlySHA256Proxy(sha)
    ema.committed_reliability_inputs = reliability.committed_reliability_inputs = inputs
    member.member_features = features.features
    reliability.VersionedReliabilityMemory.refresh, ema.FrameAtomicStateEMA.refresh = M_refresh, EMA_refresh
    try:
        yield raw, sha, features, index
    finally:
        index.clear()
        reliability.VersionedReliabilityMemory.refresh, ema.FrameAtomicStateEMA.refresh = refresh_M, refresh_EMA
        member.member_features = ORIGINAL_FEATURES
        ema.hashlib = reliability.hashlib = hashlib
        ema.committed_reliability_inputs = reliability.committed_reliability_inputs = original_inputs
