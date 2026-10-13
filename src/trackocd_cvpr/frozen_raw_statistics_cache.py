"""Bounded exact frozen-Raw statistics cache prerequisite, never learned state.

No controller integration/asset reads at import. LRU eviction affects ONLY
derived statistics and recomputes exact values; not support/history eviction.
All callers must still preserve separate alias/video/frame support scopes.
"""
from collections import OrderedDict
from dataclasses import dataclass
import struct
import torch
from .committed_support_bridge import _decode


@dataclass(frozen=True)
class FrozenRawStatistics:
    vector_float32_bytes: bytes
    quality_float32_bytes: bytes
    observed_count: int


class FrozenRawStatisticsCache:
    def __init__(self, *, max_bytes=67108864):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive bounded derived-statistics cache')
        self._limit = max_bytes; self._rows = OrderedDict(); self._bytes = 0

    @property
    def retained_bytes(self): return self._bytes

    def get(self, payload):
        if type(payload) is not bytes or len(payload) < 8:
            raise ValueError('Immutable Raw-only framing')
        count = struct.unpack('<Q', payload[:8])[0]
        if not 1 <= count <= 16384 or len(payload) != 8 + count * 3100:
            raise ValueError('Exact original bounded Raw framing')
        if payload in self._rows:
            self._rows.move_to_end(payload)
            return self._rows[payload]
        prefix = _decode(payload, 'cpu')
        if (not torch.isfinite(prefix.visual).all() or not torch.isfinite(prefix.quality).all()
                or torch.any((prefix.quality < 0) | (prefix.quality > 1))):
            raise ValueError('Same original finite Raw/quality contract')
        weights = prefix.quality
        if weights.sum() <= 1e-12: weights = torch.ones_like(weights)
        # EXACT original CPU Torch operation/order, no numpy substitute,
        # per-group normalization, rounding, label or learned-embedding cache.
        vector = (prefix.visual * weights[:, None]).sum(dim=0) / weights.sum()
        value = FrozenRawStatistics(vector.numpy().astype('<f4').tobytes(),
                                    prefix.quality.mean().numpy().astype('<f4').tobytes(), count)
        size = len(payload) + len(value.vector_float32_bytes) + len(value.quality_float32_bytes) + 8
        if size > self._limit: return value  # no eviction for an uncacheable request
        while self._rows and self._bytes + size > self._limit:
            key, old = self._rows.popitem(last=False)
            self._bytes -= len(key) + len(old.vector_float32_bytes) + len(old.quality_float32_bytes) + 8
        if size <= self._limit:
            self._rows[payload] = value; self._bytes += size
        return value

    def inputs(self, payloads):
        import numpy as np
        payloads = tuple(payloads)
        if not 1 <= len(payloads) <= 512:
            raise ValueError('Same original whole member-group bound, no truncation')
        counts = []
        for payload in payloads:
            if type(payload) is not bytes or len(payload) < 8: raise ValueError('Immutable Raw support')
            count = struct.unpack('<Q', payload[:8])[0]
            if not 1 <= count <= 16384 or len(payload) != 8 + count * 3100: raise ValueError('Exact Raw framing')
            counts.append(count)
        if sum(counts) > 16384: raise ValueError('Same whole-observation bound, no pruning')
        values = [self.get(payload) for payload in payloads]
        # Fresh tensors: downstream normalization or autograd cannot corrupt
        # cached immutable values or another scope sharing the same Raw bytes.
        vectors = torch.stack([torch.from_numpy(np.frombuffer(v.vector_float32_bytes, dtype='<f4').copy()) for v in values])
        quality = torch.tensor([struct.unpack('<f', v.quality_float32_bytes)[0] for v in values], dtype=torch.float32)
        return vectors, quality, torch.tensor(counts, dtype=torch.float32)
