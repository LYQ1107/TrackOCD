"""Immutable committed-prefix snapshot to numeric M inputs; source-only.

Take the snapshot BEFORE current-frame commits. This adapter cannot attest
that timing, category birth legality, GT independence or external supervision.
Tokens are output correspondence handles only; never reliability features.
"""
import struct
import torch
from .prefix_member_bank import CommittedRawSupport
from .committed_support_bridge import _decode
from .member_reliability import weighted_support


def committed_reliability(model, snapshot, selected_token, *, device='cpu'):
    snapshot = tuple(snapshot)
    if (len(snapshot) > 4096 or type(selected_token) is not str
            or any(type(row) is not CommittedRawSupport for row in snapshot)
            or len({row.token for row in snapshot}) != len(snapshot)):
        raise ValueError('Bounded immutable snapshot/external binding required')
    groups = next((row.payloads for row in snapshot if row.token == selected_token), None)
    if groups is None or not 1 <= len(groups) <= 512:
        raise ValueError('Selected nonempty bounded support required; no pruning')
    counts = []
    for payload in groups:
        if type(payload) is not bytes or len(payload) < 8:
            raise ValueError('Immutable Raw support framing')
        count = struct.unpack('<Q', payload[:8])[0]
        if count < 1 or len(payload) != 8 + count * 3100:
            raise ValueError('Exact Raw prefix framing')
        counts.append(count)
    if sum(counts) > 16384:
        raise ValueError('Whole-support observation bound; no silent truncation')
    vectors, quality = [], []
    for payload in groups:
        prefix = _decode(payload, device)
        if (not torch.isfinite(prefix.visual).all() or not torch.isfinite(prefix.quality).all()
                or torch.any((prefix.quality < 0) | (prefix.quality > 1))):
            raise ValueError('Finite visible Raw and quality only')
        weights = prefix.quality
        if weights.sum() <= 1e-12:
            weights = torch.ones_like(weights)
        vectors.append((prefix.visual * weights[:, None]).sum(dim=0) / weights.sum())
        quality.append(prefix.quality.mean())
    return weighted_support(model, torch.stack(vectors), torch.stack(quality),
                            torch.tensor(counts, dtype=torch.float32, device=device))
