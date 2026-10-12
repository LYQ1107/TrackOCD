"""Real prefix-bank Raw snapshot -> differentiable anonymous support encoding.

External selection/bindings are not model inputs. This is not retrieval, loss
target construction, full controller integration, or a training result. The
snapshot must be taken before current-frame commits; this function cannot
certify when its caller took it. No learned embedding is cached in the bank.
"""
import struct
import numpy as np
import torch
from .prefix_member_bank import CommittedRawSupport
from .differentiable_support_bank import VisiblePrefix, reencode_supports


def _decode(payload, device):
    if type(payload) is not bytes or len(payload) < 8:
        raise ValueError('Immutable bank Raw payload required')
    count = struct.unpack('<Q', payload[:8])[0]
    if count < 1 or count > 16384 or len(payload) != 8 + count * (768 * 4 + 4 * 4 + 4 + 8):
        raise ValueError('Exact bounded Raw prefix framing required')
    offset, arrays = 8, []
    for width, dtype in ((768, '<f4'), (4, '<f4'), (1, '<f4'), (1, '<i8')):
        array = np.frombuffer(payload, dtype=dtype, count=count * width, offset=offset).copy()
        offset += count * width * np.dtype(dtype).itemsize
        arrays.append(torch.from_numpy(array.reshape(count, width) if width > 1 else array).to(device))
    return VisiblePrefix(*arrays)


def reencode_committed_supports(model, snapshot, selected_tokens, *, device='cpu'):
    """One numeric row per selected nonempty anonymous token; binding external.

    Selected tokens are output correspondence handles, not labels/features.
    Equal-payload groups in different video/frame scopes remain separate votes
    exactly as the bank specifies. Bounds refuse the whole call, never truncate.
    Returns Raw/adapted prototypes plus the existing five-stat numeric contract.
    """
    snapshot, selected = tuple(snapshot), tuple(selected_tokens)
    if (len(snapshot) > 4096 or len(selected) > 4096
            or any(type(row) is not CommittedRawSupport for row in snapshot)
            or any(type(token) is not str for token in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('Bounded immutable snapshot and unique external bindings required')
    lookup = {row.token: row.payloads for row in snapshot}
    if len(lookup) != len(snapshot):
        raise ValueError('Duplicate snapshot bindings')
    if any(token not in lookup or not lookup[token] for token in selected):
        raise ValueError('Selected token absent or empty tombstone')
    payload_groups = [lookup[token] for token in selected]
    if sum(len(group) for group in payload_groups) > 4096:
        raise ValueError('Support group bound; no pruning')
    # Validate lengths/count budget before allocating any decoded tensors.
    observations = 0
    for group in payload_groups:
        for payload in group:
            if type(payload) is not bytes or len(payload) < 8:
                raise ValueError('Immutable Raw payload required')
            count = struct.unpack('<Q', payload[:8])[0]
            if count < 1 or len(payload) != 8 + count * 3100:
                raise ValueError('Exact Raw prefix framing required')
            observations += count
    if observations > 16384:
        raise ValueError('Visible observation bound; no truncation')
    groups = [[_decode(payload, device) for payload in group] for group in payload_groups]
    return reencode_supports(model, groups, torch.zeros(len(selected), device=device),
                             torch.ones(len(selected), device=device))
