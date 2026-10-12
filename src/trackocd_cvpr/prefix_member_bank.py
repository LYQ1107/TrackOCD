"""Replace committed Raw prefixes, not append every cumulative prefix again.

Source/synthetic-only prerequisite for persistent-memory research. No policy,
learned reliability, quarantine decision, GT purity or scientific PASS here.
Opaque route handles are bookkeeping only. Same-frame exact visible aliases
share one support group; later divergence splits groups prospectively. Groups
are NOT certified independent observations. Past commit records never change.
"""
from dataclasses import dataclass
import hashlib
import re
import struct
import numpy as np

from src.trackocd_core.features import prefix_view
from src.trackocd_cvpr.causal_observation_router import visible_bytes


@dataclass(frozen=True, slots=True)
class _Support:
    epoch: int
    frame: int
    token: str
    payload: bytes
    vector_bytes: bytes
    mean_quality: float


@dataclass(frozen=True, slots=True)
class MemberCommit:
    video_epoch: int
    frame_index: int
    routing_id: str
    previous_token: str | None
    token: str
    visible_input_sha256: str


@dataclass(frozen=True, slots=True)
class CommittedRawSupport:
    # Token is an external binding, never a representation/scorer input.
    token: str
    payloads: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class StateEvidence:
    token: str
    support_groups: int
    routed_members: int
    vector_sum_float64_bytes: bytes
    mean_quality: float
    dispersion: float
    status: str  # EMPTY / ZERO_RESULTANT / DEFINED, not GT purity/lifecycle.

    def prototype(self):
        value = np.frombuffer(self.vector_sum_float64_bytes, dtype='<f8').copy()
        norm = np.linalg.norm(value)
        return None if self.status != 'DEFINED' else value / norm


def _extends(old, new):
    """Compare raw visible segments, not GT/route IDs or a hash assumption."""
    old_count, new_count = struct.unpack('<Q', old[:8])[0], struct.unpack('<Q', new[:8])[0]
    if new_count < old_count:
        return False
    old_start = new_start = 8
    for stride in (768 * 4, 4 * 4, 4, 8):
        if old[old_start:old_start + old_count * stride] != new[new_start:new_start + old_count * stride]:
            return False
        old_start += old_count * stride
        new_start += new_count * stride
    return True


class PrefixMemberBank:
    """Atomic category support with latest *committed* prefix per route.

    Token assignments must originate from a separately audited past/current
    policy, never GT. None means WAIT: retain prior committed support. Token
    reassignment moves current support but retains old commit history; strict
    lineage-history purity must still be measured by an external evaluator.
    This class does not allocate/verify semantic births or score candidates.
    """
    def __init__(self, *, max_members=4096, max_tokens=4096, max_commits=65536,
                 max_payload_bytes=67108864, max_videos=2048):
        bounds = (max_members, max_tokens, max_commits, max_payload_bytes, max_videos)
        if any(type(n) is not int or n < 1 for n in bounds):
            raise ValueError('Positive bounded member-bank limits required')
        self._bounds = bounds
        self._members, self._tokens, self._commits = {}, set(), ()
        self._epoch, self._last_frame = 0, None

    def start_next_video(self):
        if self._epoch + 1 >= self._bounds[4]:
            raise ValueError('Video bound; no history or category reset')
        self._epoch += 1
        self._last_frame = None

    def commit_history(self):
        return self._commits  # Immutable records; no GT or writable arrays.

    def raw_support_snapshot(self):
        """Immutable latest committed, exact-alias-deduplicated Raw prefixes.

        Same grouping as state_evidence, including distinct video/frame scopes.
        No routing IDs, GT, old cumulative votes or learned cached embeddings.
        Call before any current-frame assignment to avoid peer self-support.
        Empty token tombstones remain explicit, not silently revived/pruned.
        """
        by_token = {token: {} for token in self._tokens}
        for item in self._members.values():
            by_token[item.token][(item.epoch, item.frame, item.payload)] = item.payload
        return tuple(CommittedRawSupport(token, tuple(groups[key] for key in sorted(groups)))
                     for token, groups in sorted(by_token.items()))

    def state_evidence(self):
        result = {}
        by_token = {}
        for item in self._members.values():
            by_token.setdefault(item.token, []).append(item)
        for token in sorted(self._tokens):  # Output binding order only, no scoring.
            members = by_token.get(token, [])
            groups = {(item.epoch, item.frame, item.payload): item for item in members}
            total = np.zeros(768, dtype='<f8')
            quality = 0.
            for key in sorted(groups):  # Canonical visible/provenance order, not IDs.
                item = groups[key]
                total += np.frombuffer(item.vector_bytes, dtype='<f4')
                quality += item.mean_quality
            count = len(groups)
            norm = float(np.linalg.norm(total))
            status = 'EMPTY' if not count else 'ZERO_RESULTANT' if norm <= 1e-12 else 'DEFINED'
            result[token] = StateEvidence(token, count, len(members), total.tobytes(),
                quality / count if count else 0., float(np.clip(1 - norm / count, 0, 1)) if count else 0., status)
        return result

    def apply_frame(self, batch, tokens):
        """Validate everything, stage replacements/history, then commit once."""
        frame = batch.frame_index
        if (type(frame) is not int or frame < 0 or self._last_frame is not None and frame <= self._last_frame):
            raise ValueError('Strict current-video frame chronology required')
        tokens = tuple(tokens)
        views, groups, signatures = batch.model_views, batch.routing_groups, batch.visible_input_sha256
        if len({len(tokens), len(views), len(groups), len(signatures)}) != 1:
            raise ValueError('Aligned atomic groups and assignments required')
        ids = [identity for group in groups for identity in group]
        if (any(not group for group in groups) or any(type(key) is not str or not key for key in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Unique opaque routing aliases required')
        members, all_tokens, commits = dict(self._members), set(self._tokens), []
        payload_pool = {item.payload: item.payload for item in members.values()}
        last_payload = None
        for view, aliases, signature, token in zip(views, groups, signatures, tokens):
            elapsed = np.asarray(view.elapsed_frames)
            if (elapsed.dtype.kind not in 'iu' or elapsed.ndim != 1 or not len(elapsed)
                    or elapsed[0] != 0 or elapsed[-1] > frame):
                raise ValueError('Exact integer current elapsed chronology required')
            canonical = prefix_view(view.visual, view.boxes_normalized_xyxy, view.quality,
                                    view.elapsed_frames, len(view.quality))
            payload = visible_bytes(canonical)
            if (canonical.elapsed_frames[-1] > frame or hashlib.sha256(payload).hexdigest() != signature
                    or np.any(canonical.quality > 1)):
                raise ValueError('Current visible fingerprint/time/quality differs')
            if last_payload is not None and payload <= last_payload:
                raise ValueError('Canonical unique visible groups required, not ID ordering or duplicate alias writes')
            last_payload = payload
            for identity in aliases:
                old = members.get((self._epoch, identity))
                if old is not None and not _extends(old.payload, payload):
                    raise ValueError('Committed observed prefix cannot shrink or change its past')
            if token is None:
                continue
            if type(token) is not str or re.fullmatch(r'S:(?:0|[1-9][0-9]*)', token) is None:
                raise ValueError('Anonymous policy output token required, not a GT label')
            payload = payload_pool.setdefault(payload, payload)
            vector_bytes = canonical.weighted_mean().astype('<f4').tobytes()
            support = _Support(self._epoch, frame, token, payload, vector_bytes, float(canonical.quality.mean()))
            all_tokens.add(token)
            for identity in aliases:
                key = (self._epoch, identity)
                old = members.get(key)
                members[key] = support
                commits.append(MemberCommit(self._epoch, frame, identity, old.token if old else None, token, signature))
        retained_payloads = {item.payload for item in members.values()}
        if (len(members) > self._bounds[0] or len(all_tokens) > self._bounds[1]
                or len(self._commits) + len(commits) > self._bounds[2]
                or sum(len(value) for value in retained_payloads) > self._bounds[3]):
            raise ValueError('Member/token/history/payload bound; no eviction or purity laundering')
        self._members, self._tokens = members, all_tokens
        self._commits += tuple(commits)
        self._last_frame = frame
        return tuple(commits)
