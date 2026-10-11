"""Current-only frame batches; routing IDs never order or enter model inputs.

This is a new scheduling interface, not a repaired historical controller or
dense-video benchmark. Equal complete visible inputs form one atomic group;
a downstream controller must broadcast one prediction to its routing members
and apply one group write, never sequentially write indistinguishable members.
"""
from dataclasses import dataclass
import hashlib
import struct
import numpy as np
from src.trackocd_core.features import PrefixView, prefix_view


@dataclass(frozen=True, slots=True)
class CurrentObservation:
    routing_id: str
    visual: np.ndarray
    box_normalized_xyxy: np.ndarray
    quality: float


@dataclass(frozen=True, slots=True)
class FrameBatch:
    frame_index: int
    model_views: tuple[PrefixView, ...]
    routing_groups: tuple[tuple[str, ...], ...]
    visible_input_sha256: tuple[str, ...]


def visible_bytes(view):
    """Exact current input only, with explicit dimension and dtype framing."""
    pieces = [struct.pack('<Q', len(view.quality))]
    for array, dtype in ((view.visual, '<f4'), (view.boxes_normalized_xyxy, '<f4'),
                         (view.quality, '<f4'), (view.elapsed_frames, '<i8')):
        pieces.append(np.asarray(array, dtype=dtype).tobytes(order='C'))
    return b''.join(pieces)


class CurrentObservationRouter:
    """One video at a time. No labels, future length, ID arithmetic or lookahead.

    Frames, including empty frames, must arrive in strictly increasing order.
    Bounds refer only to already observed input; exceeding one refuses the frame
    instead of dropping a track or silently changing the protocol.
    """
    def __init__(self, *, max_tracks=4096, max_observations=65536):
        if type(max_tracks) is not int or type(max_observations) is not int or min(max_tracks, max_observations) < 1:
            raise ValueError('Positive observed-input bounds required')
        self.max_tracks, self.max_observations = max_tracks, max_observations
        self._history, self._last_frame, self._observations = {}, None, 0

    def ingest(self, frame_index, observations):
        if (type(frame_index) is not int or frame_index < 0
                or self._last_frame is not None and frame_index <= self._last_frame):
            raise ValueError('Strict current frame chronology required')
        observations = tuple(observations)
        ids = [o.routing_id for o in observations]
        if any(type(i) is not str or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Opaque unique routing IDs required within a frame')
        if len(set(self._history).union(ids)) > self.max_tracks or self._observations + len(ids) > self.max_observations:
            raise ValueError('Observed input bound; no hidden sampling or eviction')
        # Validate/copy every current input before changing any persistent state.
        pending, groups = {}, {}
        for observation in observations:
            visual = np.asarray(observation.visual, dtype=np.float32)
            box = np.asarray(observation.box_normalized_xyxy, dtype=np.float32)
            quality = np.asarray(observation.quality, dtype=np.float32)
            if visual.shape != (768,) or box.shape != (4,) or quality.shape != ():
                raise ValueError('Current observation shape differs')
            old = self._history.get(observation.routing_id)
            if old is None:
                arrays = (visual[None], box[None], quality.reshape(1), np.array([frame_index], dtype=np.int64))
            else:
                arrays = (np.concatenate((old[0], visual[None])), np.concatenate((old[1], box[None])),
                          np.concatenate((old[2], quality.reshape(1))), np.append(old[3], frame_index))
            view = prefix_view(*arrays, len(arrays[2]))
            pending[observation.routing_id] = (*[a for a in (view.visual, view.boxes_normalized_xyxy, view.quality)],
                                               np.array(arrays[3], dtype=np.int64, copy=True))
            signature = visible_bytes(view)
            if signature not in groups:
                groups[signature] = (view, [])
            groups[signature][1].append(observation.routing_id)
        # Lexicographic complete visible content, not a physical-ID tie breaker.
        ordered = sorted(groups)
        batch = FrameBatch(frame_index, tuple(groups[k][0] for k in ordered),
                           tuple(tuple(groups[k][1]) for k in ordered),
                           tuple(hashlib.sha256(k).hexdigest() for k in ordered))
        self._history.update(pending)
        self._observations += len(ids)
        self._last_frame = frame_index
        return batch
