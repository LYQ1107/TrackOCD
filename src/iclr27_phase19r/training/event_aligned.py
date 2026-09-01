"""Legal pseudo-held causal event episodes for Phase19R corrective training.

The builder consumes only supported TRAIN-category metadata and causal prefixes.
Category IDs stay in ``oracle_category_for_loss_only`` exactly as in the mixed
episode trainer; the model sees only the role and episode-local known mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase19r.data.episodes import MetaEpisode, StreamItem


def reliable_prefix(data: Any, key: str) -> int:
    rows = data.track_rows[key]
    return max(0, next((j for j, i in enumerate(rows)
                        if data.rows[i].get("assigned") == "1"
                        and float(data.rows[i].get("row_iou", 0.0)) >= .5), len(rows) - 1))


def _positions(data: Any, key: str, count: int, rng: np.random.Generator,
               first_reliable: int | None = None) -> list[int]:
    n = len(data.track_rows[key])
    if n <= 1:
        return [0] * int(count)
    # Keep every generated prefix causal and include early, reliable, and late
    # states.  The tiny random jitter changes only the legal index shard, never
    # the category visibility or target semantics.
    vals = np.linspace(0, n - 1, int(count), dtype=np.float64)
    if first_reliable is not None and count >= 4:
        vals[1] = min(n - 1, max(0, int(first_reliable)))
    out = np.rint(vals).astype(np.int64).tolist()
    if count > 6:
        j = int(rng.integers(1, count - 1))
        out[j] = int(rng.integers(0, n))
    return [max(0, min(n - 1, int(x))) for x in out]


def event_to_episode(data: Any, event: dict[str, Any], *, rng: np.random.Generator,
                     episode_id: str, source_steps: int = 8,
                     target_steps: int = 16) -> MetaEpisode:
    """Convert one pseudo event to a 24-item causal training episode."""
    kind = str(event["kind"])
    target_cat = int(event["target_category_gt_denominator_only"])
    source_cat = int(event.get("distractor_category_gt_denominator_only", target_cat))
    source_key = str(event["source_tracklet_keys"][0])
    target_key = str(event["target_tracklet_key"])
    target_reliable = int(event.get("target_first_reliable_prefix_index_gt_only", reliable_prefix(data, target_key)))
    items: list[StreamItem] = []
    for pos in _positions(data, source_key, source_steps, rng):
        raw, geom, quality, actual = data.prefix(source_key, pos)
        items.append(StreamItem(raw=raw, geom=geom, quality=float(quality),
                                oracle_category_for_loss_only=source_cat,
                                role="pseudo_novel", track_key=source_key,
                                video_id=int(data.track_video[source_key]),
                                prefix_position=int(actual)))
    target_positions = _positions(data, target_key, target_steps, rng,
                                  first_reliable=target_reliable)
    for j, pos in enumerate(target_positions):
        raw, geom, quality, actual = data.prefix(target_key, pos)
        item = StreamItem(raw=raw, geom=geom, quality=float(quality),
                          oracle_category_for_loss_only=target_cat,
                          role="pseudo_novel", track_key=target_key,
                          video_id=int(data.track_video[target_key]),
                          prefix_position=int(actual),
                          hard_negative=bool(kind == "negative_new" and j == 0))
        items.append(item)

    # Registered action semantics: source creates a state; positive target
    # reuses it, negative target is a fresh category.  Prefixes before the
    # first reliable point train the same DEFER timing as the evaluator.
    seen: set[int] = set()
    for item in items:
        cat = item.oracle_category_for_loss_only
        if cat not in seen:
            item.target_kind = "NEW"
            seen.add(int(cat))
        else:
            item.target_kind = "EXISTING"
        if item.track_key == target_key and item.prefix_position < target_reliable:
            item.target_kind = "DEFER"
    if kind == "positive_existing":
        # The target must be an EXISTING event after the reliable prefix even
        # when its sampled position happens to repeat an early endpoint.
        for item in items[source_steps:]:
            if item.prefix_position >= target_reliable:
                item.target_kind = "EXISTING"

    masked = {target_cat}
    if kind == "negative_new":
        masked.add(source_cat)
    known_mask = np.zeros(len(data.supported_ids), dtype=bool)
    for i, cat in enumerate(data.supported_ids):
        if bool(data.active_known_mask[i]) and int(cat) not in masked:
            known_mask[i] = True
    active = tuple(int(cat) for cat in data.supported_ids if cat in data.known_to_index
                   and bool(known_mask[data.known_to_index[int(cat)]]))
    pseudo = (source_cat, target_cat) if source_cat != target_cat else (target_cat,)
    return MetaEpisode(active_known_ids=active, pseudo_novel_ids=tuple(pseudo),
                       known_mask=known_mask, items=items, episode_id=episode_id)


def event_index_spec(data: Any, event: dict[str, Any], *, rng: np.random.Generator,
                     episode_id: str) -> dict[str, Any]:
    """Serialize an event-aligned episode without feature arrays."""
    from src.iclr27_phase19r.data.episodes import episode_to_index
    return episode_to_index(event_to_episode(data, event, rng=rng, episode_id=episode_id))


def load_event_manifest(path: str | Path) -> list[dict[str, Any]]:
    import json
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

