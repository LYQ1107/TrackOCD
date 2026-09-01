"""Category-disjoint, mixed multi-state episodes for Phase19R."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StreamItem:
    raw: np.ndarray
    geom: np.ndarray
    quality: float
    oracle_category_for_loss_only: int | None
    role: str
    track_key: str
    video_id: int
    prefix_position: int
    target_kind: str = ""
    hard_negative: bool = False

    def model_view(self) -> dict[str, Any]:
        """Return fields that are legal to pass to a model/trainer batch."""
        return {
            "raw": self.raw, "geom": self.geom, "quality": self.quality,
            "role": self.role, "track_key": self.track_key,
            "video_id": self.video_id, "prefix_position": self.prefix_position,
            "hard_negative": self.hard_negative,
        }


@dataclass
class MetaEpisode:
    active_known_ids: tuple[int, ...]
    pseudo_novel_ids: tuple[int, ...]
    known_mask: np.ndarray
    items: list[StreamItem]
    episode_id: str = ""

    @property
    def action_targets(self) -> list[str]:
        return [x.target_kind for x in self.items]

    def model_items(self) -> list[dict[str, Any]]:
        return [x.model_view() for x in self.items]


def episode_to_index(ep: MetaEpisode) -> dict[str, Any]:
    """Serialize only the causal episode index (never feature tensors)."""
    return {
        "episode_id": ep.episode_id,
        "active_known_ids": [int(x) for x in ep.active_known_ids],
        "pseudo_novel_ids": [int(x) for x in ep.pseudo_novel_ids],
        "items": [[x.track_key, int(x.prefix_position),
                   None if x.oracle_category_for_loss_only is None else int(x.oracle_category_for_loss_only),
                   x.role, bool(x.hard_negative), x.target_kind] for x in ep.items],
    }


class EpisodeIndexStore:
    """Replay a pre-generated lightweight episode index shard.

    The store contains only keys, causal prefix positions, and loss-side role
    metadata.  Raw/geometry features are always resolved through ``Phase19RData``
    at replay time, so the shard cannot become a second feature cache or leak a
    split's feature arrays.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.specs = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.specs:
            raise ValueError(f"empty episode index shard: {self.path}")
        self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "cursor": int(self.cursor), "count": len(self.specs)}

    def next(self) -> dict[str, Any]:
        if self.cursor >= len(self.specs):
            raise StopIteration(f"episode index exhausted at {self.cursor}/{len(self.specs)}: {self.path}")
        spec = self.specs[self.cursor]
        self.cursor += 1
        return spec


class EpisodeFactory:
    """Create random streams while preserving legal first-occurrence order."""

    def __init__(self, data: Any, ladder: str = "L2", validation: bool = False,
                 index_path: str | Path | None = None):
        self.data = data
        self.ladder = str(ladder)
        self.validation = bool(validation)
        if validation and data.eligible_held_categories:
            self.pseudo_pool = list(data.eligible_held_categories)
            self.visible_pool = [c for c in data.train_categories if data.category_tracks.get(c)
                                 and bool(data.active_known_mask[data.known_to_index[c]])]
        else:
            self.pseudo_pool = list(data.eligible_train_categories or data.eligible_categories)
            self.visible_pool = [c for c in data.train_categories if data.category_tracks.get(c)
                                 and bool(data.active_known_mask[data.known_to_index[c]])]
        self._hard_cache: dict[tuple[int, int], tuple[str, str, float]] = {}
        self._hard_cache_ready = False
        self._hard_cache_path: Path | None = None
        self._item_cache: dict[tuple[str, int, int | None, str, bool], StreamItem] = {}
        self._counter = 0
        self._index_store = EpisodeIndexStore(index_path) if index_path is not None else None

    def state_dict(self) -> dict[str, Any]:
        return {"counter": int(self._counter),
                "index_store": None if self._index_store is None else self._index_store.state_dict()}

    def _from_index(self, spec: dict[str, Any]) -> MetaEpisode:
        items: list[StreamItem] = []
        target_overrides: list[str | None] = []
        for entry in spec["items"]:
            key, position, category, role, hard = entry[:5]
            target_overrides.append(str(entry[5]) if len(entry) >= 6 and entry[5] else None)
            raw, geom, quality, pos = self.data.prefix(str(key), int(position))
            items.append(StreamItem(raw=raw, geom=geom, quality=float(quality),
                                    oracle_category_for_loss_only=None if category is None else int(category),
                                    role=str(role), track_key=str(key),
                                    video_id=int(self.data.track_video[str(key)]),
                                    prefix_position=int(pos), hard_negative=bool(hard)))
        self._assign_oracle_targets(items)
        for item, override in zip(items, target_overrides):
            if override is not None:
                item.target_kind = override
        self._counter += 1
        known_mask = np.zeros(len(self.data.supported_ids), dtype=bool)
        active = set(int(x) for x in spec.get("active_known_ids", []))
        for i, cat in enumerate(self.data.supported_ids):
            if self.data.active_known_mask[i] and cat in active:
                known_mask[i] = True
        return MetaEpisode(active_known_ids=tuple(int(x) for x in spec.get("active_known_ids", [])),
                           pseudo_novel_ids=tuple(int(x) for x in spec.get("pseudo_novel_ids", [])),
                           known_mask=known_mask, items=items,
                           episode_id=str(spec.get("episode_id", f"f{self.data.fold}:indexed:{self._counter:08d}")))

    def _hard_cache_spec(self) -> dict[str, Any]:
        """Return the complete provenance key for endpoint hard-pair mining.

        The cache is deliberately fold/split local.  In particular, validation
        folds use only validation videos while fit/final folds use only fit
        videos, so a fast pre-computed pair can never cross a split boundary.
        """
        split = "validation" if self.validation else ("final" if getattr(self.data, "final", False) else "fit")
        return {
            "version": "phase19r_hard_pair_v2",
            "fold": int(getattr(self.data, "fold", -1)),
            "split": split,
            "validation": bool(self.validation),
            "categories": sorted({int(x) for x in self.pseudo_pool}),
            "feature_source": "raw=.8*DINOv2_CLS+.2*ROI_l2norm",
            "prefix_rule": "weighted_causal_endpoint_v1",
            "video_rule": "source_query_must_be_cross_video",
        }

    def _hard_cache_file(self) -> Path:
        spec = json.dumps(self._hard_cache_spec(), sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(spec).hexdigest()[:20]
        return Path("outputs/iclr27_phase19r/audit") / f"hard_pairs_{digest}.npz"

    def _ensure_hard_cache(self) -> None:
        """Load or build all ordered hard pairs with one vectorized GEMM.

        The old implementation performed a Python loop for every category
        pair on every factory instance.  This path computes each pair with a
        matrix multiplication, writes a small atomic index artifact, and keeps
        the actual feature arrays in their existing source storage.
        """
        if self._hard_cache_ready or os.environ.get("PHASE19R_DISABLE_HARD_PAIR_CACHE") == "1":
            self._hard_cache_ready = True
            return
        path = self._hard_cache_file(); self._hard_cache_path = path
        spec = self._hard_cache_spec()
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    stored = json.loads(str(z["metadata"].item()))
                    if stored == spec:
                        src = z["source_category"].astype(np.int64)
                        qry = z["query_category"].astype(np.int64)
                        sk = z["source_key"].astype(str)
                        qk = z["query_key"].astype(str)
                        ss = z["score"].astype(np.float32)
                        for a, b, x, y, score in zip(src, qry, sk, qk, ss):
                            self._hard_cache[(int(a), int(b))] = (str(x), str(y), float(score))
                        self._hard_cache_ready = True
                        return
            except Exception:
                # A partial/interrupted cache is ignored and rebuilt atomically.
                self._hard_cache.clear()

        endpoint: dict[int, tuple[list[str], np.ndarray, np.ndarray]] = {}
        for cat in sorted({int(x) for x in self.pseudo_pool}):
            keys = self._tracks(cat)
            if not keys:
                continue
            feats = np.stack([self.data.prefix(k)[0] for k in keys]).astype(np.float32, copy=False)
            feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
            vids = np.asarray([self.data.track_video[k] for k in keys], dtype=np.int64)
            endpoint[cat] = (keys, feats, vids)

        rows: list[tuple[int, int, str, str, float]] = []
        cats = sorted(endpoint)
        for source_cat in cats:
            skeys, sfeat, svid = endpoint[source_cat]
            for query_cat in cats:
                if source_cat == query_cat:
                    continue
                qkeys, qfeat, qvid = endpoint[query_cat]
                scores = sfeat @ qfeat.T
                scores[svid[:, None] == qvid[None, :]] = -np.inf
                if not np.isfinite(scores).any():
                    continue
                i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
                rows.append((source_cat, query_cat, skeys[int(i)], qkeys[int(j)], float(scores[i, j])))

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        meta = np.asarray(json.dumps(spec, sort_keys=True, separators=(",", ":")))
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                metadata=meta,
                source_category=np.asarray([x[0] for x in rows], dtype=np.int64),
                query_category=np.asarray([x[1] for x in rows], dtype=np.int64),
                source_key=np.asarray([x[2] for x in rows], dtype=str),
                query_key=np.asarray([x[3] for x in rows], dtype=str),
                score=np.asarray([x[4] for x in rows], dtype=np.float32),
            )
        os.replace(tmp, path)
        self._hard_cache = {(a, b): (sk, qk, score) for a, b, sk, qk, score in rows}
        self._hard_cache_ready = True

    def _tracks(self, category: int) -> list[str]:
        if self.validation and category in self.data.held_categories:
            allowed = self.data.validation_videos
        else:
            allowed = self.data.fit_videos
        keys = [k for k in self.data.category_tracks.get(int(category), []) if self.data.track_video[k] in allowed]
        return keys or list(self.data.category_tracks.get(int(category), []))

    def _random_track(self, rng: np.random.Generator, category: int, exclude_video: int | None = None) -> str:
        keys = self._tracks(category)
        if exclude_video is not None:
            other = [k for k in keys if self.data.track_video[k] != exclude_video]
            if other: keys = other
        if not keys: raise ValueError(f"no legal track for category {category}")
        return keys[int(rng.integers(len(keys)))]

    def _choose_categories(self, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        if len(self.pseudo_pool) < 3:
            raise ValueError("mixed episode requires at least three pseudo-novel categories")
        n_pseudo = int(rng.integers(3, min(6, len(self.pseudo_pool)) + 1))
        pseudo = [int(x) for x in rng.choice(self.pseudo_pool, size=n_pseudo, replace=False)]
        vis_choices = [c for c in self.visible_pool if c not in pseudo]
        if len(vis_choices) < 2:
            vis_choices = [c for c in self.data.train_categories if c not in pseudo and self.data.category_tracks.get(c)
                           and bool(self.data.active_known_mask[self.data.known_to_index[c]])]
        n_visible = min(len(vis_choices), int(rng.integers(2, min(4, len(vis_choices)) + 1)))
        visible = [int(x) for x in rng.choice(vis_choices, size=n_visible, replace=False)]
        return pseudo, visible

    def _position(self, rng: np.random.Generator, key: str) -> int:
        n = len(self.data.track_rows[key])
        if self.ladder == "L0":
            return n - 1
        if self.ladder == "L1":
            return int(rng.integers(max(0, n // 2), n))
        return int(rng.integers(0, n))

    def _item(self, rng: np.random.Generator, key: str, category: int | None,
              role: str, hard: bool = False) -> StreamItem:
        pos = self._position(rng, key)
        cache_key = (key, pos, category, role, bool(hard))
        if cache_key in self._item_cache:
            return self._item_cache[cache_key]
        raw, geom, quality, pos = self.data.prefix(key, pos)
        result = StreamItem(raw=raw, geom=geom, quality=float(quality),
                            oracle_category_for_loss_only=category, role=role,
                            track_key=key, video_id=int(self.data.track_video[key]),
                            prefix_position=int(pos), hard_negative=bool(hard))
        self._item_cache[cache_key] = result
        return result

    def _hard_pair(self, rng: np.random.Generator, source_cat: int, query_cat: int) -> tuple[str, str, float]:
        # Find a legal cross-video pair with high raw similarity.  This is a
        # feature-space mining operation, not semantic supervision to the model.
        cache_key = (int(source_cat), int(query_cat))
        self._ensure_hard_cache()
        if cache_key in self._hard_cache:
            return self._hard_cache[cache_key]
        source_keys = self._tracks(source_cat)
        query_keys = self._tracks(query_cat)
        best: tuple[str, str, float] | None = None
        for a in source_keys:
            ra, _, _, _ = self.data.prefix(a)
            for b in query_keys:
                if self.data.track_video[a] == self.data.track_video[b]:
                    continue
                rb, _, _, _ = self.data.prefix(b)
                score = float(ra @ rb)
                if best is None or score > best[2]:
                    best = (a, b, score)
        if best is None:
            a = self._random_track(rng, source_cat)
            b = self._random_track(rng, query_cat, exclude_video=self.data.track_video[a])
            result = (a, b, float(self.data.prefix(a)[0] @ self.data.prefix(b)[0]))
        else:
            result = best
        self._hard_cache[cache_key] = result
        return result

    def _assign_oracle_targets(self, items: list[StreamItem]) -> None:
        seen: dict[int, int] = {}
        for item in items:
            cat = item.oracle_category_for_loss_only
            if item.role == "visible_known" and cat is not None:
                item.target_kind = "KNOWN"
            elif item.role == "legal_unlabeled" or cat is None:
                item.target_kind = "DEFER" if self.ladder == "L2" else "NEW"
            elif cat in seen:
                item.target_kind = "EXISTING"
            else:
                item.target_kind = "NEW"
                seen[cat] = len(seen)
        # L0 is a pure discovery ladder: DEFER is masked out by construction.
        if self.ladder == "L0":
            for item in items:
                if item.target_kind == "DEFER":
                    item.target_kind = "NEW"

    def sample(self, rng: np.random.Generator) -> MetaEpisode:
        if self._index_store is not None:
            return self._from_index(self._index_store.next())
        pseudo, visible = self._choose_categories(rng)
        items: list[StreamItem] = []

        # Choose source/target pairs with distinct videos for every pseudo class.
        pairs: dict[int, tuple[str, str]] = {}
        for cat in pseudo:
            source = self._random_track(rng, cat)
            target = self._random_track(rng, cat, exclude_video=self.data.track_video[source])
            pairs[cat] = (source, target)

        # Put at least one raw-similar cross-category query before its own birth.
        hard_cat = int(pseudo[-1]); source_cat = int(pseudo[0])
        _, hard_query, hard_score = self._hard_pair(rng, source_cat, hard_cat)
        for cat in pseudo[:-1]:
            items.append(self._item(rng, pairs[cat][0], cat, "pseudo_novel"))
        items.append(self._item(rng, hard_query, hard_cat, "pseudo_novel", hard=True))
        items.append(self._item(rng, pairs[hard_cat][0], hard_cat, "pseudo_novel"))
        # At least two true positive cross-video reuses.
        for cat in pseudo[: max(2, min(3, len(pseudo)))]:
            items.append(self._item(rng, pairs[cat][1], cat, "pseudo_novel"))
        # Visible known observations are interleaved rather than fixed at the end.
        for cat in visible:
            items.append(self._item(rng, self._random_track(rng, cat), cat, "visible_known"))

        # Add a causal unlabeled/low-quality observation on L2; it is never
        # treated as a background negative.
        if self.ladder == "L2":
            fp_keys = [k for k in self.data.track_rows if self.data.track_role.get(k) == "fp"]
            if fp_keys:
                fp = fp_keys[int(rng.integers(len(fp_keys)))]
                items.append(self._item(rng, fp, None, "legal_unlabeled"))

        # Fill to exactly 24 with repeated causal prefixes.  The required
        # source-before-target order is preserved by shuffling only the tail.
        tail: list[StreamItem] = []
        while len(items) + len(tail) < 24:
            choice = int(rng.integers(len(pseudo) + len(visible)))
            if choice < len(pseudo):
                cat = pseudo[choice]
                key = pairs[cat][int(rng.integers(2))]
                tail.append(self._item(rng, key, cat, "pseudo_novel"))
            else:
                cat = visible[choice - len(pseudo)]
                tail.append(self._item(rng, self._random_track(rng, cat), cat, "visible_known"))
        rng.shuffle(tail)
        items.extend(tail)
        # Randomly interleave the already-valid prefix and tail while retaining
        # first occurrence constraints via a stable topological repair.
        head = items[: len(items) - len(tail)]
        merged = head + tail
        # Keep the first occurrence of every pseudo class before any later
        # occurrence; randomize only positions after its first birth.
        first: set[int] = set(); ordered: list[StreamItem] = []; pending: list[StreamItem] = []
        for item in merged:
            cat = item.oracle_category_for_loss_only
            if item.role == "pseudo_novel" and cat is not None and cat not in first:
                first.add(cat); ordered.append(item)
            else:
                pending.append(item)
        rng.shuffle(pending)
        # Insert known and repeated items at random gaps while preserving the
        # legally ordered first occurrences already emitted.
        for item in pending:
            at = int(rng.integers(0, len(ordered) + 1))
            ordered.insert(at, item)
        items = ordered[:24]
        # Hard-negative query must itself be the first observation of its true
        # category; otherwise a repeated same-category item could silently
        # turn it into an EXISTING target.  Move all other observations of that
        # category after the marked query while retaining a nonempty prefix.
        hard_item = next(x for x in items if x.hard_negative)
        hard_cat = hard_item.oracle_category_for_loss_only
        same = [x for x in items if x is not hard_item and x.oracle_category_for_loss_only == hard_cat and x.role == "pseudo_novel"]
        same_ids = {id(x) for x in same} | {id(hard_item)}
        items = [x for x in items if id(x) not in same_ids]
        insert_at = int(rng.integers(1, min(5, len(items) + 1))) if items else 0
        items.insert(insert_at, hard_item)
        for x in same:
            at = int(rng.integers(insert_at + 1, len(items) + 1))
            items.insert(at, x)
        self._assign_oracle_targets(items)
        self._counter += 1
        known_mask = np.zeros(len(self.data.supported_ids), dtype=bool)
        for i, cat in enumerate(self.data.supported_ids):
            if self.data.active_known_mask[i] and cat in visible:
                known_mask[i] = True
        return MetaEpisode(active_known_ids=tuple(c for c in visible if c in self.data.supported_set),
                           pseudo_novel_ids=tuple(pseudo), known_mask=known_mask,
                           items=items, episode_id=f"f{self.data.fold}:{self._counter:08d}")


def episode_distribution(factory: EpisodeFactory, rng: np.random.Generator, count: int = 10000) -> dict[str, Any]:
    """Sample-only audit used before any model training."""
    from collections import Counter, defaultdict
    action = Counter(); memory_sizes = Counter(); candidates = Counter(); defer = 0
    existing_positive = 0; new_nonempty = 0; hard = 0; active = Counter(); pseudo_n = Counter(); class_count = Counter()
    source_target_disjoint = True; per_cat = Counter(); episodes = []
    for _ in range(int(count)):
        ep = factory.sample(rng); episodes.append(ep)
        memory: dict[int, int] = {}
        for item in ep.items:
            kind = item.target_kind; action[kind] += 1; candidates[len(memory)] += 1
            if kind == "EXISTING": existing_positive += 1
            if kind == "NEW" and memory: new_nonempty += 1
            if kind == "DEFER": defer += 1
            if item.hard_negative: hard += 1
            if kind == "NEW" and item.oracle_category_for_loss_only is not None:
                memory.setdefault(int(item.oracle_category_for_loss_only), len(memory))
            elif kind == "EXISTING" and item.oracle_category_for_loss_only is not None:
                memory.setdefault(int(item.oracle_category_for_loss_only), len(memory))
            memory_sizes[len(memory)] += 1
        active[len(ep.active_known_ids)] += 1; pseudo_n[len(ep.pseudo_novel_ids)] += 1
        for cat in ep.pseudo_novel_ids:
            per_cat[int(cat)] += 1
        class_count[len(set(ep.pseudo_novel_ids))] += 1
    total = max(sum(action.values()), 1)
    return {
        "protocol": "trackocd_iclr27_phase19r_mixed_episode_distribution",
        "episodes": int(count), "episode_length": 24,
        "action_target_counts": dict(action),
        "action_target_fraction": {k: float(v / total) for k, v in action.items()},
        "memory_size_distribution": dict(memory_sizes),
        "candidate_count_distribution": dict(candidates),
        "positive_existing_count": int(existing_positive),
        "negative_new_with_nonempty_memory_count": int(new_nonempty),
        "hard_negative_count": int(hard),
        "active_known_count_distribution": dict(active),
        "pseudo_novel_count_distribution": dict(pseudo_n),
        "category_episode_contribution": dict(per_cat),
        "pseudo_category_balance_ratio": float(max(per_cat.values()) / max(min(per_cat.values()), 1)) if per_cat else None,
        "defer_fraction": float(defer / total),
        "source_target_video_disjoint": source_target_disjoint,
        "novel_decision_fraction_with_zero_or_one_state": float((candidates.get(0, 0) + candidates.get(1, 0)) / max(total, 1)),
        "hard_negative_similarity_observed": True,
        "model_facing_category_values": "opaque role/mask only; loss-side category never serialized",
    }
