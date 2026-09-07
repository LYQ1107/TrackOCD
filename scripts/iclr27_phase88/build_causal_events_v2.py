#!/usr/bin/env python3
"""Build Phase88 source-bank causal events with visual hard negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase88"
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def first_reliable(data, key: str) -> int:
    return int(data._phase88_store.reliability_prefix(key, 16)) if hasattr(data, "_phase88_store") else 1


def prefix_vector(data, key: str) -> np.ndarray:
    indices = data.track_rows[key]
    # Use a causal quality-weighted mean of observed vectors, matching Phase19RData.prefix.
    raw, _, _, _ = data.prefix(key)
    return np.asarray(raw, dtype=np.float32)


def legal_tracks(data, *, fit: bool) -> list[str]:
    fold = data.fold_record
    videos = set(int(v) for v in (fold.get("fit_videos", []) if fit else fold.get("validation_videos", [])))
    held = set(int(v) for v in fold.get("held_categories", []))
    out = []
    for key, cat in data.track_category.items():
        if data.track_role.get(key) != "supported_known":
            continue
        if data.track_video[key] not in videos:
            continue
        if fit and cat in held:
            continue
        out.append(key)
    return sorted(out)


def make_fold(fold: int, split: str, max_variants: int, seed: int) -> tuple[list[dict], dict]:
    from src.iclr27_phase19r.data.stream import Phase19RData
    from src.iclr27_phase88.data import FeatureStore

    data = Phase19RData(fold)
    store = FeatureStore(fold, data)
    data._phase88_store = store
    is_fit = split == "fit"
    targets = legal_tracks(data, fit=is_fit)
    pool = set(targets)
    by_cat: dict[int, list[str]] = defaultdict(list)
    for key in targets:
        by_cat[data.track_category[key]].append(key)
    vectors = {key: prefix_vector(data, key) for key in targets}
    events: list[dict] = []
    stats = Counter()
    rng = random.Random(seed + fold * 1009 + (0 if is_fit else 1))
    for target in targets:
        target_cat = int(data.track_category[target]); target_video = int(data.track_video[target])
        same = [k for k in by_cat[target_cat] if k != target and data.track_video[k] != target_video]
        diff = [k for k in pool if data.track_category[k] != target_cat and data.track_video[k] != target_video]
        if not diff:
            continue
        q = vectors[target]
        diff.sort(key=lambda k: (-float(np.dot(q, vectors[k])), k))
        same.sort(key=lambda k: (-float(np.dot(q, vectors[k])), k))
        hard = diff[: max(2, max_variants)]
        if same:
            for variant in range(min(max_variants, max(1, len(same)))):
                pos_sources = [same[variant % len(same)]]
                if len(same) > 1:
                    pos_sources.append(same[(variant + 1) % len(same)])
                distractors = hard[: max(0, 4 - len(pos_sources))]
                sources = list(dict.fromkeys(pos_sources + distractors))
                events.append({
                    "event_id": f"f{fold}-{split}-pos-{len(events):06d}",
                    "fold": fold, "split": split, "polarity": "positive",
                    "source_tracks": [{"track_key": s, "category_for_loss_only": int(data.track_category[s]), "video_id": int(data.track_video[s])} for s in sources],
                    "source_track_keys": sources,
                    "target_track_key": target,
                    "target_category_for_loss_only": target_cat,
                    "target_video": target_video,
                    "reliable_prefix_for_loss_only": store.reliability_prefix(target),
                    "masked_known_categories_for_loss_only": [target_cat],
                    "metadata_only": ["category_for_loss_only", "video_id", "track_key", "reliable_prefix_for_loss_only"],
                })
                stats["positive"] += 1
        for variant in range(min(max_variants, len(hard))):
            sources = hard[variant:variant + max(2, min(4, len(hard)))]
            if len(sources) < 2:
                sources = hard[:min(4, len(hard))]
            events.append({
                "event_id": f"f{fold}-{split}-neg-{len(events):06d}",
                "fold": fold, "split": split, "polarity": "negative",
                "source_tracks": [{"track_key": s, "category_for_loss_only": int(data.track_category[s]), "video_id": int(data.track_video[s])} for s in sources],
                "source_track_keys": sources,
                "target_track_key": target,
                "target_category_for_loss_only": target_cat,
                "target_video": target_video,
                "reliable_prefix_for_loss_only": store.reliability_prefix(target),
                "masked_known_categories_for_loss_only": [target_cat],
                "metadata_only": ["category_for_loss_only", "video_id", "track_key", "reliable_prefix_for_loss_only"],
            })
            stats["negative"] += 1
    rng.shuffle(events)
    # Natural wrong bindings are rare in the legal source-bank stream.  The
    # registered TRAIN-only 10% augmentation creates a provisional wrong
    # binding at target entry; it never alters validation/held events.
    if is_fit and events:
        reset_n = max(1, int(round(0.10 * len(events))))
        reset_ids = set(rng.sample(range(len(events)), reset_n))
        for i, event in enumerate(events):
            event["reset_injection"] = bool(i in reset_ids)
    else:
        for event in events:
            event["reset_injection"] = False
    max_events = 12000 if is_fit else 2500
    events = events[:max_events]
    stats.update({"targets": len(targets), "events": len(events), "positive_final": sum(e["polarity"] == "positive" for e in events), "negative_final": sum(e["polarity"] == "negative" for e in events)})
    return events, {
        "fold": fold, "split": split, "target_tracks": len(targets),
        "events": len(events), "positive": sum(e["polarity"] == "positive" for e in events),
        "negative": sum(e["polarity"] == "negative" for e in events),
        "mean_source_count": float(np.mean([len(e["source_track_keys"]) for e in events])) if events else 0.0,
        "hard_negative_visual_cosine": {"min": float(min((np.dot(vectors[e["target_track_key"]], vectors[e["source_track_keys"][0]]) for e in events if e["polarity"] == "negative"), default=0.0)), "max": float(max((np.dot(vectors[e["target_track_key"]], vectors[e["source_track_keys"][0]]) for e in events if e["polarity"] == "negative"), default=0.0))},
        "source_before_target_is_episode_order_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=88001)
    args = parser.parse_args()
    counts = {}
    for fold in range(4):
        for split in ("fit", "val"):
            events, info = make_fold(fold, split, max_variants=4, seed=args.seed)
            atomic_jsonl(OUT / "manifests" / f"{split}_events_v2_f{fold}.jsonl", events)
            counts[f"{split}{fold}"] = info
    manifest = {
        "schema_version": "trackocd.phase88.causal_event_v2.v1",
        "phase": 88,
        "protocol": "episode source-before-target; source_video != target_video; no numeric video chronology",
        "counts": counts,
        "source_count_range": [2, 4], "max_variants_per_target": 4,
        "model_input_fields": ["causal_raw_sequence", "causal_geometry_sequence", "quality_sequence", "support_features", "causal_state", "known_mask"],
        "loss_only_fields": ["polarity", "category_for_loss_only", "video_id", "track_key", "reliable_prefix_for_loss_only"],
        "hard_negative": "same split, different category/video, highest cosine of causal track vectors",
        "future_rows_or_tracks": False, "ids_or_text_as_model_input": False,
        "held_event_overlap": False,
    }
    atomic_json(OUT / "manifests" / "causal_event_v2_manifest.json", manifest)
    atomic_json(OUT / "audit" / "causal_event_v2_build.json", manifest)
    atomic_json(OUT / "completion" / "build_events_v2.done", {"status": "DONE", "manifest": str((OUT / "manifests" / "causal_event_v2_manifest.json").resolve())})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
