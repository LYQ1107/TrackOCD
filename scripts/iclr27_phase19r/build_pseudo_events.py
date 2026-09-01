#!/usr/bin/env python
"""Build legal pseudo-held causal events from supported TRAIN categories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.iclr27_phase19r.data.stream import Phase19RData, OUT
from src.iclr27_phase19r.training.event_aligned import reliable_prefix


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def choose_categories(data: Phase19RData, limit: int = 6) -> list[int]:
    eligible = []
    for cat in data.eligible_train_categories:
        keys = [k for k in data.category_tracks.get(cat, []) if data.track_video[k] in data.fit_videos]
        if len({data.track_video[k] for k in keys}) >= 2:
            eligible.append(int(cat))
    if len(eligible) <= limit:
        return eligible
    # Deterministic coverage over the sorted supported TRAIN pool; this keeps
    # a large category from consuming every pseudo event.
    idx = np.rint(np.linspace(0, len(eligible) - 1, limit)).astype(int)
    return [eligible[int(i)] for i in sorted(set(idx))]


def endpoint_hard_source(data: Phase19RData, target_key: str, excluded: set[int],
                         candidate_categories: list[int]) -> tuple[int, str, float]:
    tv = int(data.track_video[target_key]); q = data.prefix(target_key)[0]
    best: tuple[int, str, float] | None = None
    for cat in candidate_categories:
        if int(cat) in excluded:
            continue
        keys = [k for k in data.category_tracks.get(int(cat), [])
                if data.track_video[k] in data.fit_videos and data.track_video[k] != tv]
        for key in keys:
            score = float(q @ data.prefix(key)[0])
            if best is None or score > best[2]:
                best = (int(cat), str(key), score)
    if best is None:
        raise RuntimeError(f"no legal cross-video distractor for {target_key}")
    return best


def build_fold(fold: int, max_categories: int, max_events_per_category: int) -> dict:
    data = Phase19RData(fold)
    cats = choose_categories(data, max_categories)
    positive: list[dict] = []; negative: list[dict] = []
    for cat in cats:
        keys = [k for k in data.category_tracks.get(cat, []) if data.track_video[k] in data.fit_videos]
        by_video: dict[int, list[str]] = {}
        for key in sorted(keys):
            by_video.setdefault(int(data.track_video[key]), []).append(key)
        target_keys = sorted(keys, key=lambda k: (int(data.track_video[k]), k))[:max_events_per_category]
        for n, target_key in enumerate(target_keys):
            target_video = int(data.track_video[target_key])
            source_choices = [k for k in sorted(keys) if data.track_video[k] != target_video]
            if not source_choices:
                continue
            source_key = source_choices[0]
            prefix = reliable_prefix(data, target_key)
            base = {
                "fold": int(fold), "target_category_gt_denominator_only": int(cat),
                "category_gt_denominator_only": int(cat),
                "source_tracklet_keys": [str(source_key)], "source_video": int(data.track_video[source_key]),
                "target_tracklet_key": str(target_key), "target_video": target_video,
                "target_first_reliable_prefix_index_gt_only": int(prefix),
                "target_row_keys": [str(data.rows[i]["row_key"]) for i in data.track_rows[target_key][prefix:prefix + 1]],
                "pseudo_source": "supported_train_category_metadata_only",
            }
            p = dict(base); p.update({"kind": "positive_existing", "expected_first_commit": "EXISTING_NOVEL(source_state)",
                                      "event_key": f"p19r-pseudo-pos:f{fold}:c{cat}:n{n}"})
            positive.append(p)
            dcat, dkey, score = endpoint_hard_source(data, target_key, {int(cat)}, data.eligible_train_categories)
            q = dict(base); q.update({"kind": "negative_new", "distractor_category_gt_denominator_only": int(dcat),
                                      "source_tracklet_keys": [str(dkey)], "source_video": int(data.track_video[dkey]),
                                      "raw_hard_negative_similarity": float(score), "expected_first_commit": "NEW_NOVEL",
                                      "event_key": f"p19r-pseudo-neg:f{fold}:c{cat}:n{n}"})
            p["masked_known_categories"] = [int(cat)]
            q["masked_known_categories"] = sorted({int(cat), int(dcat)})
            negative.append(q)
    rows = sorted(positive + negative, key=lambda x: x["event_key"])
    out = OUT / "manifests" / f"pseudo_train_events_fold{fold}.jsonl"
    atomic_jsonl(out, rows)
    summary = {"fold": int(fold), "categories": cats, "positive_events": len(positive),
               "negative_events": len(negative), "events": len(rows), "source": "supported TRAIN only",
               "held_or_public_categories_used": [], "path": str(out)}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit" / f"pseudo_train_events_fold{fold}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), default=None)
    p.add_argument("--max-categories", type=int, default=6)
    p.add_argument("--max-events-per-category", type=int, default=8)
    a = p.parse_args()
    folds = range(4) if a.fold is None else [a.fold]
    print(json.dumps([build_fold(f, a.max_categories, a.max_events_per_category) for f in folds], indent=2), flush=True)


if __name__ == "__main__":
    main()
