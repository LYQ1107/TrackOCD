"""Train Known-only metadata split; no descriptors/score driven selection.

Prototype videos are globally reserved first. Distinct pseudo-category roles
then reserve cross-video support, and remaining categories fit the adapter.
Unusable tiny categories remain prototype-supported, never falsely trainable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import random
import math


PARTITIONS = ("prototype", "representation_fit", "development", "policy_train", "heldout_selection")


def known_tracks(annotation, known_ids, maximum_observations=16):
    images = {int(i["id"]): i for i in annotation["images"]}
    grouped = {}
    for ann in annotation["annotations"]:
        category = int(ann["category_id"])
        if category not in known_ids:
            continue
        key = int(ann["video_id"]), int(ann["track_id"])
        row = grouped.setdefault(key, {"video_id": key[0], "physical_track_id": key[1],
                                      "category_id": category, "observations": []})
        if row["category_id"] != category:
            raise ValueError("Inconsistent Train Known track category")
        im = images[int(ann["image_id"])]
        x, y, w, h = map(float, ann["bbox"])
        if int(im["video_id"]) != key[0] or not all(math.isfinite(n) for n in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError("Invalid Train Known observation")
        row["observations"].append({"image_id": int(im["id"]), "frame_id": int(im["frame_index"]),
                                    "image_path": im["file_name"], "bbox_xyxy": [x, y, x+w, y+h]})
    rows = []
    for key, row in sorted(grouped.items()):
        obs = sorted(row["observations"], key=lambda o: (o["frame_id"], o["image_id"]))
        if len({o["frame_id"] for o in obs}) != len(obs):
            raise ValueError("Duplicate physical observation frame")
        rows.append({**row, "key": f"train_gt_{key[0]}_{key[1]}", "total_observations": len(obs),
                     "observations": obs[:maximum_observations]})
    return rows


def split_tracks(rows, known_ids, config):
    if not rows or any(r["category_id"] not in known_ids for r in rows):
        raise ValueError("Only nonempty inherited Train Known rows allowed")
    by_category = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_category[row["category_id"]][row["video_id"]].append(row)
    rng = random.Random(config["split_seed"])
    video_order = sorted({r["video_id"] for r in rows}); rng.shuffle(video_order)
    rank = {v: i for i, v in enumerate(video_order)}
    owner = {}
    # Rare classes first, reuse already-reserved prototype videos where possible.
    for c in sorted(by_category, key=lambda c: (len(by_category[c]), c)):
        vs = by_category[c]
        if not any(owner.get(v) == "prototype" for v in vs):
            available = sorted(vs, key=rank.get)
            owner[available[0]] = "prototype"
    counts = {c: sum(len(rs) for rs in vs.values()) for c, vs in by_category.items()}
    protected = set(sorted(counts, key=lambda c: (-counts[c], c))[:2])
    categories = {"representation_fit": [], "development": [], "policy_train": [], "heldout_selection": []}
    assigned = set()
    minimum = config["cross_video_minimum_videos"]
    for part in ("heldout_selection", "policy_train", "development"):
        candidates = [c for c in by_category if c not in assigned | protected]
        rng.shuffle(candidates)
        # Sparse candidates get first access; category ID tie order was randomized.
        candidates.sort(key=lambda c: len(by_category[c]))
        for c in candidates:
            available = [v for v in by_category[c] if owner.get(v, part) == part]
            if len(available) < minimum:
                continue
            selected = sorted(available, key=lambda v: (owner.get(v) != part, rank[v]))[:minimum]
            owner.update({v: part for v in selected})
            categories[part].append(c); assigned.add(c)
            if len(categories[part]) == config["pseudo_categories_per_role"][part]:
                break
        if len(categories[part]) != config["pseudo_categories_per_role"][part]:
            raise ValueError("Insufficient category/video-disjoint pseudo role support: " + part)
    for c in sorted(by_category, key=lambda c: (len(by_category[c]), c)):
        if c in assigned:
            continue
        available = [v for v in by_category[c] if owner.get(v, "representation_fit") == "representation_fit"]
        if len(available) >= minimum:
            selected = sorted(available, key=lambda v: (v not in owner, rank[v]))[:minimum]
            owner.update({v: "representation_fit" for v in selected})
            categories["representation_fit"].append(c)
    if len(categories["representation_fit"]) < 2:
        raise ValueError("Insufficient fitting categories for valid contrastive negatives")
    # Reserve Known probes in each disjoint stream, without taking a fit video.
    for part in ("development", "policy_train", "heldout_selection"):
        for c in sorted(categories["representation_fit"], key=lambda c: (len(by_category[c]), c)):
            if any(owner.get(v) == part for v in by_category[c]):
                continue
            available = sorted((v for v in by_category[c] if v not in owner), key=rank.get)
            if available:
                owner[available[0]] = part
    choices = [part for part, weight in config["remainder_video_weights"].items() for _ in range(weight)]
    for v in sorted(rank, key=rank.get):
        if v not in owner:
            owner[v] = rng.choice(choices)
    fit = set(categories["representation_fit"])
    selected = []
    for row in rows:
        part = owner[row["video_id"]]; c = row["category_id"]
        allowed = part == "prototype" or (part == "representation_fit" and c in fit) or (
            part in categories and part != "representation_fit" and c in fit | set(categories[part]))
        if allowed:
            selected.append({**row, "partition": part, "simulation_role": "known" if part in {"prototype", "representation_fit"} or c in fit else "pseudo_novel"})
    prototype_ids = sorted({r["category_id"] for r in selected if r["partition"] == "prototype"})
    assert set(prototype_ids) == set(by_category)
    partition_summary = {}
    for part in PARTITIONS:
        subset = [r for r in selected if r["partition"] == part]
        partition_summary[part] = {"tracks": len(subset), "observations": sum(len(r["observations"]) for r in subset),
                                  "videos": sorted({r["video_id"] for r in subset}),
                                  "known_tracks": sum(r["simulation_role"] == "known" for r in subset),
                                  "pseudo_novel_tracks": sum(r["simulation_role"] == "pseudo_novel" for r in subset)}
        if not subset:
            raise ValueError("Empty registered partition: " + part)
    vids = [set(partition_summary[p]["videos"]) for p in PARTITIONS]
    assert all(not a & b for i, a in enumerate(vids) for b in vids[i+1:])
    class_sets = [set(categories[p]) for p in categories]
    assert all(not a & b for i, a in enumerate(class_sets) for b in class_sets[i+1:])
    for c in fit:
        assert len({r["video_id"] for r in selected if r["partition"] == "representation_fit" and r["category_id"] == c}) >= minimum
    return selected, {"categories": categories, "partitions": partition_summary,
                      "selected_tracks": len(selected), "selected_observations": sum(len(r["observations"]) for r in selected),
                      "prototype_supported_known_ids": prototype_ids, "prototype_missing_known_ids": sorted(set(known_ids)-set(prototype_ids)),
                      "category_disjointness": True, "global_video_disjointness": True,
                      "heldout_pseudo_categories_not_used_for_adapter_or_policy_training": True,
                      "known_probe_categories_intentionally_shared": True,
                      "feature_or_metric_based_sampling": False,
                      "unselected_tracks": len(rows)-len(selected), "protected_fitting_categories": sorted(protected)}


def resource_statistics(rows, known_ids):
    by = defaultdict(list)
    for r in rows:
        by[r["category_id"]].append(r)
    choose2 = lambda n: n*(n-1)//2
    stats = {}
    for c, rs in sorted(by.items()):
        counts = Counter(r["video_id"] for r in rs)
        stats[str(c)] = {"tracks": len(rs), "videos": len(counts),
                         "observations": sum(r["total_observations"] for r in rs),
                         "observation_count_histogram": dict(sorted(Counter(r["total_observations"] for r in rs).items())),
                         "tracks_at_prefix": {str(p): sum(r["total_observations"] >= p for r in rs) for p in (1,2,4,8,16)},
                         "cross_video_positive_track_pairs": choose2(len(rs))-sum(choose2(n) for n in counts.values())}
    return {"known_categories_supported": len(by), "known_tracks": len(rows),
            "known_observations": sum(r["total_observations"] for r in rows), "per_category": stats,
            "cross_video_positive_pairs": sum(s["cross_video_positive_track_pairs"] for s in stats.values()),
            "different_category_negative_pairs": choose2(len(rows))-sum(choose2(len(rs)) for rs in by.values()),
            "missing_train_known_ids": sorted(set(known_ids)-set(by)),
            "short_tracks_preserved": True, "no_minimum_p16_filter": True}
