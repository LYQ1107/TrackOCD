"""Deterministic Train Known-only, category/video-disjoint GT pilot selection.

Labels are permitted for sampling/supervision, never part of PrefixView/model
input. Pseudo-Novel is a Train simulation role, not a new TAO category split.
"""
from __future__ import annotations

from collections import defaultdict


def select_pilot(annotation: dict, known_ids: set[int], config: dict) -> tuple[list[dict], dict]:
    images = {int(image["id"]): image for image in annotation["images"]}
    grouped = {}
    for ann in annotation["annotations"]:
        category = int(ann["category_id"])
        if category not in known_ids:
            continue
        key = int(ann["video_id"]), int(ann["track_id"])
        row = grouped.setdefault(key, {"video_id": key[0], "gt_track_id": key[1],
                                      "category": category, "observations": []})
        if row["category"] != category:
            raise ValueError("Inconsistent Train Known physical-track category")
        image = images[int(ann["image_id"])]
        if int(image["video_id"]) != key[0]:
            raise ValueError("Train observation/image video mismatch")
        row["observations"].append({"annotation": ann, "image": image})
    eligible = defaultdict(lambda: defaultdict(list))
    for key, row in sorted(grouped.items()):
        if len(row["observations"]) >= config["minimum_observations"]:
            row["observations"].sort(key=lambda ob: (int(ob["image"]["frame_index"]), int(ob["image"]["id"])))
            frames = [int(ob["image"]["frame_index"]) for ob in row["observations"]]
            if len(set(frames)) != len(frames):
                raise ValueError("Duplicate Train observations at one physical frame")
            eligible[row["category"]][row["video_id"]].append(row)
    n_known = config["known_categories"]
    n_pseudo = config["pseudo_novel_categories_per_partition"]
    n_stream = config["stream_videos_per_category_per_partition"]
    n_rep = config["representation_tracks_per_known_category"] + config["prototype_tracks_per_known_category"]
    required_known_videos = n_rep + 2 * n_stream
    reserved = [c for c in sorted(eligible) if len(eligible[c]) >= required_known_videos][:n_known]
    pool = [c for c in sorted(eligible) if len(eligible[c]) >= n_stream and c not in reserved]
    if len(reserved) != n_known or len(pool) < 2 * n_pseudo:
        raise RuntimeError("INSUFFICIENT_TRAIN_SUPPORT_FOR_REGISTERED_PILOT")
    pseudo = {"policy_train": pool[:n_pseudo], "heldout_selection": pool[n_pseudo:2 * n_pseudo]}
    owner, chosen = {}, {}
    for partition, categories in pseudo.items():
        for category in sorted(categories, key=lambda c: (len(eligible[c]), c)):
            videos = [v for v in sorted(eligible[category]) if owner.get(v, partition) == partition][:n_stream]
            if len(videos) != n_stream:
                raise RuntimeError("INSUFFICIENT_DISJOINT_PSEUDO_NOVEL_VIDEOS")
            owner.update({v: partition for v in videos})
            chosen[partition, category] = videos
    selected_known = []
    for category in sorted(eligible):
        if category in pool[:2 * n_pseudo] or len(eligible[category]) < required_known_videos:
            continue
        local = {}
        for partition, count in (("representation", n_rep), ("policy_train", n_stream),
                                 ("heldout_selection", n_stream)):
            videos = [v for v in sorted(eligible[category])
                      if owner.get(v, partition) == partition and v not in local][:count]
            if len(videos) != count:
                break
            local.update({v: partition for v in videos})
        else:
            selected_known.append(category)
            owner.update(local)
            for partition in config["partitions"]:
                chosen[partition, category] = [v for v, part in local.items() if part == partition]
        if len(selected_known) == n_known:
            break
    if len(selected_known) != n_known:
        raise RuntimeError("INSUFFICIENT_GLOBALLY_DISJOINT_KNOWN_VIDEOS")
    selected = []
    for (partition, category), videos in sorted(chosen.items()):
        for index, video in enumerate(videos):
            row = dict(eligible[category][video][0])
            purpose = "stream"
            if partition == "representation":
                purpose = "prototype" if index < config["prototype_tracks_per_known_category"] else "representation_fit"
            selected.append(dict(row, observations=row["observations"][:config["observations_per_track"]],
                                 partition=partition, purpose=purpose,
                                 simulation_role="known" if category in selected_known else "pseudo_novel"))
    if len(selected) > config["maximum_tracks"] or sum(len(r["observations"]) for r in selected) > config["maximum_observations"]:
        raise ValueError("Registered small pilot cap exceeded")
    sets = {part: {r["video_id"] for r in selected if r["partition"] == part} for part in config["partitions"]}
    if any(sets[a] & sets[b] for i, a in enumerate(sets) for b in list(sets)[i + 1:]):
        raise AssertionError("Pilot videos overlap between partitions")
    if set(selected_known) & (set(pseudo["policy_train"]) | set(pseudo["heldout_selection"])):
        raise AssertionError("Pseudo-Novel categories overlap adaptation categories")
    summary = {
        "selected_known_categories": selected_known,
        "policy_train_pseudo_novel_categories": pseudo["policy_train"],
        "heldout_selection_pseudo_novel_categories": pseudo["heldout_selection"],
        "tracks": len(selected), "observations": sum(len(r["observations"]) for r in selected),
        "partition_tracks": {part: sum(r["partition"] == part for r in selected) for part in sets},
        "partition_videos": {part: sorted(videos) for part, videos in sets.items()},
        "purpose_tracks": {purpose: sum(r["purpose"] == purpose for r in selected)
                           for purpose in ("prototype", "representation_fit", "stream")},
        "video_disjointness_verified": True, "pseudo_novel_category_disjointness_verified": True,
        "all_categories_in_inherited_known": all(r["category"] in known_ids for r in selected),
        "eligible_long_track_categories": len(eligible),
        "eligible_long_track_count": sum(len(rows) for videos in eligible.values() for rows in videos.values()),
        "sampling_uses_features_or_scores": False,
        "heldout_pseudo_categories_unseen_to_adaptation_and_policy_fit": True,
        "known_probe_categories_are_intentionally_shared_not_category_holdout": True,
    }
    return selected, summary
