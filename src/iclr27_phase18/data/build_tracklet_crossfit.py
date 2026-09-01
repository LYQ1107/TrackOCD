"""Build the prediction-independent Phase18 tracklet/cross-fit population.

This module deliberately does not load a learned checkpoint or inspect a
feature value.  Categories, folds, events, and denominators are derived only
from the corrected proposal population and exact GT observability labels.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "data/iclr27_phase18/sources/public_rows_corrected.csv"
DINO2 = ROOT / "data/iclr27_phase18/sources/public_dinov2_cls_roi.npz"
DINO3 = ROOT / "data/iclr27_phase18/sources/full_public_dinov3.npz"
OUT = ROOT / "outputs/iclr27_phase18"
PRIMARY_IOU = 0.5
N_FOLDS = 4
FIT_ROLES = {"known_bank", "novel_correspondence_train"}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_lines(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_rows() -> tuple[list[dict[str, Any]], list[str]]:
    with ROWS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 43423, len(rows)
    keys = [r["row_key"] for r in rows]
    assert len(set(keys)) == len(keys)
    for i, r in enumerate(rows):
        r["source_row_index"] = i
        r["video_i"] = int(r["video_id"])
        r["track_i"] = int(r["track_id"])
        r["category_i"] = int(r["gt_category_id_common"])
        r["event_i"] = int(r["event_rank"])
        r["frame_i"] = int(r["frame_id"])
        r["iou_f"] = float(r["row_iou"])
        r["reliable_exact"] = bool(r["assigned"] == "1" and r["iou_f"] >= PRIMARY_IOU)
    return rows, fieldnames


def feature_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_keys = [r["row_key"] for r in rows]
    z2 = np.load(DINO2, mmap_mode="r")
    z3 = np.load(DINO3, mmap_mode="r")
    k2 = [str(x) for x in z2["row_keys"]]
    k3 = [str(x) for x in z3["row_keys"]]
    assert len(k2) == len(k3) == len(row_keys) == 43423
    assert len(set(k2)) == len(k2) and len(set(k3)) == len(k3)
    assert set(k2) == set(row_keys) == set(k3)
    m2 = {k: i for i, k in enumerate(k2)}
    m3 = {k: i for i, k in enumerate(k3)}
    for r in rows:
        r["dinov2_index"] = m2[r["row_key"]]
        r["dinov3_index"] = m3[r["row_key"]]
    return {
        "rows": len(row_keys),
        "source_csv_sha256": sha_file(ROWS.resolve()),
        "source_row_keys_ordered_sha256": sha_lines(row_keys),
        "source_row_keys_sorted_sha256": sha_lines(sorted(row_keys)),
        "dinov2_path": str(DINO2.resolve()),
        "dinov2_shape_cls": list(z2["cls"].shape),
        "dinov2_shape_roi": list(z2["roi"].shape),
        "dinov2_row_keys_ordered_sha256": sha_lines(k2),
        "dinov2_row_keys_sorted_sha256": sha_lines(sorted(k2)),
        "dinov2_exact_order_match": k2 == row_keys,
        "dinov2_set_match": True,
        "dinov3_path": str(DINO3.resolve()),
        "dinov3_shape": list(z3["features"].shape),
        "dinov3_row_keys_ordered_sha256": sha_lines(k3),
        "dinov3_row_keys_sorted_sha256": sha_lines(sorted(k3)),
        "dinov3_exact_order_match": k3 == row_keys,
        "dinov3_set_match": True,
    }


def build_tracklets(rows: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[int, list[str]]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["video_i"], r["track_i"])].append(r)
    tracklets: dict[tuple[int, int], dict[str, Any]] = {}
    cat_video_tracks: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for physical, rr in sorted(grouped.items()):
        rr.sort(key=lambda r: (r["event_i"], r["source_row_index"]))
        reliable_novel = Counter(
            r["category_i"]
            for r in rr
            if r["gt_role_common"] == "novel" and r["reliable_exact"]
        )
        reliable_any = Counter(
            r["category_i"]
            for r in rr
            if r["gt_role_common"] in {"novel", "supported_known"} and r["reliable_exact"]
        )
        assert len(reliable_novel) <= 1, (physical, reliable_novel)
        label = reliable_any.most_common(1)[0][0] if reliable_any else -1
        label_role = next(
            (r["gt_role_common"] for r in rr if r["reliable_exact"] and r["category_i"] == label),
            "fp",
        )
        first = next(
            (i for i, r in enumerate(rr) if r["reliable_exact"] and r["category_i"] == label),
            -1,
        )
        key = f"v{physical[0]}:p{physical[1]}"
        item = {
            "tracklet_key": key,
            "video_id": physical[0],
            "physical_track_id_index_only": physical[1],
            "row_indices": [r["source_row_index"] for r in rr],
            "row_keys": [r["row_key"] for r in rr],
            "dinov2_indices": [r["dinov2_index"] for r in rr],
            "dinov3_indices": [r["dinov3_index"] for r in rr],
            "event_ranks": [r["event_i"] for r in rr],
            "frames": [r["frame_i"] for r in rr],
            "rows": len(rr),
            "label_category_gt_only": label,
            "label_role_gt_only": label_role,
            "first_reliable_prefix_index_gt_only": first,
            "reliable_row_count_gt_only": int(reliable_any.get(label, 0)),
            "row_keys_sha256": sha_lines(r["row_key"] for r in rr),
        }
        tracklets[physical] = item
        if reliable_novel:
            c = next(iter(reliable_novel))
            cat_video_tracks[c][physical[0]].append(key)
    for videos in cat_video_tracks.values():
        for v in videos:
            videos[v].sort()
    return tracklets, cat_video_tracks


def census(rows: list[dict[str, Any]], tracklets: dict[tuple[int, int], dict[str, Any]], cat_video_tracks: dict[int, dict[int, list[str]]]) -> tuple[list[int], dict[str, Any]]:
    eligible = sorted(c for c, videos in cat_video_tracks.items() if len(videos) >= 2)
    cats = {}
    for c in eligible:
        rr = [r for r in rows if r["gt_role_common"] == "novel" and r["category_i"] == c]
        all_tracks = {(r["video_i"], r["track_i"]) for r in rr}
        reliable_rows = [r for r in rr if r["reliable_exact"]]
        reliable_tracks = {(r["video_i"], r["track_i"]) for r in reliable_rows}
        by_video = {}
        for v in sorted({r["video_i"] for r in rr}):
            rv = [r for r in rr if r["video_i"] == v]
            relv = [r for r in rv if r["reliable_exact"]]
            by_video[str(v)] = {
                "rows": len(rv),
                "reliable_rows": len(relv),
                "proposal_tracklets": len({r["track_i"] for r in rv}),
                "reliable_proposal_tracklets": len({r["track_i"] for r in relv}),
            }
        cats[str(c)] = {
            "rows": len(rr),
            "assigned_rows": sum(r["assigned"] == "1" for r in rr),
            "reliable_rows": len(reliable_rows),
            "videos": len({r["video_i"] for r in rr}),
            "reliable_videos": len({r["video_i"] for r in reliable_rows}),
            "proposal_tracklets": len(all_tracks),
            "reliable_proposal_tracklets": len(reliable_tracks),
            "mean_proposal_tracklet_rows": float(np.mean([tracklets[k]["rows"] for k in all_tracks])),
            "mean_area_fraction": float(np.mean([float(r["area_fraction"]) for r in rr])),
            "by_video": by_video,
        }
    result = {
        "protocol": "trackocd_iclr27_phase18_eligible_category_census",
        "selection_uses_feature_values": False,
        "selection_uses_model_predictions": False,
        "reliability": "assigned == 1 and exact current row IoU >= 0.5",
        "eligible_definition": "novel GT category with >=1 reliable physical tracklet in >=2 videos",
        "eligible_categories": eligible,
        "eligible_category_count": len(eligible),
        "eligible_rows": sum(x["rows"] for x in cats.values()),
        "eligible_reliable_rows": sum(x["reliable_rows"] for x in cats.values()),
        "categories": cats,
    }
    assert len(eligible) == 11, eligible
    assert result["eligible_rows"] == 377, result["eligible_rows"]
    assert result["eligible_reliable_rows"] == 221, result["eligible_reliable_rows"]
    return eligible, result


def folds_for(eligible: list[int], census_value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(eligible, key=lambda c: (-census_value["categories"][str(c)]["reliable_rows"], c))
    held: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for i, c in enumerate(ordered):
        block, offset = divmod(i, N_FOLDS)
        fold = offset if block % 2 == 0 else N_FOLDS - 1 - offset
        held[fold].append(c)
    cat_videos: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        if r["category_i"] >= 0:
            cat_videos[r["category_i"]].add(r["video_i"])
    entries = []
    for f in range(N_FOLDS):
        held_cats = sorted(held[f])
        cal_cats = sorted(held[(f + 1) % N_FOLDS])
        held_videos = sorted(set().union(*(cat_videos[c] for c in held_cats)))
        fit = [
            r for r in rows
            if r["role17"] in FIT_ROLES
            and r["category_i"] not in set(held_cats + cal_cats)
            and r["video_i"] not in set(held_videos)
        ]
        cal = [
            r for r in rows
            if r["category_i"] in set(cal_cats)
            and r["video_i"] not in set(held_videos)
        ]
        held_rows = [r for r in rows if r["category_i"] in set(held_cats)]
        entry = {
            "fold": f,
            "held_categories": held_cats,
            "nested_calibration_categories": cal_cats,
            "strict_excluded_videos": held_videos,
            "fit_roles": sorted(FIT_ROLES),
            "fit_row_count": len(fit),
            "fit_row_keys_sha256": sha_lines(r["row_key"] for r in fit),
            "fit_category_count": len({r["category_i"] for r in fit if r["category_i"] >= 0}),
            "nested_calibration_row_count": len(cal),
            "nested_calibration_row_keys_sha256": sha_lines(r["row_key"] for r in cal),
            "held_row_count": len(held_rows),
            "held_row_keys_sha256": sha_lines(r["row_key"] for r in held_rows),
            "held_categories_in_fit": sorted(set(held_cats) & {r["category_i"] for r in fit}),
            "calibration_categories_in_fit": sorted(set(cal_cats) & {r["category_i"] for r in fit}),
            "held_videos_in_fit": sorted(set(held_videos) & {r["video_i"] for r in fit}),
        }
        assert not entry["held_categories_in_fit"]
        assert not entry["calibration_categories_in_fit"]
        assert not entry["held_videos_in_fit"]
        entries.append(entry)
    result = {
        "protocol": "trackocd_iclr27_phase18_category_video_safe_crossfit",
        "construction": "sort by descending reliable-row count/category ID; four-fold alternating serpentine placement",
        "outer_fold_count": N_FOLDS,
        "nested_calibration": "next outer fold cyclically; categories excluded from fitting losses",
        "public_role": "cross-fit public development evidence; not blind external test",
        "ordered_categories_for_placement": ordered,
        "folds": entries,
    }
    result["fold_sha256"] = canonical_sha(result)
    return result


def build_events(eligible: list[int], folds: dict[str, Any], tracklets: dict[tuple[int, int], dict[str, Any]], cat_video_tracks: dict[int, dict[int, list[str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {x["tracklet_key"]: x for x in tracklets.values()}
    cat_to_fold = {
        c: f["fold"] for f in folds["folds"] for c in f["held_categories"]
    }
    positives = []
    for c in eligible:
        videos = cat_video_tracks[c]
        for source_video, target_video in itertools.permutations(sorted(videos), 2):
            source_keys = list(videos[source_video])
            for target_key in videos[target_video]:
                target = by_key[target_key]
                prefix = target["first_reliable_prefix_index_gt_only"]
                assert prefix >= 0
                event_key = f"pos:c{c}:sv{source_video}:tv{target_video}:tt{target_key}"
                positives.append({
                    "event_key": event_key,
                    "kind": "positive_existing",
                    "fold": cat_to_fold[c],
                    "category_gt_denominator_only": c,
                    "source_video": source_video,
                    "target_video": target_video,
                    "source_tracklet_keys": source_keys,
                    "target_tracklet_key": target_key,
                    "target_first_reliable_prefix_index_gt_only": prefix,
                    "target_pre_prefix_row_keys": target["row_keys"][:prefix],
                    "target_post_prefix_row_keys": target["row_keys"][prefix:],
                    "target_all_row_keys": target["row_keys"],
                    "expected_first_commit": "EXISTING_NOVEL(source_state)",
                    "source_row_keys_sha256": sha_lines(
                        k for tk in source_keys for k in by_key[tk]["row_keys"]
                    ),
                    "target_row_keys_sha256": sha_lines(target["row_keys"]),
                })
    positives.sort(key=lambda x: x["event_key"])
    negatives = []
    fold_cats = {f["fold"]: f["held_categories"] for f in folds["folds"]}
    for p in positives:
        cats = fold_cats[p["fold"]]
        pos = cats.index(p["category_gt_denominator_only"])
        distractor_cat = cats[(pos + 1) % len(cats)]
        candidate_videos = [v for v in sorted(cat_video_tracks[distractor_cat]) if v != p["target_video"]]
        assert candidate_videos
        distractor_video = candidate_videos[0]
        source_keys = list(cat_video_tracks[distractor_cat][distractor_video])
        negatives.append({
            "event_key": p["event_key"].replace("pos:", "neg:", 1),
            "paired_positive_event_key": p["event_key"],
            "kind": "negative_new",
            "fold": p["fold"],
            "target_category_gt_denominator_only": p["category_gt_denominator_only"],
            "distractor_category_gt_denominator_only": distractor_cat,
            "source_video": distractor_video,
            "target_video": p["target_video"],
            "source_tracklet_keys": source_keys,
            "target_tracklet_key": p["target_tracklet_key"],
            "target_first_reliable_prefix_index_gt_only": p["target_first_reliable_prefix_index_gt_only"],
            "target_pre_prefix_row_keys": p["target_pre_prefix_row_keys"],
            "target_post_prefix_row_keys": p["target_post_prefix_row_keys"],
            "target_all_row_keys": p["target_all_row_keys"],
            "expected_first_commit": "NEW_NOVEL",
        })
    assert len(positives) == 41, len(positives)
    assert len(negatives) == 41
    return positives, negatives


def write_manifests(rows: list[dict[str, Any]], fieldnames: list[str], tracklets: dict[tuple[int, int], dict[str, Any]], positives: list[dict[str, Any]], negatives: list[dict[str, Any]], folds: dict[str, Any], alignment: dict[str, Any], census_value: dict[str, Any]) -> None:
    cat_to_fold = {c: f["fold"] for f in folds["folds"] for c in f["held_categories"]}
    event_membership: dict[str, list[str]] = defaultdict(list)
    for e in positives + negatives:
        for key in e["source_tracklet_keys"] + [e["target_tracklet_key"]]:
            event_membership[key].append(e["event_key"])
    # Compact but complete row-aligned manifest.
    columns = [
        "source_row_index", "row_key", "dinov2_index", "dinov3_index", "video_id",
        "frame_id", "event_rank", "track_id", "role17", "gt_role_common",
        "gt_category_id_common", "assigned", "row_iou", "score", "area_fraction",
        "image_width", "image_height", "bbox_xyxy", "causal_prefix_count",
        "causal_prefix_age", "causal_prefix_age_norm", "causal_box_stability_iou",
        "eligible_outer_fold_gt_only",
    ]
    row_path = OUT / "manifests/row_aligned_tracklet_manifest.csv"
    tmp = row_path.with_name(row_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in columns}
            out["eligible_outer_fold_gt_only"] = cat_to_fold.get(r["category_i"], "")
            w.writerow(out)
    os.replace(tmp, row_path)
    tracklet_lines = []
    for item in sorted(tracklets.values(), key=lambda x: x["tracklet_key"]):
        out = dict(item)
        out["eligible_outer_fold_gt_only"] = cat_to_fold.get(item["label_category_gt_only"])
        out["fixed_event_memberships_gt_only"] = sorted(event_membership.get(item["tracklet_key"], []))
        tracklet_lines.append(json.dumps(out, sort_keys=True))
    atomic_text(OUT / "manifests/tracklet_manifest.jsonl", "\n".join(tracklet_lines) + "\n")
    atomic_text(OUT / "episodes/identifiable_positive_events.jsonl", "\n".join(json.dumps(x, sort_keys=True) for x in positives) + "\n")
    atomic_text(OUT / "episodes/identifiable_negative_events.jsonl", "\n".join(json.dumps(x, sort_keys=True) for x in negatives) + "\n")
    atomic_json(OUT / "manifests/feature_alignment.json", alignment)
    atomic_json(OUT / "manifests/eligible_category_census.json", census_value)
    atomic_json(OUT / "manifests/fold_manifest.json", folds)


def denominator_artifact(positives: list[dict[str, Any]], negatives: list[dict[str, Any]], folds: dict[str, Any]) -> dict[str, Any]:
    per_fold = {}
    for f in range(N_FOLDS):
        pp = [x for x in positives if x["fold"] == f]
        nn = [x for x in negatives if x["fold"] == f]
        per_fold[str(f)] = {
            "positive_events": len(pp),
            "negative_events": len(nn),
            "categories": sorted({x["category_gt_denominator_only"] for x in pp}),
            "source_videos": sorted({x["source_video"] for x in pp}),
            "target_videos": sorted({x["target_video"] for x in pp}),
            "target_tracklets": len({x["target_tracklet_key"] for x in pp}),
            "positive_event_keys_sha256": sha_lines(x["event_key"] for x in pp),
            "negative_event_keys_sha256": sha_lines(x["event_key"] for x in nn),
        }
    result = {
        "protocol": "trackocd_iclr27_phase18_identifiable_ct_denominators",
        "created_before_model_predictions": True,
        "prediction_independent": True,
        "primary_iou": PRIMARY_IOU,
        "reliability": "assigned == 1 and exact current row IoU >= 0.5",
        "positive_event_definition": "each eligible category, each ordered distinct reliable-video pair, each reliable target physical tracklet",
        "primary_event_start": "target first exact reliable prefix",
        "positive_event_count": len(positives),
        "negative_event_count": len(negatives),
        "eligible_category_count": len({x["category_gt_denominator_only"] for x in positives}),
        "target_tracklet_appearances": len(positives),
        "unique_target_tracklets": len({x["target_tracklet_key"] for x in positives}),
        "events_with_unreliable_prefix": sum(x["target_first_reliable_prefix_index_gt_only"] > 0 for x in positives),
        "events_reliable_at_first_row": sum(x["target_first_reliable_prefix_index_gt_only"] == 0 for x in positives),
        "post_prefix_rows": sum(len(x["target_post_prefix_row_keys"]) for x in positives),
        "pre_prefix_rows": sum(len(x["target_pre_prefix_row_keys"]) for x in positives),
        "positive_event_keys": [x["event_key"] for x in positives],
        "positive_event_keys_sha256": sha_lines(x["event_key"] for x in positives),
        "positive_target_post_prefix_row_keys_sha256": sha_lines(k for x in positives for k in x["target_post_prefix_row_keys"]),
        "negative_event_keys": [x["event_key"] for x in negatives],
        "negative_event_keys_sha256": sha_lines(x["event_key"] for x in negatives),
        "fold_manifest_sha256": folds["fold_sha256"],
        "per_fold": per_fold,
    }
    result["denominator_sha256"] = canonical_sha(result)
    return result


def oracle_contracts(positives: list[dict[str, Any]], denominators: dict[str, Any]) -> dict[str, Any]:
    n = len(positives)
    old_local_success = sum(x["target_first_reliable_prefix_index_gt_only"] == 0 for x in positives)
    result = {
        "protocol": "trackocd_iclr27_phase18_oracle_contracts_pretraining",
        "denominator_sha256": denominators["denominator_sha256"],
        "oracles": {
            "O0_SEMANTIC_LABEL_ORACLE_UNCONSTRAINED": {
                "deployable": False, "uses_gt_semantic_label": True,
                "commit_ct_numerator": n, "commit_ct_denominator": n, "commit_ct_recall": 1.0,
                "interpretation": "unconstrained evaluator ceiling",
            },
            "O1_LEGAL_RELIABILITY_AND_SEMANTIC_ORACLE": {
                "deployable": False, "chronology_legal": True,
                "uses_exact_reliability": True, "uses_gt_semantic_correspondence": True,
                "commit_ct_numerator": n, "commit_ct_denominator": n, "commit_ct_recall": 1.0,
                "category_count": 11,
                "interpretation": "source state is born before target; target defers before exact reliable prefix and then assigns EXISTING",
            },
            "O2_LEGAL_SEMANTIC_ORACLE_LEARNED_RELIABILITY": {
                "deployable": False, "pending_learned_component": "reliability/readiness predictions",
                "denominator": n,
            },
            "O3_LEARNED_SEMANTIC_EXACT_RELIABILITY": {
                "deployable": False, "pending_learned_component": "semantic-state predictions",
                "denominator": n,
            },
            "O4_OLD_LOCAL_FIRST_PERFECT_PAIR": {
                "deployable": False, "chronology_legal": True,
                "commit_ct_numerator": old_local_success, "commit_ct_denominator": n,
                "commit_ct_recall": old_local_success / n,
                "failure_mode": "an unreliable first target row creates/locks a local NEW before later reliable evidence",
            },
            "O5_MERGE_CAPABLE_PERFECT_PAIR": {
                "deployable": False, "chronology_legal": True,
                "commit_ct_numerator": n, "commit_ct_denominator": n, "commit_ct_recall": 1.0,
                "interpretation": "legal DEFER then current/future merge; earlier actions immutable",
            },
        },
        "identifiability_passed": n > 0 and len({x["category_gt_denominator_only"] for x in positives}) == 11,
    }
    assert result["identifiability_passed"]
    assert result["oracles"]["O1_LEGAL_RELIABILITY_AND_SEMANTIC_ORACLE"]["commit_ct_numerator"] > 0
    return result


def main() -> None:
    rows, fieldnames = load_rows()
    alignment = feature_alignment(rows)
    tracklets, cat_video_tracks = build_tracklets(rows)
    eligible, census_value = census(rows, tracklets, cat_video_tracks)
    folds = folds_for(eligible, census_value, rows)
    positives, negatives = build_events(eligible, folds, tracklets, cat_video_tracks)
    write_manifests(rows, fieldnames, tracklets, positives, negatives, folds, alignment, census_value)
    denominators = denominator_artifact(positives, negatives, folds)
    atomic_json(OUT / "manifests/identifiable_ct_denominators.json", denominators)
    oracles = oracle_contracts(positives, denominators)
    atomic_json(OUT / "eval/oracle_contracts.json", oracles)
    summary = {
        "rows": len(rows), "tracklets": len(tracklets), "eligible_categories": eligible,
        "positive_events": len(positives), "negative_events": len(negatives),
        "events_with_unreliable_prefix": denominators["events_with_unreliable_prefix"],
        "folds": [{"fold": f["fold"], "held": f["held_categories"], "cal": f["nested_calibration_categories"], "fit_rows": f["fit_row_count"]} for f in folds["folds"]],
        "O1_commit_ct": oracles["oracles"]["O1_LEGAL_RELIABILITY_AND_SEMANTIC_ORACLE"]["commit_ct_recall"],
        "O4_commit_ct": oracles["oracles"]["O4_OLD_LOCAL_FIRST_PERFECT_PAIR"]["commit_ct_recall"],
        "O5_commit_ct": oracles["oracles"]["O5_MERGE_CAPABLE_PERFECT_PAIR"]["commit_ct_recall"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
