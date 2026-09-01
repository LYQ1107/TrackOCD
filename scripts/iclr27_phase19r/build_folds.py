"""Build balanced eligible folds and evaluator-isolated held-known events."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources"
OUT = ROOT / "outputs/iclr27_phase19r"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    os.replace(tmp, path)


def sha_rows(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(x, sort_keys=True) for x in rows).encode()).hexdigest()


def main() -> None:
    rows = list(csv.DictReader((SRC / "public_rows_corrected.csv").open(newline="")))
    supported = sorted(int(x) for x in json.loads((SRC / "supported_known_ids.json").read_text()))
    sup = set(supported)
    z = np.load(SRC / "public_cls_roi.npz", mmap_mode="r")
    raw = .8 * np.asarray(z["cls"], np.float32) + .2 * np.asarray(z["roi"], np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-6)
    tracks: dict[str, list[int]] = defaultdict(list); cat_tracks: dict[int, list[str]] = defaultdict(list)
    video: dict[str, int] = {}; category: dict[str, int] = {}; role: dict[str, str] = {}
    for i, r in enumerate(rows):
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"
        tracks[key].append(i); video[key] = int(r["video_id"]); category[key] = int(r["gt_category_id_common"]); role[key] = r.get("gt_role_common", "")
    for key, idx in tracks.items():
        idx.sort(key=lambda i: (int(rows[i]["event_rank"]), i))
        c = category[key]
        if c in sup and role[key] == "supported_known": cat_tracks[c].append(key)
    cat_videos: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for c, keys in cat_tracks.items():
        for k in sorted(set(keys)): cat_videos[c][video[k]].append(k)
    eligible = {}
    track_vec = {}
    for c in supported:
        by = cat_videos.get(c, {})
        keys = [k for v in by.values() for k in v]
        vecs = []
        for k in keys:
            idx = tracks[k]; q = np.asarray([max(.02, .62*float(rows[i]["score"]) + .23*float(rows[i]["causal_box_stability_iou"])) for i in idx], np.float32)
            v = np.average(raw[idx], axis=0, weights=q); v /= max(float(np.linalg.norm(v)), 1e-6); track_vec[k] = v.astype(np.float32); vecs.append(v)
        eligible[c] = {
            "eligible": bool(len(by) >= 4), "reason": "at_least_four_videos" if len(by) >= 4 else "fewer_than_four_videos",
            "videos": sorted(int(v) for v in by), "video_count": len(by), "track_count": len(keys),
            "rows": int(sum(len(tracks[k]) for k in keys)), "tracks_by_video": {str(v): len(by[v]) for v in sorted(by)},
            "raw_track_mean": float(np.mean(vecs)) if vecs else 0.0,
        }
    eligible_cats = [c for c in supported if eligible[c]["eligible"]]
    # Greedy serpentine assignment balances both track and video load.
    loads = [{"tracks": 0, "videos": 0, "difficulty": 0.0, "categories": []} for _ in range(4)]
    order = sorted(eligible_cats, key=lambda c: (-eligible[c]["track_count"], -eligible[c]["video_count"], c))
    for i, c in enumerate(order):
        # Seed each fold with three categories before load balancing.  This
        # keeps every held validation stream able to form the required
        # 3-category mixed episode; the very large category 805 is then
        # isolated from most track-count mass and event counts are capped.
        if i < 12:
            j = i % 4
        else:
            j = min(range(4), key=lambda k: (loads[k]["tracks"], loads[k]["videos"], k))
        loads[j]["categories"].append(c); loads[j]["tracks"] += eligible[c]["track_count"]; loads[j]["videos"] += eligible[c]["video_count"]; loads[j]["difficulty"] += abs(eligible[c]["raw_track_mean"])
    category_split = {}
    for c in supported:
        vids = sorted(cat_videos[c])
        if len(vids) >= 4:
            cut = max(2, len(vids) // 2); cut = min(cut, len(vids) - 2)
            category_split[str(c)] = {"source_videos": vids[:cut], "target_videos": vids[cut:], "all_videos": vids, "eligible": True}
        elif len(vids) >= 2:
            cut = max(1, len(vids) - 1)
            category_split[str(c)] = {"source_videos": vids[:cut], "target_videos": vids[cut:], "all_videos": vids, "eligible": False}
    folds = []
    pos_events: list[dict] = []; neg_events: list[dict] = []
    for fold, load in enumerate(loads):
        held = sorted(load["categories"])
        held_videos = sorted({v for c in held for v in category_split[str(c)]["all_videos"]})
        fit_videos = sorted({v for c in supported if c not in held for v in category_split.get(str(c), {}).get("source_videos", []) if v not in held_videos})
        validation_videos = held_videos
        held_tracks = [k for c in held for v in category_split[str(c)]["target_videos"] for k in cat_videos[c][v]]
        fit_tracks = [k for c in supported if c not in held for k in cat_tracks.get(c, []) if video[k] in fit_videos]
        folds.append({"fold": fold, "held_categories": held, "fit_videos": fit_videos,
                      "validation_videos": validation_videos, "held_track_count": len(held_tracks),
                      "fit_track_count": len(fit_tracks), "held_video_count": len(held_videos),
                      "held_category_video_counts": {str(c): {"source": len(category_split[str(c)]["source_videos"]), "target": len(category_split[str(c)]["target_videos"])} for c in held}})

        # Equal cap: two directed source/target events per source-target pair,
        # capped at four positive events per category.
        for c in held:
            src_keys = [k for v in category_split[str(c)]["source_videos"] for k in cat_videos[c][v]]
            tgt_keys = [k for v in category_split[str(c)]["target_videos"] for k in cat_videos[c][v]]
            pairs = [(s, t) for s in src_keys for t in tgt_keys if video[s] != video[t]][:4]
            for n, (s, t) in enumerate(pairs):
                ekey = f"p19r-pos:f{fold}:c{c}:s{video[s]}:t{video[t]}:n{n}"
                pos = {"event_key": ekey, "kind": "positive_existing", "fold": fold,
                       "category_gt_denominator_only": c, "source_tracklet_keys": [s], "source_video": video[s],
                       "target_tracklet_key": t, "target_video": video[t],
                       "target_row_keys": [rows[i]["row_key"] for i in tracks[t]],
                       "target_first_reliable_prefix_index_gt_only": max(0, next((j for j, i in enumerate(tracks[t]) if rows[i]["assigned"] == "1" and float(rows[i]["row_iou"]) >= .5), len(tracks[t]) - 1)),
                       "expected_first_commit": "EXISTING_NOVEL(source_state)"}
                pos_events.append(pos)
                # Hard negative: nearest legal track from another category in
                # the same fold, with source/target videos disjoint.
                best = None
                for d in held:
                    if d == c: continue
                    for ds in cat_tracks.get(d, []):
                        if video[ds] == video[t]: continue
                        score = float(track_vec.get(ds, np.zeros(768)) @ track_vec.get(t, np.zeros(768)))
                        if best is None or score > best[0]: best = (score, ds, d)
                if best is None:
                    for d in eligible_cats:
                        if d == c: continue
                        for ds in cat_tracks.get(d, []):
                            if video[ds] != video[t]:
                                score = float(track_vec[ds] @ track_vec[t])
                                if best is None or score > best[0]: best = (score, ds, d)
                assert best is not None
                neg = {"event_key": ekey.replace("p19r-pos", "p19r-neg"), "kind": "negative_new", "fold": fold,
                       "distractor_category_gt_denominator_only": int(best[2]), "target_category_gt_denominator_only": c,
                       "source_tracklet_keys": [best[1]], "source_video": video[best[1]], "target_tracklet_key": t,
                       "target_video": video[t], "target_row_keys": [rows[i]["row_key"] for i in tracks[t]],
                       "target_first_reliable_prefix_index_gt_only": pos["target_first_reliable_prefix_index_gt_only"],
                       "raw_hard_negative_similarity": float(best[0]), "expected_first_commit": "NEW_NOVEL"}
                neg_events.append(neg)
    atomic_json(OUT / "audit/eligible_category_audit.json", {"protocol": "trackocd_iclr27_phase19r_eligibility", "supported_count": len(supported), "eligible_count": len(eligible_cats), "categories": {str(c): eligible[c] for c in supported}, "eligible_categories": eligible_cats})
    atomic_json(OUT / "manifests/category_video_split.json", {"protocol": "trackocd_iclr27_phase19r_category_video_split", "splits": category_split, "eligible_categories": eligible_cats})
    manifest = {"protocol": "trackocd_iclr27_phase19r_balanced_folds", "source_rows": len(rows), "supported_ids": supported, "eligible_categories": eligible_cats, "folds": folds,
                "positive_event_count": len(pos_events), "negative_event_count": len(neg_events),
                "event_keys_sha256": {"positive": sha_rows(pos_events), "negative": sha_rows(neg_events)},
                "model_facing_event_fields": ["event_key", "fold", "source_tracklet_keys", "target_tracklet_key", "target_video", "role"],
                "evaluator_only_truth": "category fields are isolated in held_known_*_events.jsonl"}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(OUT / "manifests/fold_manifest.json", manifest)
    atomic_jsonl(OUT / "manifests/held_known_positive_events.jsonl", pos_events)
    atomic_jsonl(OUT / "manifests/held_known_negative_events.jsonl", neg_events)
    model_events = []
    for e in pos_events + neg_events:
        model_events.append({"event_key": e["event_key"], "kind": e["kind"], "fold": e["fold"], "source_tracklet_keys": e["source_tracklet_keys"], "target_tracklet_key": e["target_tracklet_key"], "target_video": e["target_video"], "role": "pseudo_novel_cross_video"})
    atomic_jsonl(OUT / "manifests/held_known_model_events.jsonl", sorted(model_events, key=lambda x: x["event_key"]))
    atomic_json(OUT / "audit/fold_build_summary.json", {"protocol": "trackocd_iclr27_phase19r_fold_audit", "folds": folds, "eligible_categories": eligible_cats, "positive_events": len(pos_events), "negative_events": len(neg_events), "source_target_video_disjoint": all(e["source_video"] != e["target_video"] for e in pos_events + neg_events), "category_event_cap": 4})
    print(json.dumps({"complete": True, "eligible_categories": eligible_cats, "folds": folds, "positive_events": len(pos_events), "negative_events": len(neg_events), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
