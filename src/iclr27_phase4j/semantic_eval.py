"""Phase 4J semantic evaluation on the 20-video subset.

Grounds every logged detection to TAO GT by IoU >= 0.5 (offline only) and
reports the Phase 4I routing metrics plus the Phase 4J commitment metrics:

- FP Novel Observation Rate      (detections routed novel)
- FP Global Memory Admission Rate (tracklets that ever write global memory)
- FP Stable Novel Birth Rate      (NEW_NOVEL birth whose prototype has
                                   final support >= 2)
- commit coverage / latency per GT-known / GT-novel / FP tracklets
- global novel memory size at end
- valid novel reuse (GT-novel tracklets committing to an id born elsewhere)
- known-absorption of FP-born prototypes (pollution -> association link)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_JSON = Path(os.environ.get(
    "PHASE4L_TAO_JSON",
    ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" /
    "validation_20.json"))
KNOWN_IDS = Path(os.environ.get(
    "PHASE4L_KNOWN_IDS",
    ROOT / "data" / "trackocd_v1" / "pure" / "splits" /
    "supported_known_ids.json"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(known_ids):
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        cat = int(ann["category_id"])
        bbox = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            "track_id": int(ann["track_id"]),
            "category_id": cat,
            "role": "known" if cat in known_ids else "novel",
        })
    return out


def load_logs(log_root):
    rows = []
    for p in sorted(log_root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tracklet_roles(rows, gt):
    """Role per physical tracklet: majority of matched GT roles; FP if
    never matched.  Also records GT track ids covered by the tracklet."""
    matched = defaultdict(list)
    for r in rows:
        if r.get("physical_track_id") is None:
            continue
        key = (r.get("video_id"), r["physical_track_id"])
        best, bi = None, 0.5
        for g in gt.get(int(r["image_id"]), []):
            v = iou(r["bbox"], g["bbox"])
            if v >= bi:
                bi, best = v, g
        if best is not None:
            matched[key].append(best["role"])
    roles = {}
    for key, rs in matched.items():
        c = Counter(rs)
        roles[key] = {"role": c.most_common(1)[0][0]}
    # keep gt track ids too
    gt_of_track = defaultdict(set)
    for r in rows:
        tid = r.get("physical_track_id")
        if tid is None:
            continue
        key = (r.get("video_id"), tid)
        best, bi = None, 0.5
        for g in gt.get(int(r["image_id"]), []):
            v = iou(r["bbox"], g["bbox"])
            if v >= bi:
                bi, best = v, g
        if best is not None:
            gt_of_track[key].add(best["track_id"])
    for key in list(roles):
        roles[key]["gt_track_ids"] = gt_of_track.get(key, set())
    return roles


def known_commit_frame(seq, thr_field="decision_threshold"):
    """First age>=2 where two consecutive frames are KNOWN with the same
    known class; returns commit age or None."""
    seq = sorted(seq, key=lambda r: (r["frame_id"], r["det_idx"]))
    prev = None
    for r in seq:
        thr = r.get(thr_field, 0.5)
        pred = "known" if r["p_known"] >= thr else "novel"
        kc = r.get("known_class_id")
        if prev is not None and pred == "known" and prev["pred"] == "known" \
                and prev["kc"] == kc:
            return int(r["track_age"])
        prev = {"pred": pred, "kc": kc}
    return None


def evaluate(log_root, out_csv, out_track_csv):
    known = set(json.loads(KNOWN_IDS.read_text()))
    gt = load_gt(known)
    rows = load_logs(log_root)
    roles = tracklet_roles(rows, gt)

    per_track = defaultdict(list)
    global_support = defaultdict(int)
    for r in rows:
        tid = r.get("physical_track_id")
        if tid is not None:
            per_track[(r.get("video_id"), tid)].append(r)
        gid = r.get("global_novel_id")
        if isinstance(gid, int):
            global_support[gid] = max(global_support[gid],
                                      int(r.get("novel_mem_support", 0)))

    # ---- Phase 4I-style frame metrics on GT-matched detections ----
    stats = Counter()
    track_sem = defaultdict(list)
    gt_novel_ids = defaultdict(set)
    gt_known_ids = defaultdict(set)
    for r in rows:
        tid = r.get("physical_track_id")
        if tid is not None:
            track_sem[(r.get("video_id"), tid)].append(r)
        thr = r.get("decision_threshold", 0.5)
        pred = "known" if r["p_known"] >= thr else "novel"
        best, bi = None, 0.5
        for g in gt.get(int(r["image_id"]), []):
            v = iou(r["bbox"], g["bbox"])
            if v >= bi:
                bi, best = v, g
        if best is None:
            stats["unmatched"] += 1
            continue
        stats["matched"] += 1
        if pred == best["role"]:
            stats["routing_correct"] += 1
        if best["role"] == "known":
            stats["known_matched"] += 1
            if pred == "novel":
                stats["k2n"] += 1
            if pred == "known":
                if r.get("known_class_id") == best["category_id"]:
                    stats["known_class_correct"] += 1
                stats["known_class_total"] += 1
            if pred == "novel" and isinstance(r.get("global_novel_id"), int):
                gt_known_ids[r["physical_track_id"]].add("NOVEL_COMMIT")
        else:
            stats["novel_matched"] += 1
            if pred == "known":
                stats["n2k"] += 1
            if pred == "novel":
                sem = r.get("novel_id")
                gt_novel_ids[best["track_id"]].add(
                    sem if sem is not None else -1)

    sem_switches = 0
    sem_switch_total = 0
    for key, seq in track_sem.items():
        seq = sorted(seq, key=lambda r: (r["frame_id"], r["det_idx"]))
        prev = None
        for r in seq:
            cur = r.get("semantic_id")
            if prev is not None and cur != prev:
                sem_switches += 1
            if prev is not None:
                sem_switch_total += 1
            prev = cur

    # ---- per-tracklet commitment ----
    track_rows = []
    role_counts = Counter()
    role_committed = Counter()
    role_committed_len2 = Counter()
    role_tracklets_len2 = Counter()
    latencies = defaultdict(list)
    fp_obs_rate_num = Counter()
    fp_obs_rate_den = Counter()
    fp_admission = Counter()
    fp_stable_birth = Counter()
    fp_total = Counter()
    new_novel_births = Counter()
    birth_creator = {}
    gt_novel_reuse = 0
    fp_birth_later_used_by_gt_novel = 0
    known_committed_to_novel = 0
    known_committed_to_novel_tracks = set()
    novel_reuse_tracks = set()
    fp_birth_used_tracks = set()

    for key, seq in per_track.items():
        tid = key[1]
        role = roles.get(key, {}).get("role", "fp")
        role_counts[role] += 1
        seq = sorted(seq, key=lambda r: (r["frame_id"], r["det_idx"]))
        len_ge2 = len(seq) >= 2
        if len_ge2:
            role_tracklets_len2[role] += 1
        thr0 = seq[0].get("decision_threshold", 0.5)
        novel_frames = sum(1 for r in seq
                           if r["p_known"] < r.get("decision_threshold", 0.5))
        fp_obs_rate_num[role] += novel_frames
        fp_obs_rate_den[role] += len(seq)
        commit_age = None
        first_commit_action = None
        committed_id = None
        for r in seq:
            if r.get("commit_state") == "committed":
                commit_age = int(r["track_age"])
                committed_id = r.get("global_novel_id")
                break
        if role == "known":
            ca = known_commit_frame(seq)
            if ca is not None:
                commit_age = min(commit_age, ca) if commit_age is not None \
                    else ca
        if commit_age is not None:
            role_committed[role] += 1
            if len_ge2:
                role_committed_len2[role] += 1
            latencies[role].append(max(commit_age - 1, 0))
            # find the commit action on the commit frame
            for r in seq:
                if r.get("track_age") == commit_age:
                    first_commit_action = r.get("semantic_action")
                    break
        if role == "fp":
            fp_total[role] += 1
            if commit_age is not None:
                fp_admission[role] += 1
                if first_commit_action == "NEW_NOVEL" and \
                        committed_id is not None and \
                        global_support.get(committed_id, 0) >= 2:
                    fp_stable_birth[role] += 1
        # novel birth bookkeeping for reuse/pollution analysis
        for r in seq:
            if r.get("semantic_action") == "NEW_NOVEL":
                gid = r.get("global_novel_id")
                if isinstance(gid, int):
                    new_novel_births[gid] += 1
                    birth_creator.setdefault(gid, key)
            if r.get("commit_state") == "committed" and \
                    isinstance(r.get("global_novel_id"), int):
                gid = r["global_novel_id"]
                creator = birth_creator.get(gid)
                if role == "novel" and creator is not None and \
                        creator != key:
                    novel_reuse_tracks.add(key)
                if role == "known":
                    known_committed_to_novel_tracks.add(key)
        track_rows.append({
            "physical_track_id": tid,
            "video_id": key[0],
            "role": role,
            "gt_track_ids": "|".join(str(x) for x in
                                     roles.get(key, {}).get("gt_track_ids", [])),
            "length": len(seq),
            "novel_frames": novel_frames,
            "commit_age": commit_age if commit_age is not None else "",
            "commit_action": first_commit_action or "",
            "committed_id": committed_id if committed_id is not None else "",
        })

    # fp-born prototypes later used by GT-novel tracklets
    for key, seq in per_track.items():
        tid = key[1]
        role = roles.get(key, {}).get("role", "fp")
        if role != "novel":
            continue
        for r in seq:
            gid = r.get("global_novel_id")
            if isinstance(gid, int) and birth_creator.get(gid) is not None \
                    and roles.get(birth_creator[gid], {}).get("role",
                                                              "fp") == "fp":
                fp_birth_used_tracks.add(key)
                break

    def rate(counter, den, default=0.0):
        return round(counter / den, 4) if den else default

    summary = {
        "tag": log_root.name,
        "routing_accuracy": rate(stats["routing_correct"], stats["matched"]),
        "k2n_rate_known_denom": rate(stats["k2n"], stats["known_matched"]),
        "n2k_rate_novel_denom": rate(stats["n2k"], stats["novel_matched"]),
        "known_class_accuracy": rate(stats["known_class_correct"],
                                     stats["known_class_total"]),
        "semantic_id_switch_rate": rate(sem_switches, sem_switch_total),
        "novel_consistency": rate(
            sum(1 for ids in gt_novel_ids.values()
                if len(ids) == 1 and -1 not in ids),
            len(gt_novel_ids)),
        "matched_detections": stats["matched"],
        "unmatched_detections": stats["unmatched"],
        "tracklets_known": role_counts.get("known", 0),
        "tracklets_novel": role_counts.get("novel", 0),
        "tracklets_fp": role_counts.get("fp", 0),
        "commit_coverage_known": rate(role_committed.get("known", 0),
                                      role_counts.get("known", 0)),
        "commit_coverage_novel": rate(role_committed.get("novel", 0),
                                      role_counts.get("novel", 0)),
        "commit_coverage_fp": rate(role_committed.get("fp", 0),
                                   role_counts.get("fp", 0)),
        "commit_coverage_len2_known": rate(
            role_committed_len2.get("known", 0),
            role_tracklets_len2.get("known", 0)),
        "commit_coverage_len2_novel": rate(
            role_committed_len2.get("novel", 0),
            role_tracklets_len2.get("novel", 0)),
        "commit_coverage_len2_fp": rate(
            role_committed_len2.get("fp", 0),
            role_tracklets_len2.get("fp", 0)),
        "commit_latency_mean_known": round(float(np.mean(
            latencies.get("known", [0]) or [0])), 2),
        "commit_latency_mean_novel": round(float(np.mean(
            latencies.get("novel", [0]) or [0])), 2),
        "commit_latency_mean_fp": round(float(np.mean(
            latencies.get("fp", [0]) or [0])), 2),
        "commit_latency_median_known": round(float(np.median(
            latencies.get("known", [0]) or [0])), 2),
        "commit_latency_median_novel": round(float(np.median(
            latencies.get("novel", [0]) or [0])), 2),
        "commit_latency_median_fp": round(float(np.median(
            latencies.get("fp", [0]) or [0])), 2),
        "commit_latency_p90_novel": round(float(np.percentile(
            latencies.get("novel", [0]) or [0], 90)), 2),
        "commit_latency_p90_fp": round(float(np.percentile(
            latencies.get("fp", [0]) or [0], 90)), 2),
        "fp_novel_observation_rate": rate(
            fp_obs_rate_num.get("fp", 0), fp_obs_rate_den.get("fp", 0)),
        "fp_global_memory_admission_rate": rate(
            fp_admission.get("fp", 0), fp_total.get("fp", 0)),
        "fp_stable_novel_birth_rate": rate(
            fp_stable_birth.get("fp", 0), fp_total.get("fp", 0)),
        "global_novel_memory_size": max(
            (int(r.get("n_novel_memory", 0)) for r in rows), default=0),
        "novel_ids_created": len(new_novel_births),
        "gt_novel_reuse_tracks": len(novel_reuse_tracks),
        "fp_birth_later_used_by_gt_novel": len(fp_birth_used_tracks),
        "known_tracks_committed_to_novel": len(
            known_committed_to_novel_tracks),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    with open(out_track_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys())
                           if track_rows else ["physical_track_id"])
        w.writeheader()
        w.writerows(track_rows)
    print(json.dumps(summary, indent=1))


def by_age(log_root, out_csv):
    """Per-age-bucket routing/K2N/N2K on GT-matched detections."""
    known = set(json.loads(KNOWN_IDS.read_text()))
    gt = load_gt(known)
    buckets = [(1, 1, "age1"), (2, 2, "age2"), (3, 4, "age3_4"),
               (5, 8, "age5_8"), (9, 16, "age9_16"), (17, 10 ** 6,
                                                      "age17plus")]
    rows = []
    for b_lo, b_hi, name in buckets:
        stats = Counter()
        for r in load_logs(log_root):
            age = int(r.get("track_age", 1))
            if not (b_lo <= age <= b_hi):
                continue
            thr = r.get("decision_threshold", 0.5)
            pred = "known" if r["p_known"] >= thr else "novel"
            best, bi = None, 0.5
            for g in gt.get(int(r["image_id"]), []):
                v = iou(r["bbox"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            if best is None:
                continue
            stats["n"] += 1
            if pred == best["role"]:
                stats["route"] += 1
            if best["role"] == "known":
                stats["known_n"] += 1
                if pred == "novel":
                    stats["k2n"] += 1
            else:
                stats["novel_n"] += 1
                if pred == "known":
                    stats["n2k"] += 1
        rows.append({
            "bucket": name,
            "n": stats["n"],
            "routing_accuracy": round(stats["route"] / max(stats["n"], 1), 4),
            "k2n": round(stats["k2n"] / max(stats["known_n"], 1), 4),
            "n2k": round(stats["n2k"] / max(stats["novel_n"], 1), 4),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-tracklets", required=True, type=Path)
    ap.add_argument("--by-age-csv", type=Path, default=None)
    args = ap.parse_args()
    evaluate(args.log_root, args.out, args.out_tracklets)
    if args.by_age_csv is not None:
        by_age(args.log_root, args.by_age_csv)


if __name__ == "__main__":
    main()
