"""Phase 4M semantic-resolution metrics from semantic logs.

Measures the UNRESOLVED_NOVEL policy honestly:
  - deferral rate (novel-like frames left unresolved)
  - immediate vs eventual resolution coverage
  - unresolved-at-termination rate
  - resolution latency mean / median / p90
  - resolved action mix (EXISTING vs NEW)
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
    "PHASE4M_TAO_JSON",
    ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" /
    "validation_20.json"))
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"

RESOLVED_ACTIONS = {"NEW_NOVEL", "EXISTING_NOVEL",
                    "EXISTING_NOVEL_PROVISIONAL"}
UNRESOLVED_ACTIONS = {"UNRESOLVED_NOVEL", "PROVISIONAL_NOVEL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    from src.iclr27_phase4j.semantic_eval import (
        load_gt,
        load_logs,
        tracklet_roles,
    )
    known = set(json.loads(KNOWN_IDS.read_text()))
    gt = load_gt(known)
    rows = load_logs(args.log_root)
    roles = tracklet_roles(rows, gt)
    per_track = defaultdict(list)
    for r in rows:
        tid = r.get("physical_track_id")
        if tid is not None:
            per_track[(r.get("video_id"), tid)].append(r)

    novel_like_tracks = 0
    novel_role_tracks = 0
    novel_frames = 0
    unresolved_frames = 0
    tracks_ever_deferred = 0
    immediate_resolved = 0
    eventual_resolved = 0
    unresolved_at_term = 0
    latencies = []
    action_mix = Counter()
    detail = []
    for key, seq in per_track.items():
        seq = sorted(seq, key=lambda r: (r["frame_id"], r["det_idx"]))
        role = roles.get(key, {}).get("role", "fp")
        nf = [r for r in seq
              if r["p_known"] < r.get("decision_threshold", 0.5)]
        if not nf:
            continue
        novel_like_tracks += 1
        if role == "novel":
            novel_role_tracks += 1
        novel_frames += len(nf)
        unresolved = sum(1 for r in nf
                         if r.get("semantic_action") in UNRESOLVED_ACTIONS)
        unresolved_frames += unresolved
        ever_unresolved = unresolved > 0
        first_decision = next(
            (r for r in nf if r.get("semantic_action") in RESOLVED_ACTIONS),
            None)
        first_unresolved = next(
            (r for r in nf
             if r.get("semantic_action") in UNRESOLVED_ACTIONS), None)
        resolved = first_decision is not None
        immediate = first_decision is not None and (
            first_unresolved is None or
            first_decision["frame_id"] <= first_unresolved["frame_id"])
        deferred_then_resolved = (
            first_decision is not None and first_unresolved is not None and
            first_unresolved["frame_id"] < first_decision["frame_id"])
        if ever_unresolved:
            tracks_ever_deferred += 1
        if immediate:
            immediate_resolved += 1
        if resolved:
            eventual_resolved += 1
            if deferred_then_resolved:
                latencies.append(first_decision["frame_id"] -
                                 first_unresolved["frame_id"])
            action = first_decision.get("semantic_action")
            action_mix[action] += 1
        elif first_unresolved is not None:
            unresolved_at_term += 1
        detail.append({
            "video_id": key[0], "track_id": key[1], "role": role,
            "novel_frames": len(nf), "unresolved_frames": unresolved,
            "ever_unresolved": int(ever_unresolved),
            "resolved": int(resolved),
            "immediate_resolved": int(immediate),
            "unresolved_at_termination": int(
                first_unresolved is not None and not resolved),
            "resolution_frame": (first_decision["frame_id"]
                                 if first_decision is not None else ""),
            "first_unresolved_frame": (first_unresolved["frame_id"]
                                       if first_unresolved is not None
                                       else ""),
        })

    lat = np.asarray(latencies, dtype=float) if latencies else \
        np.asarray([], dtype=float)
    summary = {
        "novel_like_tracklets": novel_like_tracks,
        "gt_novel_tracklets": novel_role_tracks,
        "novel_like_frames": novel_frames,
        "deferral_rate_frames": round(
            unresolved_frames / max(novel_frames, 1), 4),
        "tracklets_ever_deferred": tracks_ever_deferred,
        "deferral_rate_tracks": round(
            tracks_ever_deferred / max(novel_like_tracks, 1), 4),
        "immediate_resolution_coverage": round(
            immediate_resolved / max(novel_like_tracks, 1), 4),
        "eventual_resolution_coverage": round(
            eventual_resolved / max(novel_like_tracks, 1), 4),
        "unresolved_at_termination_rate": round(
            unresolved_at_term / max(novel_like_tracks, 1), 4),
        "resolution_latency_mean": round(
            float(lat.mean()), 4) if len(lat) else "",
        "resolution_latency_median": round(
            float(np.median(lat)), 4) if len(lat) else "",
        "resolution_latency_p90": round(
            float(np.percentile(lat, 90)), 4) if len(lat) else "",
        "resolved_action_mix": dict(action_mix),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    detail_out = args.out.with_name(args.out.stem + "_tracklets.csv")
    with open(detail_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    print(json.dumps(summary, indent=1))
    print("RESOLUTION_METRICS_DONE", args.out)


if __name__ == "__main__":
    main()
