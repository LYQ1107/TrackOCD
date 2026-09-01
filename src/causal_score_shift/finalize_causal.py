#!/usr/bin/env python3
"""Generate per-video / shift / subgroup diagnostics for causal methods."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.causal_score_shift.causal_routers import C1Global, C2Translation
from src.causal_score_shift.data.causal_stream import load_video_streams
from src.domain_router.evaluation.run_router import (
    load_frame_dict, val_meta, load_train_known,
)
from src.domain_router.features.router_features import compute_router_features
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_mean_features, build_prototypes
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt

OUT = PROJECT_ROOT / "outputs" / "causal_score_shift" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "causal_score_shift"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    feats_tr, labels = load_train_known("dinov2")
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    protos = build_prototypes(feats_tr, labels, set(labels.values()))
    knn = np.stack([feats_tr[s] for s in feats_tr])
    cache = {s: compute_router_features(feats_val[s], protos, knn,
                                         frames_val.get(s), meta_val.get(s))
             for s in feats_val}
    gt = load_gt("pure")
    vstreams = load_video_streams("main")

    def run(method, router):
        preds = []
        mem = B2Memory(protos, threshold=0.45, novel_only=True)
        per_video = {}
        for vid, vrows in vstreams.items():
            router.reset_video(vid)
            pv = []
            for r in vrows:
                sid = r["sample_id"]
                if sid not in cache:
                    continue
                st = dict(cache[sid])
                is_k = router.predict(st)
                if is_k:
                    bid, _ = max(protos.items(), key=lambda kv: float(np.dot(feats_val[sid], kv[1])))
                    preds.append(emit(sid, r["stream_order"], "known", known_id=bid))
                else:
                    vid2, _ = mem.predict_one(feats_val[sid], sid, r["stream_order"])
                    preds.append(emit(sid, r["stream_order"], "novel", virtual_id=vid2))
                pv.append((sid, is_k))
                router.update_after_prediction(st, is_k)
            per_video[vid] = pv
        return preds, per_video

    # per-video results for C0 and C1 on main stream
    rows_video = []
    for name, router in (("C0", None), ("C1", C1Global(0.42))):
        if name == "C0":
            from src.causal_score_shift.causal_routers import C0Legacy
            router = C0Legacy()
        preds, per_video = run(name, router)
        for vid, pv in per_video.items():
            if not pv:
                continue
            ids = [s for s, _ in pv]
            ev = TrackOCDEvaluator(gt)
            res = ev.evaluate([p for p in preds if p["sample_id"] in set(ids)],
                              subset_ids=set(ids))
            rows_video.append({
                "method": name, "video_id": vid, "tracks": len(pv),
                "known_acc": res["overall_known_acc"],
                "route_novel_acc": res["route_aware_novel_acc"],
                "novel_routing_recall": res["novel_routing_recall"],
            })
    write_csv(OUT / "per_video_results.csv", rows_video)

    # shift trajectories for C2 on a sample of videos
    ref = json.loads((RUNS / "reference_stats.json").read_text())
    shift_rows = []
    hc = ref["hc_score"].get("0.95", ref["hc_score"].get(0.95, 0.9))
    c2 = C2Translation(0.42, ref["ref_known_median"], hc,
                       0.05, 5, 0.25, 2 * ref["ref_known_mad"])
    for vid in sorted(vstreams)[:50]:
        c2.reset_video(vid)
        for r in vstreams[vid]:
            sid = r["sample_id"]
            if sid not in cache:
                continue
            st = dict(cache[sid])
            d = c2.predict(st)
            c2.update_after_prediction(st, d)
        shift_rows.append({
            "video_id": vid, "tracks": len(vstreams[vid]),
            "anchors": len(c2.anchors), "shift_ema": c2.shift_ema,
            "adapted": len(c2.anchors) >= 5,
        })
    write_csv(OUT / "shift_trajectories.csv", shift_rows)

    # subgroups for C0/C1
    from collections import Counter
    cat_count = Counter(g["ground_truth_category_id"] for g in gt if g["protocol_role"] == "novel")
    cat_video = defaultdict(set)
    for g in gt:
        if g["protocol_role"] == "novel":
            cat_video[g["ground_truth_category_id"]].add(int(g["sample_id"].split("_")[0]))
    groups = {
        "all": set(g["sample_id"] for g in gt),
        "singleton_novel": {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and cat_count[g["ground_truth_category_id"]] == 1},
        "repeated_novel": {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and cat_count[g["ground_truth_category_id"]] >= 2},
        "cross_video_novel": {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and len(cat_video[g["ground_truth_category_id"]]) >= 2},
    }
    subgroup_rows = []
    for name, router in (("C0", None), ("C1", C1Global(0.42))):
        if name == "C0":
            from src.causal_score_shift.causal_routers import C0Legacy
            router = C0Legacy()
        preds, _ = run(name, router)
        for gname, ids in groups.items():
            ev = TrackOCDEvaluator(gt)
            res = ev.evaluate([p for p in preds if p["sample_id"] in ids],
                              subset_ids=ids)
            subgroup_rows.append({
                "method": name, "group": gname, "size": len(ids),
                "route_novel_acc": res["route_aware_novel_acc"],
                "routing_recall": res["novel_routing_recall"],
                "known_acc": res["overall_known_acc"],
            })
    write_csv(OUT / "subgroup_results.csv", subgroup_rows)
    print("finalized per-video/shift/subgroup diagnostics")


if __name__ == "__main__":
    main()
