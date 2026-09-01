"""Phase 4T train-vs-dev tracker-stream audit.

Structural distributions (FP ratio, persistence, track length, score,
prior_hits, rows/frame, fragmentation, GT coverage) for the real TRAIN
tracker-induced stream and the frozen Q1 dev stream, plus (once features
exist) single-frame DINO semantic geometry. Writes stream_audit.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()))


def load_stream(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["video_id"] = int(r["video_id"])
            r["frame_id"] = int(r["frame_id"])
            r["track_id"] = int(r["track_id"])
            r["score"] = float(r["score"])
            r["prior_hits"] = int(r.get("prior_hits") or 0)
            r["gt_role"] = r.get("gt_role") or "fp"
            r["gt_iou"] = float(r.get("gt_iou") or 0.0)
            r["gt_category_id"] = int(r.get("gt_category_id") or -1)
            if r.get("gt_track_id") is not None and str(r["gt_track_id"]).strip() != "":
                r["gt_track_id"] = int(r["gt_track_id"])
            else:
                r["gt_track_id"] = -1
            rows.append(r)
    return rows


def tracklet_stats(rows: list[dict]) -> dict:
    by_track = defaultdict(list)
    for r in rows:
        by_track[(r["video_id"], r["track_id"])].append(r)
    fp_len, match_len = [], []
    matched_rows = 0
    for k, rs in by_track.items():
        rs.sort(key=lambda r: (r["frame_id"], r["track_id"]))
        mt = [r for r in rs if r["gt_role"] in ("known", "novel", "novel_role")]
        if mt:
            match_len.append(len(rs))
            matched_rows += len(mt)
        else:
            fp_len.append(len(rs))
    return {
        "n_tracklets": len(by_track),
        "n_fp_tracklets": len(fp_len),
        "fp_tracklet_ratio": round(len(fp_len) / max(len(by_track), 1), 4),
        "persistent_fp_ge2": int(sum(1 for l in fp_len if l >= 2)),
        "persistent_fp_ge4": int(sum(1 for l in fp_len if l >= 4)),
        "fp_tracklet_len": quantiles(fp_len),
        "matched_tracklet_len": quantiles(match_len),
        "matched_rows": matched_rows,
    }


def quantiles(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    a = np.asarray(vals)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "max": round(float(a.max()), 3)}


def row_stats(rows: list[dict]) -> dict:
    scores = np.asarray([r["score"] for r in rows])
    priors = np.asarray([r["prior_hits"] for r in rows], dtype=np.float64)
    role = Counter(r["gt_role"] for r in rows)
    fp_ratio = role.get("fp", 0) / max(len(rows), 1)
    vids = Counter(r["video_id"] for r in rows)
    return {
        "n_rows": len(rows),
        "fp_row_ratio": round(fp_ratio, 4),
        "role_counts": dict(role),
        "n_videos": len(vids),
        "rows_per_video": quantiles(list(vids.values())),
        "score": quantiles(scores.tolist()),
        "prior_hits": quantiles(priors.tolist()),
    }


def fragmentation(rows: list[dict]) -> dict:
    """Distinct Q1 track ids per GT track id among matched rows."""
    by_gt = defaultdict(set)
    n_matched = 0
    for r in rows:
        if r["gt_track_id"] >= 0:
            by_gt[r["gt_track_id"]].add((r["video_id"], r["track_id"]))
            n_matched += 1
    frag = sorted(len(v) for v in by_gt.values())
    return {"n_gt_tracks_matched": len(by_gt), "n_matched_rows": n_matched,
            "tracklets_per_gt": quantiles(frag)}


def geometry(rows: list[dict], feats_path: Path, label: str) -> dict | None:
    if not feats_path.exists():
        return None
    arr = np.load(feats_path)["feats"]
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    known_idx = [i for i, r in enumerate(rows) if r["gt_role"] == "known"]
    novel_idx = [i for i, r in enumerate(rows) if r["gt_role"] in ("novel", "novel_role")]
    fp_idx = [i for i, r in enumerate(rows) if r["gt_role"] == "fp"]

    def mean_cos(sel, same_cat=True):
        if len(sel) < 2:
            return 0.0
        if same_cat:
            grp = defaultdict(list)
            for i in sel:
                grp[rows[i]["gt_category_id"]].append(i)
            vals = []
            for idxs in grp.values():
                if len(idxs) >= 2:
                    sub = arr[idxs]
                    sim = sub @ sub.T
                    tri = np.triu_indices(len(idxs), 1)
                    vals.extend(sim[tri].tolist())
            return float(np.mean(vals)) if vals else 0.0
        idx = np.random.default_rng(0).choice(sel, size=min(len(sel), 2000), replace=False)
        sim = arr[idx] @ arr[idx].T
        tri = np.triu_indices(len(idx), 1)
        return float(sim[tri].mean())

    def cross(sel_a, sel_b):
        if not sel_a or not sel_b:
            return 0.0
        ia = np.random.default_rng(0).choice(sel_a, size=min(len(sel_a), 2000), replace=False)
        ib = np.random.default_rng(0).choice(sel_b, size=min(len(sel_b), 2000), replace=False)
        sim = arr[ia] @ arr[ib].T
        return float(sim.mean())

    return {
        f"{label}_known_own_cos": round(mean_cos(known_idx, True), 4),
        f"{label}_known_pair_cos": round(mean_cos(known_idx, False), 4),
        f"{label}_novel_own_cos": round(mean_cos(novel_idx, True), 4),
        f"{label}_fp_pair_cos": round(mean_cos(fp_idx, False), 4),
        f"{label}_known_fp_cos": round(cross(known_idx, fp_idx), 4),
        f"{label}_known_novel_cos": round(cross(known_idx, novel_idx), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="outputs/iclr27_phase4t/train_stream/proposals.csv")
    ap.add_argument("--dev-csv", default="outputs/iclr27_phase4q/q1_long/proposals_dev.csv")
    ap.add_argument("--train-feats", default="outputs/iclr27_phase4t/train_stream/feats.npz")
    ap.add_argument("--dev-feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--out", default="outputs/iclr27_phase4t/stream_audit/stream_audit.json")
    args = ap.parse_args()

    train = load_stream(ROOT / args.train_csv)
    dev = load_stream(ROOT / args.dev_csv)
    out = {
        "train": {**row_stats(train), **tracklet_stats(train), "fragmentation": fragmentation(train)},
        "dev": {**row_stats(dev), **tracklet_stats(dev)},
        "train_geometry": geometry(train, ROOT / args.train_feats, "train"),
        "dev_geometry": geometry(dev, ROOT / args.dev_feats, "dev"),
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
