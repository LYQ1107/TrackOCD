"""Real tracker-induced stream: convert OVTR tao_track.json to per-row
proposals with GT matching, physical cues, and DINO features."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.protocol import box_iou

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRAIN_JSON = ROOT / "data" / "raw" / "tao" / "annotations" / "train.json"
KNOWN = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()))


def load_train_gt() -> tuple[dict, dict[int, dict]]:
    d = json.loads(TRAIN_JSON.read_text())
    img = {im["id"]: im for im in d["images"]}
    gt_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        gt_by_img[int(a["image_id"])].append({
            "bbox": [a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]],
            "category_id": int(a["category_id"]),
            "track_id": int(a.get("track_id", -1)),
        })
    return img, dict(gt_by_img)


def convert_stream(track_json: Path, out_csv: Path) -> list[dict]:
    """tao_track.json rows -> per-frame proposals with gt_role/gt_category and
    causal physical cues (score, prior_hits, age, gap, running score mean)."""
    rows = json.loads(track_json.read_text())
    img, gt_by_img = load_train_gt()
    vid_of = {im["id"]: int(im["video_id"]) for im in img.values()}
    fr_of = {im["id"]: int(im["frame_index"]) for im in img.values()}
    out = []
    by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in rows:
        iid = int(r["image_id"])
        if iid not in img:
            continue
        b = [float(v) for v in r["bbox"]]
        bb = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        best, role, cat, gt_track = 0.5, "fp", -1, -1
        for g in gt_by_img.get(iid, []):
            v = box_iou(bb, g["bbox"])
            if v >= best:
                best = v
                cat = g["category_id"]
                gt_track = g["track_id"]
                role = "known" if cat in KNOWN else "novel_role"
        out.append({
            "video_id": vid_of[iid], "frame_id": fr_of[iid], "image_id": iid,
            "track_id": int(r["track_id"]), "score": float(r["score"]),
            "category_id": int(r.get("category_id", -1)),
            "bbox_xyxy": bb, "gt_role": role, "gt_iou": best,
            "gt_category_id": cat, "gt_track_id": gt_track,
        })
        by_track[(vid_of[iid], int(r["track_id"]))].append(len(out) - 1)
    out.sort(key=lambda r: (r["video_id"], r["frame_id"], r["track_id"]))
    # causal physical cues per track (prefix-only)
    seen: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"last_frame": None, "hits": 0, "score_sum": 0.0, "n": 0})
    for r in sorted(out, key=lambda r: (r["video_id"], r["frame_id"], r["track_id"])):
        key = (r["video_id"], r["track_id"])
        st = seen[key]
        hits = st["hits"]
        gap = 0 if st["last_frame"] is None else r["frame_id"] - st["last_frame"] - 1
        age = hits  # 0-based prefix age
        r["prior_hits"] = hits
        r["age"] = age
        r["gap"] = gap
        r["run_score_mean"] = st["score_sum"] / st["n"] if st["n"] else r["score"]
        r["q_phys"] = json.dumps([
            r["score"], float(np.log1p(hits)), min(age, 16) / 16.0,
            float(np.log1p(max(gap, 0))), r["run_score_mean"],
            float(np.log(max(bb[2] - bb[0], 1) * max(bb[3] - bb[1], 1)) / 12.0)])
        r["bbox_xyxy"] = json.dumps(r["bbox_xyxy"])
        st["last_frame"] = r["frame_id"]
        st["hits"] = hits + 1
        st["score_sum"] += r["score"]
        st["n"] += 1
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def build_tracklets(rows: list[dict]):
    """Group proposal rows into physical tracklets (Q1 track_id), keeping
    frame order; returns tracklets + per-tracklet GT identity (majority
    gt_category among matched rows; role known/novel_role/fp)."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["video_id"], r["track_id"])].append(r)
    tracklets = {}
    for key, rs in by_key.items():
        rs.sort(key=lambda r: (r["frame_id"], r["track_id"]))
        roles = [r["gt_role"] for r in rs]
        n_match = sum(1 for x in roles if x != "fp")
        if n_match == 0:
            gt_cat, role = -1, "fp"
        else:
            cats = [r["gt_category_id"] for r in rs if r["gt_role"] != "fp"]
            from collections import Counter
            gt_cat = Counter(cats).most_common(1)[0][0]
            role = "known" if gt_cat in KNOWN else "novel_role"
        tracklets[key] = {"rows": rs, "role": role, "gt_category_id": gt_cat,
                          "length": len(rs)}
    return tracklets


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    rows = convert_stream(Path(args.track_json), Path(args.out_csv))
    print("rows", len(rows))
    from collections import Counter
    print(dict(Counter(r["gt_role"] for r in rows)))


if __name__ == "__main__":
    main()
