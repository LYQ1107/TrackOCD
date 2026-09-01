"""Phase 4J fragment semantic continuity (extends Phase 4I audit).

For each GT track, collect the predicted physical IDs that cover it
(IoU >= 0.5 per frame).  If a GT track is fragmented into >1 physical
track, report whether the fragments keep the same *effective* semantic
identity and the same *global* novel semantic identity.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_sem(log_root):
    out = {}
    if log_root is None:
        return out
    for p in log_root.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["image_id"], tuple(r["bbox"]))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--sem-log-root", type=Path, default=None)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    tao = json.loads(TAO_JSON.read_text())
    img_by_id = {im["id"]: im for im in tao["images"]}
    gt = defaultdict(list)
    for ann in tao["annotations"]:
        b = ann["bbox"]
        gt[(int(ann["video_id"]), int(ann["track_id"]))].append({
            "image_id": ann["image_id"],
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "frame_index": img_by_id[ann["image_id"]]["frame_index"],
        })
    preds = defaultdict(list)
    for p in args.pred_dir.glob("*.json"):
        for r in json.loads(p.read_text()):
            b = r["bbox"]
            preds[r["image_id"]].append({
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                "track_id": int(r["track_id"]),
            })
    sem = load_sem(args.sem_log_root)

    rows = []
    for (vid, tid), anns in gt.items():
        anns = sorted(anns, key=lambda a: a["frame_index"])
        cover = defaultdict(list)
        for a in anns:
            for p in preds.get(a["image_id"], []):
                if iou(p["bbox"], a["bbox"]) >= 0.5:
                    cover[p["track_id"]].append(a)
        n_frag = len(cover)
        row = {"video_id": vid, "gt_track_id": tid,
               "gt_frames": len(anns), "pred_ids": n_frag,
               "fragmented": int(n_frag > 1)}
        if n_frag > 1 and args.sem_log_root is not None:
            sem_ids = set()
            global_ids = set()
            for tid_pred, alist in cover.items():
                for a in alist:
                    best = None
                    bi = 0.5
                    for (img, _b), r in sem.items():
                        if img == a["image_id"]:
                            v = iou(r["bbox"], a["bbox"])
                            if v >= bi:
                                bi, best = v, r
                    if best:
                        sem_ids.add(best.get("semantic_id"))
                        gid = best.get("global_novel_id")
                        if gid is not None:
                            global_ids.add("N" + str(gid))
            row["semantic_ids_across_fragments"] = len(sem_ids)
            row["semantic_consistent"] = int(len(sem_ids) == 1)
            row["global_novel_ids_across_fragments"] = len(global_ids)
            row["global_novel_consistent"] = int(len(global_ids) <= 1)
        rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    frag = [r for r in rows if r["fragmented"]]
    print("gt tracks", len(rows), "fragmented", len(frag),
          "share", len(frag) / max(len(rows), 1))


if __name__ == "__main__":
    main()
