"""Phase 4U instance-level data loaders.

Three legal sources:
  real      - Q1 tracker-induced TRAIN stream, known rows grouped by
              (video_id, gt_track_id) physical GT instance.
  episodic  - Phase 4S public train-known universe (2,196 physical tracks),
              non-tracker-induced supplement.
  dev       - frozen Q1 dev proposals grouped by predicted tracklet
              (video_id, track_id); only tracklets whose majority matched
              role is a supported-known category. Used for geometry only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
REAL_CSV = ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "proposals.csv"
REAL_FEATS = ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "feats.npz"
DEV_CSV = ROOT / "outputs" / "iclr27_phase4q" / "q1_long" / "proposals_dev.csv"
DEV_FEATS = ROOT / "outputs" / "iclr27_phase4s" / "q1_features" / "feats.npz"
EPISODIC_TRACKS = ROOT / "data" / "trackocd_v1" / "pure" / "public" / "train_known_tracks.jsonl"
EPISODIC_FEAT_DIR = ROOT / "data" / "caches" / "features" / "dinov2" / "train_known_mean"
SUPPORTED_KNOWN = json.loads(
    (ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text())


def norm_rows(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)


def load_real_stream() -> dict:
    rows = list(csv.DictReader(open(REAL_CSV)))
    feats = np.load(REAL_FEATS)["feats"]
    by_inst: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["gt_role"] != "known":
            continue
        key = (int(r["video_id"]), int(r["gt_track_id"]))
        by_inst[key].append((i, r))
    instances = []
    for (video, gt_track), items in by_inst.items():
        items.sort(key=lambda p: (int(p[1]["frame_id"]), int(p[1]["track_id"])))
        idx = [p[0] for p in items]
        f = norm_rows(feats[idx])
        q = np.stack([json.loads(r["q_phys"]) for _, r in items]).astype(np.float32)
        cat = int(items[0][1]["gt_category_id"])
        instances.append({
            "id": f"{video}_{gt_track}", "video": video, "cat": cat,
            "feats": f, "q": q, "n_rows": len(items),
        })
    by_cat = defaultdict(list)
    for inst in instances:
        by_cat[inst["cat"]].append(inst["id"])
    return {"name": "real", "instances": instances, "by_cat": dict(by_cat)}


def load_episodic() -> dict:
    tracks = [json.loads(line) for line in open(EPISODIC_TRACKS) if line.strip()]
    instances = []
    for t in tracks:
        p = EPISODIC_FEAT_DIR / f"{t['sample_id']}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        f = norm_rows(np.asarray(d["frame_embeddings"], dtype=np.float32))
        instances.append({
            "id": t["sample_id"], "video": int(t["video_id"]), "cat": int(t["category_id"]),
            "feats": f, "q": None, "n_rows": len(f),
        })
    by_cat = defaultdict(list)
    for inst in instances:
        by_cat[inst["cat"]].append(inst["id"])
    return {"name": "episodic", "instances": instances, "by_cat": dict(by_cat)}


def _dev_qphys(rows_by_track: dict) -> dict[int, list[float]]:
    out = {}
    for key, items in rows_by_track.items():
        items.sort(key=lambda r: (int(r["frame_id"]), int(r.get("proposal_local_id") or 0)))
        last_frame, hits, ssum, n = None, 0, 0.0, 0
        for r in items:
            gap = 0 if last_frame is None else int(r["frame_id"]) - last_frame - 1
            b = json.loads(r["bbox_xyxy"])
            area = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
            out[id(r)] = [
                float(r["score"]), float(np.log1p(hits)), min(hits, 16) / 16.0,
                float(np.log1p(max(gap, 0))),
                ssum / n if n else float(r["score"]),
                float(np.log(area) / 12.0),
            ]
            last_frame = int(r["frame_id"])
            hits += 1
            ssum += float(r["score"])
            n += 1
    return out


def load_dev() -> dict:
    rows = list(csv.DictReader(open(DEV_CSV)))
    feats = np.load(DEV_FEATS)["feats"]
    by_track: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(int(r["video_id"]), int(r["track_id"]))].append((i, r))
    qmap = _dev_qphys(
        {k: [p[1] for p in v] for k, v in by_track.items()})
    instances = []
    for (video, track), items in by_track.items():
        items.sort(key=lambda p: (int(p[1]["frame_id"]), int(p[1].get("proposal_local_id") or 0)))
        cats = Counter(int(r["gt_category_id"]) for _, r in items if r["gt_role"] == "known")
        if not cats:
            continue
        cat = cats.most_common(1)[0][0]
        if cat not in SUPPORTED_KNOWN:
            continue
        idx = [p[0] for p in items]
        f = norm_rows(feats[idx])
        q = np.stack([qmap[id(r)] for _, r in items]).astype(np.float32)
        instances.append({
            "id": f"{video}_{track}", "video": video, "cat": cat,
            "feats": f, "q": q, "n_rows": len(items),
        })
    by_cat = defaultdict(list)
    for inst in instances:
        by_cat[inst["cat"]].append(inst["id"])
    return {"name": "dev", "instances": instances, "by_cat": dict(by_cat)}


def load_source(name: str) -> dict:
    if name == "real":
        return load_real_stream()
    if name == "episodic":
        return load_episodic()
    if name == "dev":
        return load_dev()
    raise ValueError(name)


def class_sets() -> tuple[set[int], set[int]]:
    def read(name: str) -> set[int]:
        with open(ROOT / "outputs" / "orbit" / "splits" / name) as f:
            return {int(r["class_id"]) for r in csv.DictReader(f)}
    return read("meta_train_classes.csv"), read("meta_dev_classes.csv")
