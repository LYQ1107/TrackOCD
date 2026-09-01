"""Phase 4L Root Cause A: semantic admissibility dataset.

Builds detection-level and causal tracklet-prefix-level feature rows from
the frozen J1b replay (tau=0.30, M0, lambda_s=0.1) plus tracker-native
inputs that exist at frame time (detector score/bbox/mask, appearance
embedding, DINO embedding, association outcome, semantic scores).
GT is attached only as an offline diagnostic label (IoU >= 0.5).

Outputs:
  outputs/iclr27_phase4l/audit/admissibility_detection_features.csv
  outputs/iclr27_phase4l/audit/admissibility_tracklet_features.csv
  outputs/iclr27_phase4l/audit/persistent_fp_analysis.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
LOG_ROOT = ROOT / "outputs" / "iclr27_phase4j" / "semantic_logs" / "J1b"
DEC_ROOT = ROOT / "outputs" / "iclr27_phase4k" / "audit" / "prov_j1b"
EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
PRE_ASSOC = EXPORT / "pre_assoc_detections"
PKG = EXPORT / "replay_packages"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / \
    "detection_features"
TAO_JSON = EXPORT / "tao_subset" / "validation_20.json"
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
OUT = ROOT / "outputs" / "iclr27_phase4l" / "audit"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt():
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        b = ann["bbox"]
        cat = int(ann["category_id"])
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": cat,
            "role": "known" if cat in known else "novel",
        })
    return out


def match_gt(gt, image_id, bbox):
    best, bi = None, 0.5
    for g in gt.get(image_id, []):
        v = iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def frame_index_map(vid):
    frames = []
    seen = set()
    for line in (PRE_ASSOC / f"{vid}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["frame_order"] in seen:
            continue
        seen.add(r["frame_order"])
        frames.append(r["frame_order"])
    frames.sort()
    return {f: i for i, f in enumerate(frames)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt()

    # Frozen semantic backend is loaded only to recompute the diagnostic
    # known-margin (not stored in Phase 4J logs). It is never modified.
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    device = torch.device("cuda:0")
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    sem = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0")

    rows = []
    for p in sorted(LOG_ROOT.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    decisions = {}
    dec_path = DEC_ROOT / "association_decisions_j1b.jsonl"
    for line in dec_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        decisions[(int(r["video_id"]), int(r["frame_id"]),
                   int(r["raw_det_idx"]))] = r

    # pre-computed per-row appearance / mask features, grouped by video
    track_groups = defaultdict(list)
    det_csv_path = OUT / "admissibility_detection_features.csv"
    if det_csv_path.exists() and det_csv_path.stat().st_size > 1000:
        with open(det_csv_path) as f:
            det_rows = list(csv.DictReader(f))
        print("resume: detection features loaded",
              len(det_rows), flush=True)
        for r in rows:
            track_groups[(int(r["video_id"]),
                          int(r["physical_track_id"]))].append(r)
    else:
        det_rows = []
        by_video = defaultdict(list)
        for r in rows:
            by_video[int(r["video_id"])].append(r)

        for vid, vrows in by_video.items():
            fimap = frame_index_map(vid)
            feats = np.load(FEAT_ROOT / str(vid) / "feats.npz")
            dino = feats["feats"].astype(np.float32)
            dino_ids = feats["det_local_ids"].tolist()
            dino_idx = {int(i): k for k, i in enumerate(dino_ids)}
            frame_cache = {}
            # batch-embed all unique DINO features of this video once
            need_zi = sorted({dino_idx[int(r["det_idx"])] for r in vrows
                              if int(r["det_idx"]) in dino_idx})
            z_cache = {}
            for s in range(0, len(need_zi), 512):
                chunk_idx = need_zi[s:s + 512]
                x = torch.as_tensor(dino[chunk_idx], dtype=torch.float32,
                                    device=device).view(-1, 1, 768)
                mask = torch.ones(x.shape[0], 1, dtype=torch.bool,
                                  device=device)
                with torch.no_grad():
                    out = model.aggregate(x, mask)
                zs = out["z"]
                if zs.dim() == 3:
                    zs = zs[:, 0]
                zs = zs.float().cpu().numpy()
                for k, zi in enumerate(chunk_idx):
                    z_cache[dino[zi].tobytes()] = zs[k]
            for r in sorted(vrows,
                            key=lambda x: (x["frame_id"], x["det_idx"])):
                fid, did = int(r["frame_id"]), int(r["det_idx"])
                fi = fimap.get(fid)
                if fi is None:
                    continue
                npz = frame_cache.get(fi)
                if npz is None:
                    npz = np.load(PKG / str(vid) / f"frame_{fi:06d}.npz")
                    frame_cache[fi] = npz
                bbox = r["bbox"]
                det_gt = match_gt(gt, int(r["image_id"]), bbox)
                det = decisions.get((vid, fid, did))
                tf = npz["track_feats"][did] if did < len(
                    npz["track_feats"]) else np.zeros(256, dtype=np.float32)
                mask = npz["det_masks"][did] if did < len(
                    npz["det_masks"]) else np.zeros((1, 160, 288),
                                                    dtype=np.float32)
                mask_area = float((1.0 / (1.0 + np.exp(-mask)) > 0.5).mean())
                db = npz["det_bboxes"][did] if did < len(
                    npz["det_bboxes"]) else np.zeros(5, dtype=np.float32)
                bbox_norm_area = float(max(db[2], 0.0) * max(db[3], 0.0))
                mask_bbox_ratio = mask_area / bbox_norm_area \
                    if bbox_norm_area > 1e-6 else 0.0
                zi = dino_idx.get(did)
                dino_norm = float(np.linalg.norm(dino[zi])) \
                    if zi is not None else 0.0
                known_margin = 0.0
                z = None
                if zi is not None:
                    z = z_cache.get(dino[zi].tobytes())
                if z is not None:
                    ks = sem.P_known @ z
                    order = np.argsort(ks)[::-1]
                    known_margin = float(ks[order[0]] - ks[order[1]]) \
                        if len(ks) >= 2 else 0.0
                w, h = r["bbox"][2], r["bbox"][3]
                det_rows.append({
                    "video_id": vid, "frame_id": fid, "det_idx": did,
                    "image_id": int(r["image_id"]),
                    "physical_track_id": r["physical_track_id"],
                    "det_score": float(r["score"]),
                    "bbox_area": float(w * h),
                    "bbox_aspect": float(w / h) if h > 0 else 0.0,
                    "mask_area_frac": round(mask_area, 5),
                    "mask_bbox_ratio": round(mask_bbox_ratio, 5),
                    "appearance_norm": round(float(np.linalg.norm(tf)), 4),
                    "dino_norm": round(dino_norm, 4),
                    "p_known": float(r["p_known"]),
                    "best_known": float(r["best_known"]),
                    "known_margin": round(known_margin, 4),
                    "best_novel": float(r["best_novel"]),
                    "novel_id": r["novel_id"],
                    "global_novel_id": r["global_novel_id"],
                    "semantic_action": r["semantic_action"],
                    "commit_state": r["commit_state"],
                    "track_age": int(r["track_age"]),
                    "novel_support": int(r["novel_support"]),
                    "assoc_ap_score": float(det["appearance_best_score"])
                    if det else "",
                    "assoc_fn_score": float(det["final_best_score"])
                    if det else "",
                    "assoc_sem_delta": float(det["sem_delta_chosen"])
                    if det else "",
                    "assoc_assigned": int(det["assigned_id"] >= 0)
                    if det else "",
                    "gt_role": (det_gt or {}).get("role", "fp"),
                    "gt_category": (det_gt or {}).get("category_id", ""),
                    "gt_track": (det_gt or {}).get("track_id", ""),
                })
                track_groups[(vid, int(r["physical_track_id"]))].append(r)

    # ---- write detection CSV immediately (resume-safe) ----
    if det_rows and not det_csv_path.exists():
        fields = list(det_rows[0].keys())
        with open(det_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(det_rows)
        print("wrote detection features", len(det_rows), flush=True)

    # ---- tracklet-prefix aggregation ----
    km_map = {(int(r["video_id"]), int(r["frame_id"]), int(r["det_idx"])):
              float(r["known_margin"]) for r in det_rows}
    track_rows = []
    for (vid, tid), seq in track_groups.items():
        seq = sorted(seq, key=lambda x: (x["frame_id"], x["det_idx"]))
        frames = [int(r["frame_id"]) for r in seq]
        scores = [float(r["score"]) for r in seq]
        pks = [float(r["p_known"]) for r in seq]
        bks = [float(r["best_known"]) for r in seq]
        kms = [km_map.get((vid, int(r["frame_id"]), int(r["det_idx"])),
                          0.0) for r in seq]
        # appearance prefix consistency (consecutive cosine)
        app_cos = []
        for prev, cur in zip(seq[:-1], seq[1:]):
            fi_p = frame_index_map(vid).get(int(prev["frame_id"]))
            fi_c = frame_index_map(vid).get(int(cur["frame_id"]))
            if fi_p is None or fi_c is None:
                continue
            a = np.load(PKG / str(vid) / f"frame_{fi_p:06d}.npz")[
                "track_feats"][int(prev["det_idx"])].astype(np.float32)
            b = np.load(PKG / str(vid) / f"frame_{fi_c:06d}.npz")[
                "track_feats"][int(cur["det_idx"])].astype(np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                app_cos.append(float((a @ b) / (na * nb)))
        # bbox / scale / aspect consistency
        ious, scale_ch, asp_ch = [], [], []
        for prev, cur in zip(seq[:-1], seq[1:]):
            pa, ca = prev["bbox"], cur["bbox"]
            ious.append(iou([pa[0], pa[1], pa[0] + pa[2], pa[1] + pa[3]],
                            [ca[0], ca[1], ca[0] + ca[2], ca[1] + ca[3]]))
            prev_area = max(pa[2] * pa[3], 1.0)
            cur_area = max(ca[2] * ca[3], 1.0)
            scale_ch.append(float(cur_area / prev_area))
            asp_ch.append(abs(float(ca[2] / max(ca[3], 1)) -
                              float(pa[2] / max(pa[3], 1))))
        gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
        max_consec = 1
        run = 1
        for g in gaps:
            run = run + 1 if g == 1 else 1
            max_consec = max(max_consec, run)
        sems = [r.get("semantic_id") for r in seq]
        sem_switches = sum(1 for a, b in zip(sems[:-1], sems[1:]) if a != b)
        global_ids = [r.get("global_novel_id") for r in seq]
        gid_switches = sum(1 for a, b in zip(global_ids[:-1], global_ids[1:])
                           if a != b)
        assoc_aps = [float(decisions[(vid, int(r["frame_id"]),
                                      int(r["det_idx"]))]
                           ["appearance_best_score"])
                     for r in seq
                     if (vid, int(r["frame_id"]), int(r["det_idx"]))
                     in decisions]
        roles = []
        for r in seq:
            g = match_gt(gt, int(r["image_id"]), r["bbox"])
            roles.append(g["role"] if g else "fp")
        tp_frac = sum(1 for x in roles if x != "fp") / max(len(roles), 1)
        role = "tp" if tp_frac >= 0.5 else "fp"
        gt_tracks = set()
        for r in seq:
            g = match_gt(gt, int(r["image_id"]), r["bbox"])
            if g is not None:
                gt_tracks.add(g["track_id"])
        track_rows.append({
            "video_id": vid, "physical_track_id": tid,
            "length": len(seq), "max_age": int(seq[-1]["track_age"]),
            "first_frame": frames[0], "last_frame": frames[-1],
            "max_gap": max(gaps) if gaps else 0,
            "consecutive_max": max_consec,
            "mean_det_score": round(float(np.mean(scores)), 4),
            "max_det_score": round(float(np.max(scores)), 4),
            "mean_p_known": round(float(np.mean(pks)), 4),
            "std_p_known": round(float(np.std(pks)), 4),
            "mean_best_known": round(float(np.mean(bks)), 4),
            "mean_known_margin": round(float(np.mean(kms)), 4),
            "appearance_prefix_cos": round(float(np.mean(app_cos)), 4)
            if app_cos else "",
            "appearance_prefix_min_cos": round(float(np.min(app_cos)), 4)
            if app_cos else "",
            "mean_bbox_iou": round(float(np.mean(ious)), 4) if ious else "",
            "mean_scale_change": round(float(np.mean(scale_ch)), 4)
            if scale_ch else "",
            "mean_aspect_change": round(float(np.mean(asp_ch)), 4)
            if asp_ch else "",
            "semantic_switch_rate": round(
                sem_switches / max(len(sems) - 1, 1), 4),
            "gid_switch_rate": round(
                gid_switches / max(len(global_ids) - 1, 1), 4),
            "mean_assoc_ap": round(float(np.mean(assoc_aps)), 4)
            if assoc_aps else "",
            "tp_fraction": round(tp_frac, 4),
            "role": role,
            "gt_tracks": "|".join(str(x) for x in sorted(gt_tracks)),
        })

    # ---- write CSVs ----
    if det_rows:
        fields = list(det_rows[0].keys())
        with open(OUT / "admissibility_detection_features.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(det_rows)
    if track_rows:
        fields = list(track_rows[0].keys())
        with open(OUT / "admissibility_tracklet_features.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(track_rows)

    # ---- persistent FP buckets ----
    buckets = [(1, 1, "len1"), (2, 2, "len2"), (3, 5, "len3_5"),
               (6, 10, "len6_10"), (11, 20, "len11_20"), (21, 10 ** 9,
                                                          "len21plus")]
    pers_rows = []
    for lo, hi, name in buckets:
        sub = [r for r in track_rows if lo <= int(r["length"]) <= hi]
        fps = [r for r in sub if r["role"] == "fp"]
        tps = [r for r in sub if r["role"] == "tp"]
        pers_rows.append({
            "bucket": name, "n": len(sub), "fp": len(fps), "tp": len(tps),
            "fp_rate": round(len(fps) / max(len(sub), 1), 4),
            "mean_det_score_fp": round(float(np.mean(
                [r["mean_det_score"] for r in fps])), 4) if fps else "",
            "mean_p_known_fp": round(float(np.mean(
                [r["mean_p_known"] for r in fps])), 4) if fps else "",
            "mean_appearance_cos_fp": round(float(np.mean(
                [r["appearance_prefix_cos"] for r in fps
                 if r["appearance_prefix_cos"] != ""])), 4)
            if any(r["appearance_prefix_cos"] != "" for r in fps) else "",
            "mean_bbox_iou_fp": round(float(np.mean(
                [r["mean_bbox_iou"] for r in fps
                 if r["mean_bbox_iou"] != ""])), 4)
            if any(r["mean_bbox_iou"] != "" for r in fps) else "",
            "semantic_switch_rate_fp": round(float(np.mean(
                [r["semantic_switch_rate"] for r in fps])), 4)
            if fps else "",
            "mean_det_score_tp": round(float(np.mean(
                [r["mean_det_score"] for r in tps])), 4) if tps else "",
            "mean_p_known_tp": round(float(np.mean(
                [r["mean_p_known"] for r in tps])), 4) if tps else "",
            "mean_appearance_cos_tp": round(float(np.mean(
                [r["appearance_prefix_cos"] for r in tps
                 if r["appearance_prefix_cos"] != ""])), 4)
            if any(r["appearance_prefix_cos"] != "" for r in tps) else "",
            "mean_bbox_iou_tp": round(float(np.mean(
                [r["mean_bbox_iou"] for r in tps
                 if r["mean_bbox_iou"] != ""])), 4)
            if any(r["mean_bbox_iou"] != "" for r in tps) else "",
            "semantic_switch_rate_tp": round(float(np.mean(
                [r["semantic_switch_rate"] for r in tps])), 4)
            if tps else "",
        })
    with open(OUT / "persistent_fp_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pers_rows[0].keys()))
        w.writeheader()
        w.writerows(pers_rows)
    print("ADMISSIBILITY_DATASET_DONE",
          len(det_rows), "detections", len(track_rows), "tracklets")


if __name__ == "__main__":
    main()
