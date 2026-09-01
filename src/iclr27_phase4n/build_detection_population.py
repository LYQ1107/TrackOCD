"""Phase 4N frozen-stream detection population.

Joins, for every detection entering the tracker:
  - detector output (score, bbox, mask fraction, appearance norm),
  - semantic observation (p_known, best_known, best_novel, gate logit,
    M2 z norm, DINO norm, known margin),
  - association outcome (appearance/final best score, assigned track),
  - offline GT role (VALID_KNOWN / VALID_NOVEL / FP).

GT is used only for offline labeling; nothing here enters the online
loop.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
MODEL_PTH = ROOT / "runs" / "orbit_mdc" / "mdc_m2" / "model.pth"

MODES = {
    "dev": {
        "export": ROOT / "outputs" / "iclr27_phase3a" / "smoke",
        "feat": ROOT / "outputs" / "iclr27_phase4i" / "audit" /
        "detection_features",
        "zcache": ROOT / "outputs" / "iclr27_phase4m" / "audit" /
        "det_z_cache",
        "semlogs": ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev",
        "prov": ROOT / "outputs" / "iclr27_phase4m" / "prov" / "dev",
        "tao": ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
        "tao_subset" / "validation_20.json",
    },
    "heldout": {
        "export": ROOT / "outputs" / "iclr27_phase4l" / "heldout_export",
        "feat": ROOT / "outputs" / "iclr27_phase4l" / "heldout_features",
        "zcache": ROOT / "outputs" / "iclr27_phase4m" / "audit" /
        "det_z_cache_heldout",
        "semlogs": ROOT / "outputs" / "iclr27_phase4m" / "runs" /
        "heldout",
        "prov": ROOT / "outputs" / "iclr27_phase4m" / "prov" / "heldout",
        "tao": ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
        "validation_heldout_tao.json",
    },
}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt(tao_json):
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(tao_json.read_text())
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


def load_p_known(model, device):
    """Known prototype matrix via the exact Phase 4M semantic manager."""
    from src.frame_online_trackocd.semantic import build_semantic_manager
    mgr = build_semantic_manager(model, device, prefix_mode="P1",
                                 decision_threshold=0.30)
    return mgr.P_known, mgr.known_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--tag", default="m3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tao", type=Path, default=None)
    ap.add_argument("--z-cache", type=Path, default=None)
    args = ap.parse_args()
    mode = MODES[args.mode]
    device = torch.device("cuda")
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    model, _ = load_mdc_model(str(MODEL_PTH), device)
    model.eval()
    P_known, known_ids = load_p_known(model, device)
    P_known = P_known.astype(np.float32)
    gt = load_gt(args.tao if args.tao is not None else mode["tao"])
    sem_rows = {}
    for p in sorted((mode["semlogs"] / args.tag /
                     "semantic_logs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sem_rows[(int(r["video_id"]), int(r["frame_id"]),
                      int(r["det_idx"]))] = r
    assoc_rows = {}
    apath = mode["prov"] / f"{args.tag}" / \
        f"association_decisions_{args.tag}.jsonl"
    if apath.exists():
        for line in apath.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            assoc_rows[(int(r["video_id"]), int(r["frame_id"]),
                        int(r["raw_det_idx"]))] = r

    videos = sorted(int(p.stem)
                    for p in (mode["export"] /
                              "pre_assoc_detections").glob("*.jsonl"))
    out_rows = []
    for vi, vid in enumerate(videos):
        pre = []
        for line in (mode["export"] / "pre_assoc_detections" /
                     f"{vid}.jsonl").read_text().splitlines():
            r = json.loads(line)
            pre.append(r)
        feats = np.load(mode["feat"] / str(vid) / "feats.npz")
        dino_ids = feats["det_local_ids"].tolist()
        dino_frames = feats["frame_orders"].tolist()
        dino_norm = {(int(i), int(f)): float(
            np.linalg.norm(feats["feats"][k]))
            for k, (i, f) in enumerate(zip(dino_ids, dino_frames))}
        zc_path = args.z_cache if args.z_cache is not None \
            else mode["zcache"]
        zc = np.load(zc_path / f"{vid}.npz")
        z_ids = zc["det_local_ids"].tolist()
        z_frames = zc["frame_orders"].tolist() \
            if "frame_orders" in zc.files else dino_frames
        z_norm = {(int(i), int(f)): float(np.linalg.norm(zc["z"][k]))
                  for k, (i, f) in enumerate(zip(z_ids, z_frames))}
        zvec = {(int(i), int(f)): zc["z"][k].astype(np.float32)
                for k, (i, f) in enumerate(zip(z_ids, z_frames))}
        pkg_dir = mode["export"] / "replay_packages" / str(vid)
        max_frame = max(r["frame_order"] for r in pre)
        mask_app = {}
        for fi in range(max_frame + 1):
            z = np.load(pkg_dir / f"frame_{fi:06d}.npz")
            idx = z["indices"].tolist()
            masks = torch.as_tensor(z["det_masks"]).float()
            mask_frac = (torch.sigmoid(masks) > 0.5).float().mean(
                dim=(1, 2, 3)).cpu().numpy()
            app_norm = torch.norm(torch.as_tensor(z["track_feats"]),
                                  dim=1).cpu().numpy()
            for i, did in enumerate(idx):
                mask_app[(int(did), fi)] = (
                    float(mask_frac[i]), float(app_norm[i]))
        for r in pre:
            did = int(r["det_local_id"])
            fid = int(r["frame_order"])
            image_id = int(r["image_id"])
            bbox = [float(v) for v in r["bbox_xyxy_original"]]
            g = match_gt(gt, image_id, bbox)
            role = (g or {}).get("role", "fp")
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h
            aspect = w / max(h, 1e-6)
            srow = sem_rows.get((vid, fid, did))
            arow = assoc_rows.get((vid, fid, did))
            z = zvec.get((did, fid))
            kscore = -1.0
            if z is not None and len(P_known):
                ks = P_known @ z
                order = np.argsort(ks)[::-1]
                kscore = float(ks[order[0]] - ks[order[1]]) \
                    if len(order) >= 2 else float(ks[order[0]])
            p_known = float(srow["p_known"]) if srow else ""
            logit = ""
            if isinstance(p_known, float):
                logit = round(log(max(min(p_known, 1 - 1e-9), 1e-9) /
                                  (1 - max(min(p_known, 1 - 1e-9), 1e-9))),
                              4)
            ma = mask_app.get((did, fid), ("", ""))
            out_rows.append({
                "mode": args.mode, "video_id": vid, "frame_id": fid,
                "image_id": image_id, "det_local_id": did,
                "score": float(r["score"]),
                "bbox_area": round(area, 2),
                "bbox_aspect": round(aspect, 4),
                "mask_frac": ma[0], "app_norm": ma[1],
                "dino_norm": dino_norm.get((did, fid), ""),
                "z_norm": z_norm.get((did, fid), ""),
                "known_margin": round(kscore, 4) if kscore >= -0.5 else "",
                "p_known": p_known, "gate_logit": logit,
                "best_known": float(srow["best_known"]) if srow else "",
                "best_novel": float(srow["best_novel"]) if srow else "",
                "track_id": srow.get("physical_track_id") if srow else "",
                "track_age": srow.get("track_age") if srow else "",
                "semantic_action": srow.get("semantic_action") if srow
                else "",
                "resolution_state": srow.get("resolution_state") if srow
                else "",
                "global_novel_id": srow.get("global_novel_id") if srow
                else "",
                "assoc_appearance_best": (
                    float(arow["appearance_best_score"]) if arow else ""),
                "assoc_final_best": (
                    float(arow["final_best_score"]) if arow else ""),
                "assoc_assigned_id": arow.get("assigned_id") if arow
                else "",
                "assoc_sem_delta": (
                    float(arow["sem_delta_final"]) if arow else ""),
                "gt_role": role,
                "gt_category": (g or {}).get("category_id", ""),
                "gt_track_id": (g or {}).get("track_id", ""),
            })
        print("population", args.mode, vid, "rows",
              len(out_rows), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    tmp.replace(args.out)
    print("DETECTION_POPULATION_DONE", args.mode, len(out_rows))


if __name__ == "__main__":
    main()
