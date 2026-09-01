"""Raw DINOv2 -> frozen TSE -> frozen B representation survival audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase8a.model.adapter import CausalTrajectoryAdapter


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def norm(x):
    return x / (np.linalg.norm(x) + 1e-12)


def interval(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0, "mean": None, "low": None, "high": None, "std": None}
    m = float(vals.mean()); se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return {"n": int(len(vals)), "mean": m, "low": m - 1.96 * se, "high": m + 1.96 * se, "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}


def metrics(track_vecs, rep_name):
    # track_vecs: list of (key, video, category, vector)
    n = len(track_vecs)
    r1, r5, aps, proto = [], [], [], []
    by_cat = defaultdict(list); by_vid = defaultdict(list)
    for i, (key, vid, cat, v) in enumerate(track_vecs):
        cand = [(j, float(np.dot(v, track_vecs[j][3]))) for j in range(n)
                if j != i and track_vecs[j][1] != vid]
        pos = {j for j, _ in cand if track_vecs[j][2] == cat}
        if not pos:
            continue
        ranked = [j for j, _ in sorted(cand, key=lambda x: (-x[1], x[0]))]
        r1.append(float(ranked[0] in pos)); r5.append(float(bool(set(ranked[:5]) & pos)))
        hit = 0; ap = 0.0
        for rank, j in enumerate(ranked, 1):
            if j in pos:
                hit += 1; ap += hit / rank
        aps.append(ap / len(pos))
        by_cat[cat].append(float(ranked[0] in pos)); by_vid[vid].append(float(ranked[0] in pos))
        # leave-one-track-out category prototype (all categories with support)
        cls = defaultdict(list)
        for j, (_, vj, cj, wj) in enumerate(track_vecs):
            if j != i and vj != vid:
                cls[cj].append(wj)
        protos = {c: norm(np.mean(ws, axis=0)) for c, ws in cls.items() if ws}
        if protos:
            pred = max(protos, key=lambda c: float(np.dot(v, protos[c])))
            proto.append(float(pred == cat))
    same, diff = [], []
    for i, (_, vi, ci, x) in enumerate(track_vecs):
        for j in range(i + 1, n):
            _, vj, cj, y = track_vecs[j]
            if vi == vj:
                continue
            (same if ci == cj else diff).append(float(np.dot(x, y)))
    return {
        "representation": rep_name,
        "cross_video_r1": {"value": float(np.mean(r1)) if r1 else None, "queries": len(r1)},
        "cross_video_r5": {"value": float(np.mean(r5)) if r5 else None, "queries": len(r5)},
        "cross_video_map": {"value": float(np.mean(aps)) if aps else None, "queries": len(aps)},
        "leave_one_track_out_prototype": {"value": float(np.mean(proto)) if proto else None, "queries": len(proto)},
        "same_distance": {"value": float(np.mean(same)) if same else None, "pairs": len(same)},
        "different_distance": {"value": float(np.mean(diff)) if diff else None, "pairs": len(diff)},
        "distance_gap": float(np.mean(same) - np.mean(diff)) if same and diff else None,
        "category_macro_r1": interval([np.mean(x) for x in by_cat.values()]),
        "video_grouped_r1": interval([np.mean(x) for x in by_vid.values()]),
        "eligible_categories": len(by_cat), "eligible_videos": len(by_vid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="outputs/iclr27_phase14c/proposals/proposals_mixed.csv")
    ap.add_argument("--aligned", default="outputs/iclr27_phase14c/proposals/proposals_aligned.csv")
    ap.add_argument("--dinov2", default="outputs/iclr27_phase14c/features/proposal_dinov2.npz")
    ap.add_argument("--b-checkpoint", default="outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth")
    ap.add_argument("--out-features", default="outputs/iclr27_phase14c/features/proposal_tse_b.npz")
    ap.add_argument("--out", default="outputs/iclr27_phase14c/eval/representation_survival.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / args.proposals).open()))
    raw = np.load(ROOT / args.dinov2)["feats"].astype(np.float32)
    assert len(rows) == len(raw)
    device = torch.device(args.device)
    tse, _, _ = load_tse(device)
    z = project(device, tse, raw)
    ck = torch.load(ROOT / args.b_checkpoint, map_location=device, weights_only=False)
    cka = ck.get("args", {}); dim = int(cka.get("dim", 128)); frame_level = bool(cka.get("frame_level", False))
    adapter = CausalTrajectoryAdapter(dim=dim, rho_init=0.0, sigma2=1.0, frame_level=frame_level).to(device)
    adapter.load_state_dict(ck["adapter"]); adapter.eval()
    h = np.zeros((len(rows), dim), dtype=np.float32)
    chrono = sorted(range(len(rows)), key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]), int(rows[i]["proposal_local_id"]), int(rows[i]["track_id"])))
    states = {}
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            prev = states.get(key, adapter.new_state())
            out, nxt = adapter(torch.from_numpy(z[i]).to(device).unsqueeze(0), prev)
            h[i] = out[0].cpu().numpy(); states[key] = nxt.detach()
    out_feat = ROOT / args.out_features; out_feat.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_feat.with_suffix(out_feat.suffix + ".tmp")
    np.savez_compressed(tmp, tse=z.astype(np.float32), b_h=h.astype(np.float32),
                        row_keys=np.asarray([f"{r['video_id']}:{r['frame_id']}:{r['proposal_local_id']}:{r['track_id']}" for r in rows]))
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp; os.replace(generated, out_feat)

    # Evaluator-only labels are read after all model embeddings have been computed.
    aligned = list(csv.DictReader((ROOT / args.aligned).open()))
    assert len(aligned) == len(rows)
    roles = {}
    for r in aligned:
        if int(r["gt_track_id"]) >= 0 and r["gt_role"] == "novel":
            roles[(int(r["video_id"]), int(r["track_id"]))] = (int(r["video_id"]), int(r["gt_category_id"]))
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        key = (int(r["video_id"]), int(r["track_id"]))
        if key in roles: by_track[key].append(i)
    tracks = []
    for key, idxs in sorted(by_track.items()):
        idxs.sort(key=lambda i: (int(rows[i]["frame_id"]), int(rows[i]["proposal_local_id"])))
        vid, cat = roles[key]
        tracks.append((key, vid, cat, idxs))
    prefixes = [1, 2, 4, 8, 16]
    result = {"protocol": "phase14c", "aligned_novel_tracks": len(tracks), "representations": {}, "raw_signal_survives_proposals": None,
              "feature_flags": {"q1_label_used": False, "private_gt_used_for_features": False, "future_frames_used": False, "physical_id_used_as_feature": False}}
    for p in prefixes:
        for name, arr in (("raw_dinov2", raw), ("tse_128", z), ("phase8a_b_causal", h)):
            tv = []
            for key, vid, cat, idxs in tracks:
                ii = idxs[:min(p, len(idxs))]
                vec = arr[ii[-1]] if name == "phase8a_b_causal" else norm(arr[ii].mean(axis=0))
                tv.append((key, vid, cat, vec.astype(np.float32)))
            result["representations"][f"{name}_prefix{p}"] = metrics(tv, name)
    r16 = result["representations"]["raw_dinov2_prefix16"]["cross_video_r1"]["value"]
    result["raw_signal_survives_proposals"] = bool(r16 is not None and r16 > 0.0)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True)); os.replace(tmp, out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
