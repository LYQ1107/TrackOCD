"""Build known semantic anchors in Stage C TSR space + geometry audit.

Semantic observation = frozen Stage C TSR final state (256-d, unit norm).
Known anchors are class-mean states over real Q1 train tracklets; the
audit reports within/between cosine stats to choose the component family.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4t.stream_data import build_tracklets
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.trajectory.model import TSR


def load_rows():
    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    return rows, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows, feats = load_rows()
    tracklets = build_tracklets(rows)
    row_index = {id(r): i for i, r in enumerate(rows)}

    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=args.device)
    sd = ck["model"]
    rep_sd = {k[len("rep."):]: v for k, v in sd.items() if k.startswith("rep.")}
    tsr = TSR(arch="gru").to(args.device)
    tsr.load_state_dict(rep_sd)
    tsr.eval()

    cat_list = sorted(known_ids())
    cat_index = {c: i for i, c in enumerate(cat_list)}
    states_by_cat = defaultdict(list)
    phys_state = {}  # (cat, video, gt_track_id) -> mean state over tracklets
    with torch.no_grad():
        for key, tl in tracklets.items():
            c = tl["gt_category_id"]
            if tl["role"] != "known" or c not in cat_index:
                continue
            idx = [row_index[id(r)] for r in tl["rows"]]
            z = np.stack([feats[i] for i in idx]).astype(np.float32)
            z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
            q = np.stack([r["q_phys"] for r in tl["rows"]]).astype(np.float32)
            states = tsr.embed_sequence(torch.from_numpy(z).to(args.device),
                                        torch.from_numpy(q).to(args.device))
            s = states[-1].cpu().numpy().astype(np.float32)
            states_by_cat[c].append(s)
            pid = (c, int(tl["rows"][-1]["video_id"]),
                   int(tl["rows"][-1]["gt_track_id"]))
            phys_state.setdefault(pid, []).append(s)

    means = np.zeros((len(cat_list), 256), dtype=np.float32)
    kappas = np.zeros(len(cat_list), dtype=np.float32)
    counts = np.zeros(len(cat_list), dtype=np.int64)
    within = np.zeros(len(cat_list), dtype=np.float64)
    for c in cat_list:
        S = np.stack(states_by_cat[c]) if states_by_cat[c] else np.zeros((0, 256))
        counts[cat_index[c]] = len(S)
        if len(S) == 0:
            continue
        mu = S.mean(0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        means[cat_index[c]] = mu
        cos = S @ mu
        within[cat_index[c]] = float(cos.mean())
        R = float(np.linalg.norm(S.mean(0)))
        d = 255
        kappas[cat_index[c]] = float(R * (d + 1 - R * R) / max(1 - R * R, 1e-9))
    # between-category cosine stats (train-known)
    cos_all = []
    for i in range(len(cat_list)):
        for j in range(i + 1, len(cat_list)):
            if counts[i] and counts[j]:
                cos_all.append(float(means[i] @ means[j]))
    cos_all = np.asarray(cos_all)
    # cross-physical within-category cosine (different GT track ids)
    cross_phys = []
    same_phys = []
    phys_means = {}
    for pid, ss in phys_state.items():
        mu = np.mean(np.stack(ss), axis=0)
        phys_means[pid] = mu / (np.linalg.norm(mu) + 1e-12)
    pid_list = list(phys_means)
    for i in range(len(pid_list)):
        ci, vi, gi = pid_list[i]
        for j in range(i + 1, len(pid_list)):
            cj, vj, gj = pid_list[j]
            if ci != cj:
                continue
            cos = float(phys_means[pid_list[i]] @ phys_means[pid_list[j]])
            if gi != gj:
                cross_phys.append(cos)
            else:
                same_phys.append(cos)
    out = ROOT / "outputs/iclr27_phase4x/simple_mixture"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "known_anchors.npz",
                        means=means, kappas=kappas.astype(np.float32),
                        cat_ids=np.asarray(cat_list, dtype=np.int64))
    audit = {
        "n_categories": len(cat_list),
        "support_counts": {str(c): int(counts[cat_index[c]]) for c in cat_list},
        "within_cos_mean": round(float(np.mean(within[counts > 0])), 4),
        "within_cos_p10": round(float(np.percentile(within[counts > 0], 10)), 4),
        "within_cos_p90": round(float(np.percentile(within[counts > 0], 90)), 4),
        "between_cos_mean": round(float(cos_all.mean()), 4),
        "between_cos_p10": round(float(np.percentile(cos_all, 10)), 4),
        "between_cos_p90": round(float(np.percentile(cos_all, 90)), 4),
        "cross_physical_within_cos_mean": round(
            float(np.mean(cross_phys)) if cross_phys else None, 4),
        "cross_physical_within_cos_p10": round(
            float(np.percentile(cross_phys, 10)), 4) if cross_phys else None,
        "cross_physical_within_cos_p90": round(
            float(np.percentile(cross_phys, 90)), 4) if cross_phys else None,
        "same_gt_fragment_cos_mean": round(
            float(np.mean(same_phys)) if same_phys else None, 4),
        "n_cross_physical_pairs": len(cross_phys),
        "kappa_mean": round(float(np.mean(kappas[counts > 0])), 2),
        "kappa_min": round(float(np.min(kappas[counts > 0])), 2),
        "kappa_max": round(float(np.max(kappas[counts > 0])), 2),
        "cos_separation": round(float(np.mean(within[counts > 0]) - cos_all.mean()), 4),
        "embedding_norm": "unit-normalized by TSR construction",
    }
    (out / "geometry_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
