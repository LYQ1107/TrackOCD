"""Post-lock, label-read-once DEV+ aligned pair audit for Phase15A."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import csv
import numpy as np
import torch

from src.iclr27_phase15.representation.phase15a_probe import (
    RelationMLP,
    l2_one,
    pair_metrics,
    pair_table_from_matrix,
    retrieval_from_matrix,
    score_matrix,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    rows = [dict(r) for r in csv.DictReader(
        (ROOT / "outputs/iclr27_phase14c/proposals/proposals_mixed.csv").open())]
    aligned = [dict(r) for r in csv.DictReader(
        (ROOT / "outputs/iclr27_phase14c/proposals/proposals_aligned.csv").open())]
    feats = np.load(ROOT / "outputs/iclr27_phase14c/features/proposal_dinov2.npz",
                    allow_pickle=False)["feats"].astype(np.float32)
    by_track = defaultdict(list); info = {}
    for i, r in enumerate(aligned):
        if int(r["gt_track_id"]) < 0 or r["gt_role"] != "novel":
            continue
        key = (int(r["video_id"]), int(r["track_id"]))
        by_track[key].append(i)
        info[key] = (int(r["video_id"]), int(r["gt_category_id"]))
    keys = sorted(by_track)
    prefixes = [1, 2, 4, 8, 16]
    device = torch.device("cpu")
    models = {}
    for seed in [20260824, 20260825]:
        m = RelationMLP(768).to(device)
        ck = torch.load(ROOT / f"outputs/iclr27_phase15/checkpoints/relation_seed{seed}.pth",
                        map_location=device)
        m.load_state_dict(ck["state_dict"]); m.eval(); models[f"relation_seed{seed}"] = m
    result = {"protocol": "phase15a_devplus_offline_audit", "novel_aligned_tracks": len(keys),
              "cross_video_same_category_pairs": None, "representations": {},
              "labels_used_only_for_posthoc_metrics": True,
              "q1_label_used": False, "future_frames_used_for_method": False,
              "physical_id_used_as_feature": False}
    for p in prefixes:
        vec = []
        for key in keys:
            idx = sorted(by_track[key], key=lambda i: (
                int(rows[i]["frame_id"]), int(rows[i]["proposal_local_id"])))[:p]
            vec.append(l2_one(feats[idx].mean(axis=0)))
        vec = np.asarray(vec, dtype=np.float32)
        labs = np.asarray([info[k][1] for k in keys], dtype=np.int64)
        vids = np.asarray([info[k][0] for k in keys], dtype=np.int64)
        raw = vec @ vec.T
        cv, cy = pair_table_from_matrix(raw, labs, vids, True)
        item = {"raw_cosine": {"pair_cross_video": pair_metrics(np.clip((cv + 1) / 2, 0, 1), cy),
                               "retrieval": retrieval_from_matrix(vec, labs, vids, raw)}}
        for name, model in models.items():
            mat = score_matrix(model, vec, device, raw=False)
            s, y = pair_table_from_matrix(mat, labs, vids, True)
            item[name] = {"pair_cross_video": pair_metrics(s, y),
                          "retrieval": retrieval_from_matrix(vec, labs, vids, mat)}
        result["representations"][f"prefix{p}"] = item
    result["cross_video_same_category_pairs"] = int(
        sum(1 for i in range(len(keys)) for j in range(i + 1, len(keys))
            if info[keys[i]][0] != info[keys[j]][0] and info[keys[i]][1] == info[keys[j]][1]))
    atomic(ROOT / "outputs/iclr27_phase15/eval/phase15a_devplus_offline_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
