"""Replay ORBIT-D1 with per-track decision logging (read-only audit)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.action_router import ACTIONS
from src.orbit.bi_memory import BiMemory, stats_to_tensor
from src.orbit.evaluate import load_model, embed_track, build_known
from src.orbit.protocol import (
    load_frame_features, load_train_labels, load_stream, load_gt, meta_classes,
)


def replay_with_log(model, rows, feats, known_protos, radii, gt_by_sid, device):
    mem = BiMemory(known_protos, radii, novel_update_rate=0.2)
    logs = []
    seen_classes = set()
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        z, rel = embed_track(model, feats[sid], device)
        kid, ks = mem.known_id(z)
        nid, ns = mem.existing_novel(z)
        stats = mem.stats(z, rel, len(feats[sid]), known_id=kid, novel_id=nid)
        st = stats_to_tensor(stats, device=device)
        with torch.no_grad():
            logits = model.action_net(st)
        action = int(logits.argmax(dim=1).item())
        created = False
        if action == 0 and kid is not None:
            out_kind = "known"; out_id = int(kid)
        elif action == 1 and nid is not None:
            out_kind = "novel"; out_id = int(nid)
            mem.update_novel(nid, z)
        else:
            out_kind = "novel"; out_id = mem.create_novel(z); created = True
        g = gt_by_sid.get(sid)
        true_role = g.get("protocol_role", "?") if g else "?"
        true_class = g.get("ground_truth_category_id", "?") if g else "?"
        is_first = g and true_class not in seen_classes
        if g:
            seen_classes.add(true_class)
        logs.append({
            "sample_id": sid, "track_id": sid,
            "true_role": true_role, "true_class": true_class,
            "arrival_index": i, "first_occurrence": is_first,
            "predicted_action": ACTIONS[action], "predicted_known_id": kid,
            "predicted_virtual_novel_id": out_id if out_kind == "novel" else None,
            "best_known_similarity": stats[0], "second_known_similarity": stats[1],
            "known_margin": stats[2], "known_radius_norm": stats[3],
            "best_novel_similarity": stats[4], "second_novel_similarity": stats[5],
            "novel_margin": stats[6], "novel_radius_norm": stats[7],
            "track_length": len(feats[sid]), "feature_reliability": rel,
            "active_novel_prototypes": len(mem.novel),
            "created_new_prototype": created,
        })
    return logs


def write_csv(path, logs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        w.writeheader()
        w.writerows(logs)


def main():
    device = "cuda"
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    val_feats = load_frame_features("gt_tracks_mean")
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    protos, radii = build_known(model, train_feats, train_labels, set(train_labels.values()), device)
    gt = load_gt("pure")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    rows = load_stream("pure", "main_seed1027")
    logs = replay_with_log(model, rows, val_feats, protos, radii, gt_by_sid, device)
    write_csv(ROOT / "outputs/orbit_bc/audit/per_track_decisions_val_seed1027.csv", logs)
    # meta-dev proxy
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    protos_dev, radii_dev = build_known(model, train_feats, train_labels, meta_tr, device)
    dev_ids = [sid for sid, c in train_labels.items() if c in meta_dev and sid in train_feats]
    dev_gt = {sid: {"protocol_role": "novel", "ground_truth_category_id": train_labels[sid]}
              for sid in dev_ids}
    dev_rows = [{"sample_id": sid, "stream_order": i} for i, sid in enumerate(sorted(dev_ids))]
    logs_dev = replay_with_log(model, dev_rows, train_feats, protos_dev, radii_dev, dev_gt, device)
    write_csv(ROOT / "outputs/orbit_bc/audit/per_track_decisions_meta_dev.csv", logs_dev)
    print("val logs", len(logs), "meta_dev logs", len(logs_dev))


if __name__ == "__main__":
    main()
