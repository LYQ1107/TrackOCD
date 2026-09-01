"""Shared replay utilities for Phase 4C error decomposition audits.

Replays TrackOCD-Ref, ORBIT-D1 (joint) and ORBIT-BC (birth-controlled) with
per-track logging of actions, logits, memory statistics, prototype support
and both original-DINO and adapted known similarities.  The replay is causal:
tracks are processed one at a time in the frozen stream order and historical
outputs are never modified.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.action_router import ACTIONS, KNOWN, EXISTING_NOVEL, NEW_NOVEL
from src.orbit.bi_memory import BiMemory, stats_to_tensor
from src.orbit.evaluate import load_model, build_known
from src.orbit.protocol import (
    load_frame_features,
    load_mean_features,
    load_train_labels,
    load_stream,
    load_gt,
    meta_classes,
)
from src.orbit_bc.batch_orbit import embed_batch
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def frozen_known_protos(class_ids):
    """Original-DINO-space known prototypes built from train-known track means."""
    feats = load_mean_features("train_known_mean")
    labels = load_train_labels()
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if c in class_ids and sid in feats:
            sums[c] += feats[sid]
            counts[c] += 1
    protos = {}
    for c, s in sums.items():
        v = s / counts[c]
        protos[int(c)] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def replay_one(z, rel, track_len, mem, model, device, mode, birth_threshold):
    """Decision for a single track; returns (log, memory mutations done)."""
    kid, ks = mem.known_id(z)
    nid, ns = mem.existing_novel(z)
    stats = mem.stats(z, rel, track_len, known_id=kid, novel_id=nid)
    st = stats_to_tensor(stats, device=device)
    with torch.no_grad():
        logits = model.action_net(st) if model is not None else None
    if logits is not None:
        logit_known, logit_existing, logit_new = [float(x) for x in logits[0]]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_known, prob_existing, prob_new = [float(x) for x in probs]
        action = int(logits.argmax(dim=1).item())
    else:
        # Ref-style sequential threshold policy.
        logit_known = logit_existing = logit_new = float("nan")
        prob_known = prob_existing = prob_new = float("nan")
        thr = 0.45
        if kid is not None and ks >= thr:
            action = KNOWN
        elif nid is not None and ns >= thr:
            action = EXISTING_NOVEL
        else:
            action = NEW_NOVEL
    if mode == "bc" and action == NEW_NOVEL and nid is not None and ns >= birth_threshold:
        action = EXISTING_NOVEL
    created = False
    support = 0
    age = -1
    if action == KNOWN and kid is not None:
        out_kind = "known"
        out_id = int(kid)
    elif action == EXISTING_NOVEL and nid is not None:
        out_kind = "novel"
        out_id = int(nid)
        support = mem.novel_counts.get(nid, 0)
        age = mem.novel[nid].get("created_at", -1)
        mem.update_novel(nid, z)
    else:
        out_kind = "novel"
        out_id = mem.create_novel(z)
        mem.novel[out_id]["created_at"] = 0  # overwritten below with stream index
        created = True
    return {
        "action": action,
        "out_kind": out_kind,
        "out_id": out_id,
        "kid": kid,
        "ks": ks,
        "nid": nid,
        "ns": ns,
        "stats": stats,
        "logit_known": logit_known,
        "logit_existing": logit_existing,
        "logit_new": logit_new,
        "prob_known": prob_known,
        "prob_existing": prob_existing,
        "prob_new": prob_new,
        "created": created,
        "support": support,
        "age": age,
        "mem_size_after": len(mem.novel),
    }


def replay_all(rows, feats, sids, zs, rels, known_protos, radii, model, device,
               mode="joint", birth_threshold=0.0, novel_update_rate=0.2):
    """Replay a full stream; returns list of log dicts."""
    mem = BiMemory(known_protos, radii, novel_update_rate=novel_update_rate)
    logs = []
    seen_classes = set()
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        z = zs[i]
        rel = float(rels[i])
        d = replay_one(z, rel, len(feats[sid]), mem, model, device, mode, birth_threshold)
        g = r.get("_gt")
        true_role = g.get("protocol_role", "?") if g else "?"
        true_class = g.get("ground_truth_category_id", "?") if g else "?"
        is_first = g is not None and true_class not in seen_classes
        if g is not None:
            seen_classes.add(true_class)
        if d["created"]:
            mem.novel[d["out_id"]]["created_at"] = i
        s = d["stats"]
        logs.append({
            "sample_id": sid,
            "track_id": sid,
            "true_role": true_role,
            "true_class": true_class,
            "arrival_index": i,
            "first_occurrence": is_first,
            "predicted_action": ACTIONS[d["action"]],
            "predicted_known_id": d["kid"],
            "predicted_virtual_novel_id": d["out_id"] if d["out_kind"] == "novel" else None,
            "best_known_similarity": s[0],
            "second_known_similarity": s[1],
            "known_margin": s[2],
            "known_radius_norm": s[3],
            "best_novel_similarity": s[4],
            "second_novel_similarity": s[5],
            "novel_margin": s[6],
            "novel_radius_norm": s[7],
            "track_length": len(feats[sid]),
            "feature_reliability": rel,
            "active_novel_prototypes": d["mem_size_after"] - (1 if d["created"] else 0),
            "prototype_support": d["support"],
            "prototype_age": d["age"],
            "created_new_prototype": d["created"],
            "logit_known": d["logit_known"],
            "logit_existing": d["logit_existing"],
            "logit_new": d["logit_new"],
            "prob_known": d["prob_known"],
            "prob_existing": d["prob_existing"],
            "prob_new": d["prob_new"],
            "known_score_before": r.get("_known_before", float("nan")),
            "known_score_after_frozen": r.get("_known_after_frozen", float("nan")),
            "known_score_after_adapted": r.get("_known_after_adapted", float("nan")),
            "domain": r.get("_domain", "?"),
            "video_id": r.get("_video_id", "?"),
            "class_frequency": r.get("_class_freq", -1),
        })
    return logs


def decorate_rows(rows, feats, mean_feats, zs, frozen_protos, adapted_protos, labels):
    """Attach per-track metadata used by the audits (domains, freq, scores)."""
    P_frozen = np.stack([frozen_protos[c] for c in sorted(frozen_protos)])
    frozen_ids = sorted(frozen_protos)
    P_adapted = np.stack([adapted_protos[c] for c in sorted(adapted_protos)])
    adapted_ids = sorted(adapted_protos)
    freq = defaultdict(int)
    for c in labels.values():
        freq[int(c)] += 1
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        path = r.get("image_paths") or []
        r["_domain"] = path[0].split("/")[1] if path else "?"
        r["_video_id"] = r.get("video_id", "?")
        r["_class_freq"] = -1
        if sid in mean_feats:
            z0 = mean_feats[sid]
            sb = float(np.max(z0 @ P_frozen.T))
            r["_known_before"] = sb
            if i < len(zs):
                sa = float(np.max(zs[i] @ P_frozen.T))
                sad = float(np.max(zs[i] @ P_adapted.T))
            else:
                sa = sad = float("nan")
            r["_known_after_frozen"] = sa
            r["_known_after_adapted"] = sad
        g = r.get("_gt")
        if g and g["protocol_role"] in ("supported_known", "zero_shot_known"):
            r["_class_freq"] = freq.get(int(g["ground_truth_category_id"]), 0)
    return rows


def attach_gt(rows, gt_by_sid):
    for r in rows:
        r["_gt"] = gt_by_sid.get(r["sample_id"])
    return rows


def emit_preds(logs):
    preds = []
    for l in logs:
        if l["predicted_action"] == "KNOWN" and l["predicted_known_id"] is not None:
            preds.append({
                "sample_id": l["sample_id"],
                "stream_order": l["arrival_index"],
                "prediction_type": "known",
                "semantic_category_id": int(l["predicted_known_id"]),
            })
        elif l["predicted_virtual_novel_id"] is not None:
            preds.append({
                "sample_id": l["sample_id"],
                "stream_order": l["arrival_index"],
                "prediction_type": "novel",
                "virtual_category_id": int(l["predicted_virtual_novel_id"]),
            })
        else:
            preds.append({
                "sample_id": l["sample_id"],
                "stream_order": l["arrival_index"],
                "prediction_type": "unresolved",
            })
    return preds


def assignment_from_preds(preds, gt_rows):
    ev = TrackOCDEvaluator(gt_rows)
    res = ev.evaluate(preds)
    return res, ev


def build_meta_proxy_rows(max_known=600, seed=1027):
    """Deterministic train-side proxy: meta-dev classes play novel, a sampled
    subset of meta-train classes play known. Alternating order."""
    train_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    dev_ids = sorted(sid for sid, c in labels.items() if c in meta_dev and sid in train_feats)
    known_ids = sorted(sid for sid, c in labels.items() if c in meta_tr and sid in train_feats)
    rng = np.random.RandomState(seed)
    rng.shuffle(known_ids)
    known_ids = known_ids[:max_known]
    gt_by_sid = {}
    rows = []
    n = max(len(dev_ids), len(known_ids))
    for i in range(n):
        if i < len(dev_ids):
            sid = dev_ids[i]
            rows.append({"sample_id": sid})
            gt_by_sid[sid] = {"sample_id": sid,
                              "ground_truth_category_id": labels[sid],
                              "protocol_role": "novel"}
        if i < len(known_ids):
            sid = known_ids[i]
            rows.append({"sample_id": sid})
            gt_by_sid[sid] = {"sample_id": sid,
                              "ground_truth_category_id": labels[sid],
                              "protocol_role": "supported_known"}
    return rows, gt_by_sid, train_feats, labels
