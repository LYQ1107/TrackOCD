"""Phase 4H permutation experiment (P0-P4) with frozen ORBIT-MDC M2.

Replays the frozen M2 model on re-orderings of the official Pure Full
seed1027 stream.  Track features, GT roles, class identities and model
weights never change; only the arrival order changes.

  P0: original order
  P1: global random track order (5 seeds)
  P2: class-block random order of novel classes, known positions fixed
  P3: hardness-reversed order (hardest novel classes first; diagnostic)
  P4: easy-first order (diagnostic)

Diagnostic only; permutations are never used as a training/eval protocol.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_mean_features,
    load_stream,
    load_train_labels,
)
from src.orbit_msr.evaluate import embed_many, mechanism_rates
from src.orbit_msr.protocol import known_stats
from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_mdc.model import ORBITMDCModel


def load_mdc_4h(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    sd = ck["state_dict"]
    reuse_dim = int(sd["reuse.net.0.weight"].shape[1])
    model = ORBITMDCModel(
        dim=768, bottleneck=ck.get("bottleneck", 128),
        gate_dim=ck.get("gate_dim", 11), reuse_dim=reuse_dim, hidden=64,
        use_adapter=True, compat_dim=ck.get("compat_dim", 6),
        birth_dim=ck.get("birth_dim", 0))
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ck


def prepare_official():
    gt = load_gt("pure")
    rows = load_stream("pure", "main_seed1027")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    seen = set()
    for r in rows:
        g = gt_by_sid[r["sample_id"]]
        r["role"] = ("known" if g["protocol_role"] in
                     ("supported_known", "zero_shot_known") else "novel")
        r["class"] = g["ground_truth_category_id"]
        r["first_occurrence"] = r["class"] not in seen
        r["domain"] = r["image_paths"][0].split("/")[1]
        seen.add(r["class"])
    feats = {sid: f[:8] for sid, f in
             load_frame_features("gt_tracks_mean").items()}
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    labels = load_train_labels()
    return rows, gt, feats, train_feats, labels


def raw_hardness(rows, feats, labels):
    """Raw-DINO class hardness = mean best-known cosine per true class."""
    mean_feats = load_mean_features("train_known_mean")
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if sid in mean_feats:
            sums[c] += mean_feats[sid]
            counts[c] += 1
    P = np.stack([sums[c] / counts[c] for c in sorted(sums)]).astype(np.float32)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    class_ids = sorted(sums)
    hardness = defaultdict(float)
    cnt = defaultdict(int)
    for r in rows:
        if r["role"] != "novel":
            continue
        f = feats[r["sample_id"]]
        z = f.mean(axis=0)
        z = z / (np.linalg.norm(z) + 1e-12)
        hardness[r["class"]] += float(np.max(P @ z))
        cnt[r["class"]] += 1
    return {c: hardness[c] / cnt[c] for c in cnt}, class_ids


def novel_subsequence_permuted(rows, mode, seed=None, hardness=None):
    """Reinsert a permuted novel subsequence into original known positions.

    Known rows keep their original positions.  The novel subsequence is
    reordered: P2 random class blocks (seeded), P3 hardest-first,
    P4 easiest-first.  Within-class temporal order is preserved.
    """
    known = [r for r in rows if r["role"] == "known"]
    novel = [r for r in rows if r["role"] == "novel"]
    by_class = defaultdict(list)
    for r in novel:
        by_class[r["class"]].append(r)
    blocks = list(by_class.items())
    if mode == "P2":
        rng = random.Random(seed)
        rng.shuffle(blocks)
    elif mode == "P3":
        blocks = sorted(blocks, key=lambda kv: -hardness[kv[0]])
    elif mode == "P4":
        blocks = sorted(blocks, key=lambda kv: hardness[kv[0]])
    new_novel = [r for _, rs in blocks for r in rs]
    out = []
    ni = 0
    for r in rows:
        if r["role"] == "known":
            out.append(r)
        else:
            out.append(new_novel[ni])
            ni += 1
    assert ni == len(new_novel)
    return out


def replay_order(model, ck, rows, zs, rels, protos, radii, known_ids, device,
                 gate_thr=0.5, compat_thr=0.45, compat_margin=0.05,
                 snapshot_cb=None, track_lens=None):
    """Causal replay of frozen M2 on an arbitrary order (no embedding recompute)."""
    P_known = np.stack([protos[c] for c in known_ids]).astype(np.float32)
    feat_names = [f.strip() for f in ck.get(
        "compat_feats", "sim,margin,radius,support,conf,mem,rel").split(",")
        if f.strip()]
    mem = IamMemory(protos, radii,
                    novel_update_rate=ck.get("novel_update_rate", 0.2))
    logs = []
    for i, r in enumerate(rows):
        if snapshot_cb is not None:
            snapshot_cb(i, len(mem.novel), mem)
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        margin_k = 0.0
        if ks.shape[0] >= 2:
            order = np.argsort(ks)[::-1]
            margin_k = float(ks[order[0]] - ks[order[1]])
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty(
            (0, 768), dtype=np.float32)
        nid = None
        best_n = second_n = -1.0
        margin_n = 0.0
        dist_n = 1.0
        if P_novel.shape[0]:
            ns = P_novel @ z
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            nid = int(sorted(mem.novel)[int(order[0])])
            r_n = mem.novel_radii.get(nid, 0.3)
            dist_n = (1.0 - best_n) / max(r_n, 1e-6)
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=(track_lens.get(r["sample_id"], 8)
                                    if track_lens else 8),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        gate_prob = float(torch.sigmoid(torch.as_tensor(gate_logit)))
        if gate_prob >= gate_thr and kid is not None:
            logs.append({"sample_id": r["sample_id"], "class": r["class"],
                         "role": r["role"], "arrival_index": i,
                         "first_occurrence": r["first_occurrence"],
                         "predicted_action": "KNOWN",
                         "predicted_known_id": kid,
                         "predicted_virtual_novel_id": None,
                         "memory_size": len(mem.novel),
                         "gate_prob": gate_prob, "best_known_sim": best_k,
                         "known_margin": margin_k, "best_novel_sim": best_n,
                         "novel_margin": margin_n})
            continue
        q_best = -1.0
        q_second = -1.0
        states = {v: mem.state(v) for v in sorted(mem.novel)}
        if P_novel.shape[0]:
            X_rows = []
            for v in sorted(mem.novel):
                st = states[v]
                X_rows.append(build_compat_features(
                    z, mem.novel[v]["proto"], st["radius"], st["support"],
                    st["conf"], len(mem.novel), rel, margin_n, feat_names))
            X = torch.as_tensor(np.asarray(X_rows, dtype=np.float32),
                                device=device)
            with torch.no_grad():
                q = torch.sigmoid(model.compat_forward(X)).cpu().numpy()
            if q.shape[0]:
                qorder = np.argsort(q)[::-1]
                q_best = float(q[qorder[0]])
                q_second = float(q[qorder[1]]) if q.shape[0] >= 2 else -1.0
                nid = int(sorted(mem.novel)[int(qorder[0])])
        reuse_ok = (q_best >= compat_thr
                    and (len(mem.novel) < 2
                         or q_best - q_second >= compat_margin))
        if reuse_ok and nid is not None:
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            logs.append({"sample_id": r["sample_id"], "class": r["class"],
                         "role": r["role"], "arrival_index": i,
                         "first_occurrence": r["first_occurrence"],
                         "predicted_action": "EXISTING_NOVEL",
                         "predicted_known_id": None,
                         "predicted_virtual_novel_id": nid,
                         "memory_size": len(mem.novel),
                         "gate_prob": gate_prob, "best_known_sim": best_k,
                         "known_margin": margin_k, "best_novel_sim": best_n,
                         "novel_margin": margin_n})
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append({"sample_id": r["sample_id"], "class": r["class"],
                         "role": r["role"], "arrival_index": i,
                         "first_occurrence": r["first_occurrence"],
                         "predicted_action": "NEW_NOVEL",
                         "predicted_known_id": None,
                         "predicted_virtual_novel_id": vid,
                         "memory_size": len(mem.novel),
                         "gate_prob": gate_prob, "best_known_sim": best_k,
                         "known_margin": margin_k, "best_novel_sim": best_n,
                         "novel_margin": margin_n})
    return logs, mem


def metrics(logs, gt):
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    aug = []
    for l in logs:
        l2 = dict(l)
        l2["true_role"] = ("supported_known" if l["role"] == "known"
                           else "novel")
        l2["true_class"] = l["class"]
        aug.append(l2)
    mr = mechanism_rates(aug, res["hungarian_assignment"])
    novel = [l for l in logs if l["role"] == "novel"]
    return {
        "all_acc": res["all_track_acc"],
        "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"],
        "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "predicted_novel_count": res["predicted_novel_count"],
        "known_to_novel": mr["known_to_novel"],
        "novel_to_known": mr["novel_to_known"],
        "repeated_false_birth": mr["repeated_false_birth"],
        "wrong_existing": mr["wrong_existing"],
        "first_merge": mr["first_merge"],
        "final_memory_size": max((l["memory_size"] for l in logs), default=0),
        "mean_memory_size": float(np.mean([l["memory_size"] for l in logs])),
    }


def class_stats(logs, hardness):
    by_class = defaultdict(list)
    for l in logs:
        by_class[l["class"]].append(l)
    rows = []
    for c, ls in by_class.items():
        role = ls[0]["role"]
        row = {"class": c, "role": role, "count": len(ls),
               "n2k_rate": (sum(1 for l in ls if l["predicted_action"] == "KNOWN")
                            / len(ls) if role == "novel" else ""),
               "k2n_rate": (sum(1 for l in ls if l["predicted_action"] != "KNOWN")
                            / len(ls) if role == "known" else ""),
               "first_arrival": min(l["arrival_index"] for l in ls),
               "mean_arrival": float(np.mean([l["arrival_index"] for l in ls])),
               "hardness": hardness.get(c, "")}
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_orders(rows, hardness, seeds):
    orders = [("P0", 0, rows)]
    for s in seeds:
        rng = random.Random(s)
        p1 = list(rows)
        rng.shuffle(p1)
        orders.append(("P1", s, p1))
        orders.append(("P2", s, novel_subsequence_permuted(rows, "P2", s)))
    orders.append(("P3", 0, novel_subsequence_permuted(rows, "P3",
                                                       hardness=hardness)))
    orders.append(("P4", 0, novel_subsequence_permuted(rows, "P4",
                                                       hardness=hardness)))
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:8")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()
    device = args.device
    rows, gt, feats, train_feats, labels = prepare_official()
    hardness, _ = raw_hardness(rows, feats, labels)
    print("hardness computed for", len(hardness), "novel classes", flush=True)
    model, ck = load_mdc_4h("runs/orbit_mdc/mdc_m2/model.pth", device)
    from src.orbit.evaluate import build_known
    known_classes = sorted(set(labels.values()))
    protos, radii = build_known(model, train_feats, labels,
                                set(known_classes), device)
    known_ids = sorted(protos)
    print("known protos", len(known_ids), flush=True)
    sids = [r["sample_id"] for r in rows]
    zs, rels = embed_many(model, feats, sids, device)
    print("embedded", len(zs), flush=True)
    track_lens = {sid: len(f) for sid, f in feats.items()}
    seeds = [1001, 2002, 3003, 4004, 5005]
    orders = build_orders(rows, hardness, seeds)
    print("orders", len(orders), flush=True)
    out_dir = ROOT / "outputs/iclr27_phase4h/audit"
    log_dir = out_dir / "permutation_track_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    res_rows = []
    cls_rows = []
    for order_idx, (mode, seed, order) in enumerate(orders):
        tag = f"{mode}_{seed}" if seed else mode
        if order_idx < args.start or order_idx >= args.end:
            continue
        logs, mem = replay_order(model, ck, order, zs, rels, protos, radii,
                                 known_ids, device, track_lens=track_lens)
        m = metrics(logs, gt)
        row = {"mode": mode, "seed": seed, **m}
        res_rows.append(row)
        for cr in class_stats(logs, hardness):
            cr.update({"mode": mode, "seed": seed})
            cls_rows.append(cr)
        write_csv(log_dir / f"{tag}{args.out_tag}.csv", logs)
        print(tag, "rn", round(m["rn_acc"], 4), "n2k", round(m["novel_to_known"], 4),
              "ari", round(m["ari"], 4), "mem", m["final_memory_size"],
              flush=True)
    write_csv(out_dir / f"permutation_results{args.out_tag}.csv", res_rows)
    write_csv(out_dir / f"permutation_class_results{args.out_tag}.csv", cls_rows)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
