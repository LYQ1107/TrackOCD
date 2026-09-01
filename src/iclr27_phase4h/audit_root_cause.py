"""Phase 4H root-cause audit: compositional hardness vs memory effect.

Components:
  1. frozen representation hardness (raw DINO + M2-adapted) per bucket/class;
  2. arrival-order permutation replays of frozen ORBIT-MDC M2;
  3. counterfactual memory replay (same query, different memory snapshots);
  4. root-cause logistic probe across permutations.

All GT usage is offline audit only; official data is used only for
diagnostics, never for training or tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_mean_features,
    load_stream,
    load_train_labels,
)
from src.orbit_msr.evaluate import embed_many, load_msr_model
from src.orbit_iam.evaluate_iam import load_iam_model


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


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
        seen.add(r["class"])
    feats = {sid: f[:8] for sid, f in
             load_frame_features("gt_tracks_mean").items()}
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    return rows, gt, feats, train_feats


def raw_known_scores(feats, sids):
    mean_feats = load_mean_features("train_known_mean")
    labels = load_train_labels()
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if sid in mean_feats:
            sums[c] += mean_feats[sid]
            counts[c] += 1
    protos = {}
    for c, s in sums.items():
        protos[int(c)] = _norm(s / counts[c])
    P = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    ids = sorted(protos)
    scores = {}
    for sid in sids:
        z = np.mean(feats[sid], axis=0)
        z = _norm(z)
        scores[sid] = P @ z
    return scores, ids, P


def hardness_rows(stream_rows, feats, adapted_scores, raw_scores, ids):
    rows = []
    for idx, r in enumerate(stream_rows):
        if r["role"] != "novel":
            continue
        sid = r["sample_id"]
        raw = raw_scores[sid]
        adapted = adapted_scores[sid]
        def stats(s):
            order = np.argsort(s)[::-1]
            return {
                "best": float(s[order[0]]),
                "second": float(s[order[1]]) if len(order) >= 2 else float(s[order[0]]),
                "margin": float(s[order[0]] - s[order[1]]) if len(order) >= 2 else 0.0,
                "top5": float(np.mean(s[order[:5]])),
                "nearest_class": int(ids[order[0]]),
            }
        a = stats(raw)
        b = stats(adapted)
        rows.append({
            "sample_id": sid, "arrival_index": idx,
            "true_class": int(r["class"]),
            "raw_best_known": a["best"], "raw_margin": a["margin"],
            "raw_top5": a["top5"], "raw_nearest_class": a["nearest_class"],
            "adapted_best_known": b["best"], "adapted_margin": b["margin"],
            "adapted_top5": b["top5"], "adapted_nearest_class": b["nearest_class"],
        })
    return rows


def bucket(i, n=5232):
    # active memory bucket approximation by arrival fraction is not accurate;
    # use actual M2 trajectory logs where available.
    return None


def build_permuted_rows(rows, perm):
    """Return a copy of rows in permuted order with recomputed first flags."""
    out = []
    seen = set()
    for idx in perm:
        r = dict(rows[idx])
        r["stream_order"] = len(out)
        r["first_occurrence"] = r["class"] not in seen
        seen.add(r["class"])
        out.append(r)
    return out


def permutations(rows, hardness, seeds=(101, 202, 303, 404, 505)):
    n = len(rows)
    novel_idx = [i for i, r in enumerate(rows) if r["role"] == "novel"]
    known_idx = [i for i, r in enumerate(rows) if r["role"] == "known"]
    novel_classes = sorted({int(rows[i]["class"]) for i in novel_idx})
    perms = {"P0": list(range(n))}
    for s in seeds:
        rng = np.random.RandomState(s)
        perms[f"P1_{s}"] = rng.permutation(n).tolist()
    # P2: known skeleton fixed; novel class blocks shuffled
    blocks = {c: [i for i in novel_idx if int(rows[i]["class"]) == c]
              for c in novel_classes}
    for s in seeds:
        rng = np.random.RandomState(s)
        order = list(novel_classes)
        rng.shuffle(order)
        nov = []
        for c in order:
            nov.extend(blocks[c])
        it = iter(nov)
        p = []
        for i in range(n):
            if rows[i]["role"] == "known":
                p.append(i)
            else:
                p.append(next(it))
        perms[f"P2_{s}"] = p
    # P3 hard-first / P4 easy-first (novel blocks ordered by class hardness)
    by_class = defaultdict(list)
    for h in hardness:
        by_class[h["true_class"]].append(h["adapted_best_known"])
    class_hard = {c: float(np.mean(v)) for c, v in by_class.items()}
    hard_order = sorted(novel_classes, key=lambda c: -class_hard[c])
    easy_order = sorted(novel_classes, key=lambda c: class_hard[c])
    for name, order in [("P3", hard_order), ("P4", easy_order)]:
        nov = []
        for c in order:
            nov.extend(blocks[c])
        it = iter(nov)
        p = []
        for i in range(n):
            if rows[i]["role"] == "known":
                p.append(i)
            else:
                p.append(next(it))
        perms[name] = p
    return perms


def replay_perm(model, ck, rows, feats, labels, device, perm_name, perm):
    from src.orbit.evaluate import build_known
    from src.orbit_iam.iam_memory import IamMemory
    from src.orbit_iam.compat import compat_matrix_for_track
    from src.orbit_msr.protocol import known_stats
    from src.iclr27_phase4d.long_stream import stage_of
    p_rows = build_permuted_rows(rows, perm)
    known_classes = sorted(set(labels.values()))
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    protos, radii = build_known(model, train_feats, labels,
                                set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in p_rows], device)
    mem = IamMemory(protos, radii, novel_update_rate=0.2)
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    feat_names = [f.strip() for f in ck.get(
        "compat_feats", "sim,margin,radius,support,mem,rel").split(",") if f.strip()]
    logs = []
    n = len(p_rows)
    for i, r in enumerate(p_rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
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
            dist_n = (1.0 - best_n) / max(mem.novel_radii.get(nid, 0.3), 1e-6)
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_prob = float(torch.sigmoid(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0]))
        action = "KNOWN"
        if gate_prob < 0.5:
            states = {v: mem.state(v) for v in sorted(mem.novel)}
            q_best = -1.0
            q_second = -1.0
            if P_novel.shape[0]:
                X = compat_matrix_for_track(
                    z, {v: mem.novel[v]["proto"] for v in sorted(mem.novel)},
                    states, len(mem.novel), rel, margin_n, feat_names)
                with torch.no_grad():
                    q = torch.sigmoid(model.compat_forward(
                        torch.as_tensor(X, dtype=torch.float32, device=device))
                    ).cpu().numpy()
                if q.shape[0]:
                    qo = np.argsort(q)[::-1]
                    q_best = float(q[qo[0]])
                    q_second = float(q[qo[1]]) if q.shape[0] >= 2 else -1.0
                    nid = int(sorted(mem.novel)[int(qo[0])])
            if (q_best >= 0.45 and (len(mem.novel) < 2 or q_best - q_second >= 0.05)):
                action = "EXISTING"
                cos = float(np.dot(mem.novel[nid]["proto"], z))
                mem.update_novel(nid, z, cos_to_center=cos, update_radius=True,
                                 margin=margin_n)
            else:
                action = "NEW"
                mem.create_novel(z, created_at=i)
        logs.append({
            "permutation": perm_name, "arrival_index": i,
            "sample_id": r["sample_id"], "true_role": r["role"],
            "true_class": int(r["class"]), "first_occurrence": r["first_occurrence"],
            "predicted_action": action, "gate_prob": gate_prob,
            "memory_size": len(mem.novel),
            "stage": stage_of(i, n),
        })
    return logs


def replay_until_mem(model, ck, rows, feats, labels, device, target):
    """Replay the first `target` rows of the original order; return mem."""
    from src.orbit.evaluate import build_known
    from src.orbit_iam.iam_memory import IamMemory
    from src.orbit_msr.protocol import known_stats
    known_classes = sorted(set(labels.values()))
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    protos, radii = build_known(model, train_feats, labels,
                                set(known_classes), device)
    sub = rows[:target]
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in sub], device)
    mem = IamMemory(protos, radii, novel_update_rate=0.2)
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    feat_names = [f.strip() for f in ck.get(
        "compat_feats", "sim,margin,radius,support,mem,rel").split(",") if f.strip()]
    for i, r in enumerate(sub):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
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
            dist_n = (1.0 - best_n) / max(mem.novel_radii.get(nid, 0.3), 1e-6)
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_prob = float(torch.sigmoid(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0]))
        if gate_prob >= 0.5:
            continue
        from src.orbit_iam.compat import compat_matrix_for_track
        states = {v: mem.state(v) for v in sorted(mem.novel)}
        q_best = -1.0
        q = np.array([])
        if P_novel.shape[0]:
            X = compat_matrix_for_track(
                z, {v: mem.novel[v]["proto"] for v in sorted(mem.novel)},
                states, len(mem.novel), rel, margin_n, feat_names)
            with torch.no_grad():
                q = torch.sigmoid(model.compat_forward(
                    torch.as_tensor(X, dtype=torch.float32, device=device))
                ).cpu().numpy()
            if q.shape[0]:
                q_best = float(q.max())
                nid = int(sorted(mem.novel)[int(np.argmax(q))])
        q_second = (float(np.partition(q, -2)[-2]) if q.shape[0] >= 2 else -1.0)
        if q_best >= 0.45 and (len(mem.novel) < 2 or q_best - q_second >= 0.05):
            cos = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos, update_radius=True,
                             margin=margin_n)
        else:
            mem.create_novel(z, created_at=i)
    return mem


def gate_under_snapshot(model, z, rel, feats, labels, mem, device):
    """Gate probability for one query under a given memory snapshot."""
    from src.orbit.evaluate import build_known
    from src.orbit_msr.protocol import known_stats
    known_classes = sorted(set(labels.values()))
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    protos, radii = build_known(model, train_feats, labels,
                                set(known_classes), device)
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
               .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
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
        dist_n = (1.0 - best_n) / max(mem.novel_radii.get(nid, 0.3), 1e-6)
    ks = P_known @ z
    gs = known_stats(z, P_known, radii, known_ids=known_ids,
                     best_n=best_n, second_n=second_n, margin_n=margin_n,
                     dist_n=dist_n, rel=rel,
                     track_len=8, n_novel=len(mem.novel), include_anchor=False)
    with torch.no_grad():
        gate_prob = float(torch.sigmoid(model.gate_forward(
            torch.as_tensor([gs], dtype=torch.float32, device=device))[0]))
    return gate_prob, best_n, margin_n, len(mem.novel)


def summarize_perm(logs):
    novel = [l for l in logs if l["true_role"] == "novel"]
    n2k = sum(1 for l in novel if l["predicted_action"] == "KNOWN")
    routed = sum(1 for l in novel if l["predicted_action"] != "KNOWN")
    return {
        "n2k_rate": n2k / max(len(novel), 1),
        "novel_routed": routed / max(len(novel), 1),
        "final_memory": logs[-1]["memory_size"],
        "novel_tracks": len(novel),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hardness", "permute", "counterfactual",
                                        "probe", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    import pathlib
    out = pathlib.Path(f"{ROOT}/outputs/iclr27_phase4h/audit")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    rows, gt, feats, train_feats = prepare_official()
    labels = load_train_labels()
    model, ck = load_iam_model(f"{ROOT}/runs/orbit_mdc/mdc_m2/model.pth", device)
    zs, _ = embed_many(model, feats, [r["sample_id"] for r in rows], device)

    raw_scores, ids, _ = raw_known_scores(feats, [r["sample_id"] for r in rows])
    from src.orbit.evaluate import build_known
    known_classes = sorted(set(labels.values()))
    protos, radii = build_known(model, train_feats, labels, set(known_classes),
                                device)
    P_adapted = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    adapted_scores = {sid: P_adapted @ zs[sid] for sid in zs}
    hard = hardness_rows(rows, feats, adapted_scores, raw_scores, ids)

    if args.stage in ("hardness", "all"):
        # bucket stats using original M2 trajectory memory sizes
        traj = {l["sample_id"]: l for l in csv.DictReader(open(
            f"{ROOT}/outputs/iclr27_phase4f/audit/memory_trajectory_m2_official.csv"))}
        for h in hard:
            t = traj.get(h["sample_id"], {})
            h["memory_size"] = int(t.get("memory_size", 0))
            h["n2k"] = int(t.get("predicted_action") == "KNOWN") if t else -1
        for fname, key in [("raw_feature_hardness_by_bucket.csv", "raw_best_known"),
                           ("adapted_feature_hardness_by_bucket.csv", "adapted_best_known")]:
            table = []
            for b in ["0-32", "33-128", "129-256", "257+"]:
                sel = [h for h in hard if
                       ("0-32" if h["memory_size"] < 33 else
                        "33-128" if h["memory_size"] < 129 else
                        "129-256" if h["memory_size"] < 257 else "257+") == b]
                if not sel:
                    continue
                table.append({
                    "bucket": b, "n": len(sel),
                    "mean_best_known": float(np.mean([h[key] for h in sel])),
                    "median_best_known": float(np.median([h[key] for h in sel])),
                    "p90_best_known": float(np.percentile([h[key] for h in sel], 90)),
                    "mean_margin": float(np.mean([h["raw_margin" if "raw" in key else "adapted_margin"] for h in sel])),
                    "n2k_rate": float(np.mean([h["n2k"] for h in sel])),
                })
            with open(out / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
                w.writeheader()
                w.writerows(table)
        # class hardness
        by_class = defaultdict(list)
        for h in hard:
            by_class[h["true_class"]].append(h)
        cls_rows = []
        for c, hs in by_class.items():
            cls_rows.append({
                "class": c, "track_count": len(hs),
                "mean_adapted_best_known": float(np.mean([h["adapted_best_known"] for h in hs])),
                "median_adapted_best_known": float(np.median([h["adapted_best_known"] for h in hs])),
                "mean_adapted_margin": float(np.mean([h["adapted_margin"] for h in hs])),
                "mean_raw_best_known": float(np.mean([h["raw_best_known"] for h in hs])),
                "first_arrival": min(h["arrival_index"] for h in hs),
                "mean_arrival": float(np.mean([h["arrival_index"] for h in hs])),
                "n2k_rate": float(np.mean([h["n2k"] for h in hs])),
            })
        with open(out / "novel_class_hardness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
            w.writeheader()
            w.writerows(cls_rows)
        print("hardness done", len(hard), len(cls_rows))

    if args.stage in ("permute", "all"):
        perms = permutations(rows, hard)
        results = []
        for name, perm in perms.items():
            logs = replay_perm(model, ck, rows, feats, labels, device, name, perm)
            s = summarize_perm(logs)
            s["permutation"] = name
            results.append(s)
            print(s, flush=True)
        with open(out / "permutation_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        # per-track logs for probe
        with open(out / "permutation_logs.json", "w") as f:
            all_logs = []
            for name, perm in perms.items():
                all_logs.extend(replay_perm(model, ck, rows, feats, labels,
                                            device, name, perm))
            json.dump(all_logs, f)
        print("permutations done", len(perms))

    if args.stage in ("counterfactual", "all"):
        snapshots = {t: replay_until_mem(model, ck, rows, feats, labels,
                                         device, t)
                     for t in [500, 2000, 4500]}
        qids = [r["sample_id"] for r in rows if r["role"] == "novel"][::30][:20]
        q_zs, q_rels = embed_many(model, feats, qids, device)
        cf_rows = []
        for q in qids:
            for target, mem in snapshots.items():
                gp, bn, mn, ms = gate_under_snapshot(
                    model, q_zs[q], q_rels[q], feats, labels, mem, device)
                cf_rows.append({
                    "query": q, "snapshot_arrival": target,
                    "gate_prob_under_snapshot": gp,
                    "best_novel_sim": bn, "novel_margin": mn,
                    "snapshot_memory_size": ms,
                })
        with open(out / "counterfactual_memory_replay.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cf_rows[0].keys()))
            w.writeheader()
            w.writerows(cf_rows)
        print("counterfactual done")


if __name__ == "__main__":
    main()
