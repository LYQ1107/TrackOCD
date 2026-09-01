"""Long-Stream Meta-Development protocol for Phase 4D.

Fixed-known long-stream proxy:
- known classes: all 48 train-known classes with prototypes built from their
  real train tracks (identical to the official evaluation space);
- real novel classes: the 10 frozen meta-dev classes (63 real tracks);
- synthetic novel classes: train-side feature-space perturbations of
  train-known tracks, generated a priori so that their best-known similarity
  lies in the ambiguous band below known intra-class similarity and above
  random similarity (no official-validation statistics are used to choose
  the parameters);
- the stream is strictly causal, order fixed by seed, and the active novel
  prototype count grows from ~0 to >250.

This is a development stress test only; official evaluation still uses the
frozen GT-track stream.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import (
    load_frame_features,
    load_mean_features,
    load_train_labels,
    meta_classes,
)
from src.orbit.evaluate import load_model, build_known, embed_track
from src.orbit.bi_memory import stats_to_tensor
from src.orbit_fc.evaluate import load_fc_model
from src.iclr27_phase4c.audit_common import emit_preds, assignment_from_preds


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def build_long_stream(seed=2026, n_synthetic_novel=200, tracks_range=(2, 6),
                      alpha_range=(0.35, 0.75), sigma=0.08, n_known_repeat=2):
    """Build rows, GT rows and a features dict (frame-style arrays)."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    labels = load_train_labels()
    mean_feats = load_mean_features("train_known_mean")
    frame_feats = load_frame_features("train_known_mean")
    all_classes = sorted(set(labels.values()))
    meta_dev = sorted(meta_classes("meta_dev_classes"))
    meta_train = sorted(meta_classes("meta_train_classes"))
    known_classes = all_classes  # fixed 48, as in official evaluation

    # known prototype matrix (DINO mean space) for similarity checks
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if sid in mean_feats:
            sums[c] += mean_feats[sid]
            counts[c] += 1
    known_protos = {}
    for c, s in sums.items():
        v = s / counts[c]
        known_protos[c] = _norm(v)
    P_known = np.stack([known_protos[c] for c in sorted(known_protos)]).astype(np.float32)
    known_ids = sorted(known_protos)

    # ---- real novel tracks (meta-dev) ----
    real_novel_ids = [sid for sid, c in labels.items() if c in meta_dev and sid in mean_feats]
    # ---- synthetic novel classes ----
    syn_rows = []
    syn_vid = 1000000
    pool_ids = [sid for sid, c in labels.items()
                if c in meta_train and sid in mean_feats and sid not in real_novel_ids]
    rng.shuffle(pool_ids)
    for k in range(n_synthetic_novel):
        base_sid = pool_ids[k % len(pool_ids)]
        base = mean_feats[base_sid]
        alpha = float(np_rng.uniform(*alpha_range))
        w = np_rng.randn(768).astype(np.float32)
        w = _norm(w)
        center = _norm(alpha * base + (1.0 - alpha) * w)
        # verify best-known similarity is in the ambiguous band (train-side
        # prior: below known intra-class ~0.5, above random ~0.1-0.3)
        best_k = float(np.max(P_known @ center))
        n_tracks = int(np_rng.randint(tracks_range[0], tracks_range[1] + 1))
        for j in range(n_tracks):
            noise = np_rng.randn(768).astype(np.float32)
            z = _norm(center + sigma * noise / np.linalg.norm(noise))
            # frame-style array: 6 frames with small per-frame jitter
            frames = np.stack([z for _ in range(8)]).astype(np.float32)
            syn_rows.append({"sample_id": f"syn_{syn_vid}_{j}", "class": syn_vid,
                             "role": "novel", "frames": frames,
                             "best_known_center": best_k})
        syn_vid += 1

    # ---- assemble stream ----
    known_rows = []
    base_known = [{"sample_id": sid, "class": c, "role": "known"}
                  for sid, c in labels.items() if sid in mean_feats and c in known_classes]
    for rep in range(n_known_repeat):
        rows_rep = list(base_known)
        rng.shuffle(rows_rep)
        for r in rows_rep:
            known_rows.append({"sample_id": f"{r['sample_id']}#{rep}",
                               "class": r["class"], "role": "known",
                               "orig_id": r["sample_id"]})
    real_rows = [{"sample_id": sid, "class": labels[sid], "role": "novel",
                  "frames": None, "best_known_center": None}
                 for sid in real_novel_ids]
    rng.shuffle(real_rows)
    all_rows = []
    for r in known_rows:
        all_rows.append(r)
    for r in real_rows:
        all_rows.append(r)
    for r in syn_rows:
        all_rows.append(r)
    # interleave: walk through the concatenated list with a seeded shuffle of
    # bucket order so known/novel mix throughout the stream
    order = list(range(len(all_rows)))
    rng.shuffle(order)
    rows = []
    gt_rows = []
    feats = {}
    syn_mean = {}
    seen = {}
    for i, idx in enumerate(order):
        r = all_rows[idx]
        sid = r["sample_id"]
        if r["role"] == "known":
            orig = r.get("orig_id", sid)
            feats[sid] = frame_feats[orig][:8]
            feats.setdefault(orig, frame_feats[orig][:8])
            gt_rows.append({"sample_id": sid,
                            "ground_truth_category_id": r["class"],
                            "protocol_role": "supported_known"})
        else:
            if r["frames"] is not None:
                feats[sid] = r["frames"]
                syn_mean[sid] = r["frames"][0].copy()
            else:
                feats[sid] = frame_feats[sid][:8]
            gt_rows.append({"sample_id": sid,
                            "ground_truth_category_id": r["class"],
                            "protocol_role": "novel"})
        key = r["class"]
        first = key not in seen
        seen[key] = first
        rows.append({"sample_id": sid, "stream_order": i, "role": r["role"],
                     "class": r["class"], "first_occurrence": first})
    return rows, gt_rows, feats, syn_mean


def active_bucket(active):
    if active < 33:
        return "0-32"
    if active < 129:
        return "33-128"
    if active < 257:
        return "129-256"
    return "257+"


def _log_row(r, i, action, kid, vid, active, support, bk, bn, stage):
    return {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "stage": stage,
    }


def stage_of(i, n):
    if i < n / 3:
        return "early"
    if i < 2 * n / 3:
        return "middle"
    return "late"


def replay_method(method, rows, feats, labels, device, d1_model=None,
                  fc_model=None, fc_ck=None, birth_threshold=0.55,
                  gate_thr=0.5, reuse_thr=0.45, syn_mean=None):
    n = len(rows)
    known_classes = sorted(set(labels.values()))
    logs = []
    if method == "ref":
        from src.dual_branch.memory.b2_adapter import B2Memory
        mean_feats = load_mean_features("train_known_mean")
        sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
        counts = defaultdict(int)
        for sid, c in labels.items():
            if sid in mean_feats:
                sums[c] += mean_feats[sid]
                counts[c] += 1
        protos = {}
        for c, s in sums.items():
            protos[c] = _norm(s / counts[c])
        b2 = B2Memory(protos, threshold=0.45)
        for i, r in enumerate(rows):
            z = syn_mean.get(r["sample_id"]) if syn_mean else None
            if z is None:
                orig = r["sample_id"].split("#")[0]
                z = mean_feats[orig]
            vid, kind = b2.predict_one(z, r["sample_id"], i)
            action = "KNOWN" if kind == "known" else (
                "EXISTING_NOVEL" if b2.counts.get(vid, 1) > 1 else "NEW_NOVEL")
            best_k = float(max(np.dot(z, p) for p in protos.values()))
            logs.append(_log_row(r, i, action,
                                 vid if kind == "known" else None,
                                 vid if kind == "novel" else None,
                                 len(b2.novel),
                                 b2.counts.get(vid, 0) if kind == "novel" else 0,
                                 best_k, 0.0, stage_of(i, n)))
        return logs

    if method == "bc":
        from src.orbit_fc.causal_memory import CausalNovelMemory
        protos, radii = build_known(d1_model, feats, labels, set(known_classes), device)
        mem = CausalNovelMemory(protos, radii, novel_update_rate=0.2)
        for i, r in enumerate(rows):
            z, rel = embed_track(d1_model, feats[r["sample_id"]], device)
            kid, ks = mem.known_id(z)
            nid, ns = mem.existing_novel(z)
            stats = mem.stats(z, rel, len(feats[r["sample_id"]]),
                              known_id=kid, novel_id=nid)
            with torch.no_grad():
                logits = d1_model.action_net(stats_to_tensor(stats, device))
            action = int(logits.argmax(dim=1).item())
            if action == 2 and nid is not None and ns >= birth_threshold:
                action = 1
            if action == 0 and kid is not None:
                logs.append(_log_row(r, i, "KNOWN", kid, None, len(mem.novel),
                                     0, float(ks), float(ns), stage_of(i, n)))
            elif action == 1 and nid is not None:
                support = mem.novel_counts.get(nid, 0)
                mem.update_novel(nid, z)
                logs.append(_log_row(r, i, "EXISTING_NOVEL", None, nid,
                                     len(mem.novel), support, float(ks), float(ns),
                                     stage_of(i, n)))
            else:
                vid = mem.create_novel(z, created_at=i)
                logs.append(_log_row(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                                     0, float(ks), float(ns), stage_of(i, n)))
        return logs

    if method == "fc":
        from src.orbit_fc.causal_memory import CausalNovelMemory
        from src.orbit_fc.protocol import (
            frozen_known_protos, known_stats, novel_stats,
        )
        protos, radii = build_known(fc_model, feats, labels, set(known_classes), device)
        frozen = frozen_known_protos(set(known_classes))
        P_frozen = np.stack([frozen[c] for c in sorted(frozen)]).astype(np.float32)
        mean_feats = load_mean_features("train_known_mean")
        mem = CausalNovelMemory(protos, radii, novel_update_rate=0.2)
        P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
        known_ids = sorted(protos)
        for i, r in enumerate(rows):
            z, rel = embed_track(fc_model, feats[r["sample_id"]], device)
            z0 = mean_feats.get(r["sample_id"])
            ks = P_known @ z
            kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
            best_k = float(ks.max()) if ks.shape[0] else -1.0
            P_novel = np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)]).astype(np.float32) if mem.novel else np.empty((0, 768), dtype=np.float32)
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
            anchor = float(np.max(z0 @ P_frozen.T)) if z0 is not None and P_frozen.shape[0] else -1.0
            gs = known_stats(z, P_known, radii, known_ids=known_ids,
                             anchor_best=anchor, best_n=best_n,
                             second_n=second_n, margin_n=margin_n,
                             dist_n=dist_n, rel=rel,
                             track_len=len(feats[r["sample_id"]]),
                             n_novel=len(mem.novel),
                             include_anchor=fc_ck.get("use_anchor", False))
            with torch.no_grad():
                gate_logit = float(fc_model.gate_forward(
                    stats_to_tensor(gs, device))[0])
            if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr and kid is not None:
                logs.append(_log_row(r, i, "KNOWN", kid, None, len(mem.novel),
                                     0, best_k, best_n, stage_of(i, n)))
                continue
            rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                             novel_ids=sorted(mem.novel) if mem.novel else None,
                             best_k=best_k, margin_k=_margin(ks),
                             rel=rel, track_len=len(feats[r["sample_id"]]),
                             n_novel=len(mem.novel),
                             age_norm=mem.age(nid, i) if nid is not None else 0.0)
            with torch.no_grad():
                reuse_logit = float(fc_model.reuse_forward(
                    stats_to_tensor(rs, device))[0])
            prob_reuse = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
            if prob_reuse >= reuse_thr and nid is not None:
                support = mem.support(nid)
                cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
                mem.update_novel(nid, z, cos_to_center=cos_to_center)
                logs.append(_log_row(r, i, "EXISTING_NOVEL", None, nid,
                                     len(mem.novel), support, best_k, best_n,
                                     stage_of(i, n)))
            else:
                vid = mem.create_novel(z, created_at=i)
                logs.append(_log_row(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                                     0, best_k, best_n, stage_of(i, n)))
        return logs
    raise ValueError(method)


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def evaluate_split(logs, gt_rows, select=None):
    from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
    if select is not None:
        logs = [l for l in logs if select(l)]
    if not logs:
        return None
    preds = emit_preds(logs)
    sids = {l["sample_id"] for l in logs}
    gt = [g for g in gt_rows if g["sample_id"] in sids]
    res, _ = assignment_from_preds(preds, gt)
    return res


def bucket_statistics(logs):
    rows = []
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        ls = [l for l in logs if active_bucket(l["active_novel_prototypes"]) == bucket]
        if not ls:
            continue
        roles = Counter(l["role"] for l in ls)
        first = Counter(l["first_occurrence"] for l in ls if l["role"] == "novel")
        rows.append({"bucket": bucket, "tracks": len(ls),
                     "known_tracks": roles["known"], "novel_tracks": roles["novel"],
                     "novel_first": first.get(True, 0), "novel_repeated": first.get(False, 0)})
    return rows


def save_stream_cache(rows, gt_rows, feats, syn_mean):
    out_dir = ROOT / "outputs" / "iclr27_phase4d" / "long_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = list(feats.keys())
    feats_padded = np.zeros((len(sids), 8, 768), dtype=np.float32)
    lens = np.zeros(len(sids), dtype=np.int32)
    for i, s in enumerate(sids):
        arr = feats[s]
        feats_padded[i, :arr.shape[0]] = arr
        lens[i] = arr.shape[0]
    np.savez_compressed(
        out_dir / "stream_cache.npz",
        rows=np.array([json.dumps(r) for r in rows]),
        gt=np.array([json.dumps(g) for g in gt_rows]),
        sids=np.array(sids),
        feats=feats_padded,
        lens=lens,
        syn_mean_sids=np.array(list(syn_mean.keys())),
        syn_mean_arr=np.stack([syn_mean[s] for s in syn_mean.keys()]).astype(np.float32),
    )


def load_stream_cache():
    out_dir = ROOT / "outputs" / "iclr27_phase4d" / "long_stream"
    d = np.load(out_dir / "stream_cache.npz", allow_pickle=True)
    rows = [json.loads(r) for r in d["rows"]]
    gt_rows = [json.loads(g) for g in d["gt"]]
    sids = [str(s) for s in d["sids"]]
    feats_all = d["feats"]
    lens_all = d["lens"]
    feats = {s: feats_all[i][:int(lens_all[i])] for i, s in enumerate(sids)}
    syn_sids = [str(s) for s in d["syn_mean_sids"]]
    syn_all = d["syn_mean_arr"]
    syn_mean = {s: syn_all[i] for i, s in enumerate(syn_sids)}
    return rows, gt_rows, feats, syn_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_synthetic_novel", type=int, default=200)
    ap.add_argument("--n_known_repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cache_only", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    rows, gt_rows, feats, syn_mean = build_long_stream(
        seed=args.seed, n_synthetic_novel=args.n_synthetic_novel,
        n_known_repeat=args.n_known_repeat)
    labels = load_train_labels()
    out_dir = ROOT / "outputs" / "iclr27_phase4d" / "long_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stream_manifest.json").write_text(json.dumps({
        "seed": args.seed, "n_synthetic_novel": args.n_synthetic_novel,
        "num_tracks": len(rows),
        "num_known_tracks": sum(1 for r in rows if r["role"] == "known"),
        "num_novel_tracks": sum(1 for r in rows if r["role"] == "novel"),
        "known_prototype_classes": len(set(labels.values())),
        "novel_real_classes": sorted(set(int(r["class"]) for r in rows
                                          if r["role"] == "novel" and int(r["class"]) < 1000000)),
        "novel_synthetic_classes": len(set(int(r["class"]) for r in rows
                                            if int(r["class"]) >= 1000000)),
    }, indent=1))
    save_stream_cache(rows, gt_rows, feats, syn_mean)
    if args.cache_only:
        print("cache saved")
        return
    d1_model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    fc_model, fc_ck = load_fc_model(ROOT / "runs/orbit_fc/fc_F1/model.pth", device=device)
    results = {}
    for method in ["ref", "bc", "fc"]:
        logs = replay_method(method, rows, feats, labels, device,
                             d1_model=d1_model, fc_model=fc_model, fc_ck=fc_ck,
                             syn_mean=syn_mean)
        results[method] = logs
        print(method, "tracks", len(logs), flush=True)
    with open(out_dir / "scale_bucket_statistics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "tracks", "known_tracks",
                                          "novel_tracks", "novel_first", "novel_repeated"])
        w.writeheader()
        w.writerows(bucket_statistics(results["fc"]))
    comp_rows = []
    for method, logs in results.items():
        res = evaluate_split(logs, gt_rows)
        comp_rows.append({"method": method.upper(), "scope": "overall",
                          "all_acc": res["all_track_acc"],
                          "known_acc": res["overall_known_acc"],
                          "rn_acc": res["route_aware_novel_acc"],
                          "cond_novel_acc": res["conditional_novel_acc"],
                          "routing_recall": res["novel_routing_recall"],
                          "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
                          "count_error": res["novel_count_abs_error"],
                          "predicted_novel_count": res["predicted_novel_count"]})
        print(comp_rows[-1], flush=True)
        for bucket in ["0-32", "33-128", "129-256", "257+"]:
            res_b = evaluate_split(logs, gt_rows,
                                   select=lambda l, b=bucket:
                                   active_bucket(l["active_novel_prototypes"]) == b)
            if res_b is None:
                continue
            comp_rows.append({"method": method.upper(), "scope": bucket,
                              "all_acc": res_b["all_track_acc"],
                              "known_acc": res_b["overall_known_acc"],
                              "rn_acc": res_b["route_aware_novel_acc"],
                              "cond_novel_acc": res_b["conditional_novel_acc"],
                              "routing_recall": res_b["novel_routing_recall"],
                              "nmi": res_b["novel_only_nmi"], "ari": res_b["novel_only_ari"],
                              "count_error": res_b["novel_count_abs_error"],
                              "predicted_novel_count": res_b["predicted_novel_count"]})
        for stage in ["early", "middle", "late"]:
            res_s = evaluate_split(logs, gt_rows,
                                   select=lambda l, s=stage: l["stage"] == s)
            if res_s is None:
                continue
            comp_rows.append({"method": method.upper(), "scope": stage,
                              "all_acc": res_s["all_track_acc"],
                              "known_acc": res_s["overall_known_acc"],
                              "rn_acc": res_s["route_aware_novel_acc"],
                              "cond_novel_acc": res_s["conditional_novel_acc"],
                              "routing_recall": res_s["novel_routing_recall"],
                              "nmi": res_s["novel_only_nmi"], "ari": res_s["novel_only_ari"],
                              "count_error": res_s["novel_count_abs_error"],
                              "predicted_novel_count": res_s["predicted_novel_count"]})
    with open(out_dir / "proxy_method_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)
    for method, logs in results.items():
        with open(out_dir / f"per_track_{method}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            w.writeheader()
            w.writerows(logs)
    print("done")


if __name__ == "__main__":
    main()
