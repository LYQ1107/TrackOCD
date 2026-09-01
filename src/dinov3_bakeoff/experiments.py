#!/usr/bin/env python3
"""DINOv2-DINOv3 bake-off experiments: V0, V2, O0/O1, offline oracle-K,
geometry diagnostics, backbone gate."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_bakeoff.calibration import calibrate_b2_threshold
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.data.track_stream_dataset import load_stream_rows
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_train_known, load_mean_features
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids

OUT = PROJECT_ROOT / "outputs" / "dinov3_bakeoff" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "dinov3_bakeoff"
D3 = PROJECT_ROOT / "data" / "caches" / "features" / "dinov3_vitb16_lvd1689m"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def load_dinov3_mean(subdir):
    feats = {}
    for p in (D3 / subdir).glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    return feats


def load_dinov3_single(subdir):
    feats = {}
    for p in (D3 / subdir).glob("*.json"):
        r = json.loads(p.read_text())
        v = r.get("single_embedding")
        if v is None:
            v = r["frame_embeddings"][len(r["frame_embeddings"]) // 2]
        feats[r["sample_id"]] = np.asarray(v, dtype=np.float32)
    return feats


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_online(method, proto, subset, stream, feats, protos, thr, gt, oracle=False):
    srows = load_stream_rows(stream)
    sub = subset_ids(proto, subset)
    mem = B2Memory(protos, threshold=thr, novel_only=oracle)
    gt_by_sid = {g["sample_id"]: g for g in gt}
    preds = []
    for i, r in enumerate(srows):
        sid = r["sample_id"]
        g = gt_by_sid.get(sid)
        if oracle and g is not None and g["protocol_role"] in ("supported_known", "zero_shot_known"):
            preds.append(emit(sid, i, "known", known_id=g["ground_truth_category_id"]))
            continue
        if oracle:
            vid, _ = mem.predict_one(feats[sid], sid, i)
            preds.append(emit(sid, i, "novel", virtual_id=vid))
        else:
            vid, kind = mem.predict_one(feats[sid], sid, i)
            preds.append(emit(sid, i, kind, vid if kind == "known" else None,
                              vid if kind == "novel" else None))
    ev = TrackOCDEvaluator(gt)
    res = ev.evaluate(preds, subset_ids=sub)
    row = {
        "method": method, "protocol": proto, "subset": subset, "seed": stream,
        **{k: res[k] for k in res if k != "hungarian_assignment"},
    }
    return row


def offline_oracle_k(feats, ids, labels_of, n_clusters, seed=1027):
    X = np.stack([feats[s] for s in ids])
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed, max_iter=300).fit(X)
    y = np.array([labels_of[s] for s in ids])
    cats = sorted(set(int(v) for v in y))
    W = np.zeros((n_clusters, len(cats)), dtype=np.int64)
    for lab, c in zip(km.labels_, y):
        W[lab, cats.index(int(c))] += 1
    rows_, cols_ = linear_sum_assignment(-W)
    cmap = {int(rows_[i]): cats[int(cols_[i])] for i in range(len(rows_))}
    preds = np.array([cmap[int(l)] for l in km.labels_])
    acc = (preds == y).mean()
    nmi = normalized_mutual_info_score(y, preds)
    ari = adjusted_rand_score(y, preds)
    # macro per class
    macro = np.mean([(preds[y == c] == c).mean() for c in cats])
    frag = np.mean([len(set(preds[y == c])) for c in cats])
    used = defaultdict(set)
    for p, c in zip(preds, y):
        used[int(p)].add(int(c))
    merge = sum(1 for s in used.values() if len(s) > 1) / len(used) if used else 0.0
    return {
        "acc": float(acc), "nmi": float(nmi), "ari": float(ari),
        "macro_acc": float(macro), "mean_fragmentation": float(frag),
        "merge_error": float(merge),
    }


def geometry(feats, private_rows, ids):
    X = {s: feats[s] / (np.linalg.norm(feats[s]) + 1e-12) for s in ids}
    cat = {r["sample_id"]: r["ground_truth_category_id"] for r in private_rows}
    video = {r["sample_id"]: int(r["sample_id"].split("_")[0]) for r in private_rows}
    # intra-track cross-frame cosine (mean features only give 1 frame; use frames)
    intra_track = None
    # intra/inter class on track means
    intra = []
    inter = []
    by_cat = defaultdict(list)
    for s in ids:
        by_cat[cat[s]].append(s)
    cats = [c for c in by_cat if len(by_cat[c]) >= 2]
    for c in cats:
        grp = by_cat[c]
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                intra.append(float(np.dot(X[grp[i]], X[grp[j]])))
    for a in range(0, len(ids), 200):
        for b in range(a + 1, len(ids), 200):
            sa, sb = ids[a], ids[b]
            if cat[sa] != cat[sb]:
                inter.append(float(np.dot(X[sa], X[sb])))
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    # kNN purity k=1/5/10 on all tracks; cross-video same-class
    arr = np.stack([X[s] for s in ids])
    sim = arr @ arr.T
    np.fill_diagonal(sim, -1)
    def purity(k, cross_video=False):
        hits = 0
        total = 0
        for i in range(len(ids)):
            top = np.argsort(sim[i])[-k:][::-1]
            for j in top:
                if cross_video and video[ids[i]] == video[ids[j]]:
                    continue
                hits += 1 if cat[ids[i]] == cat[ids[j]] else 0
                total += 1
        return hits / total if total else 0.0
    return {
        "intra_class_cosine": intra_mean,
        "inter_class_cosine": inter_mean,
        "intra_inter_ratio": intra_mean / max(inter_mean, 1e-9),
        "knn_purity_k1": purity(1),
        "knn_purity_k5": purity(5),
        "knn_purity_k10": purity(10),
        "knn_purity_k5_cross_video": purity(5, cross_video=True),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    d2_mean = load_mean_features("dinov2", "gt_tracks_mean")
    d2_single = load_mean_features("dinov2", "gt_tracks_single")
    d3_mean = load_dinov3_mean("gt_tracks/mean")
    d3_single = load_dinov3_single("gt_tracks/single")
    d2_tr, labels = load_train_known("dinov2")
    d3_tr = load_dinov3_mean("train_known")
    d2_protos = build_protos(d2_tr, labels)
    d3_protos = build_protos(d3_tr, labels)
    thr_d3, curve = calibrate_b2_threshold(d3_tr, labels)
    (RUNS / "calibration_curves.json").write_text(
        json.dumps({"dinov3": curve, "best_threshold": thr_d3}, indent=2))
    print("DINOv3 threshold", thr_d3, flush=True)

    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        subsets = SUBSETS if proto == "pure" else ("full",)
        for subset in subsets:
            for stream in STREAMS:
                rows.append(run_online("V0", proto, subset, stream, d2_mean, d2_protos, 0.45, gt))
                rows.append(run_online("V2", proto, subset, stream, d3_mean, d3_protos, thr_d3, gt))
                rows.append(run_online("O0", proto, subset, stream, d2_mean, d2_protos, 0.45, gt, oracle=True))
                rows.append(run_online("O1", proto, subset, stream, d3_mean, d3_protos, thr_d3, gt, oracle=True))
                print(proto, subset, stream, flush=True)
    write_csv(OUT / "v0_reproduction.csv", [r for r in rows if r["method"] == "V0"])
    write_csv(OUT / "v2_dinov3_mean_b2.csv", [r for r in rows if r["method"] == "V2"])
    write_csv(OUT / "oracle_backbone_comparison.csv", [r for r in rows if r["method"] in ("O0", "O1")])

    # offline oracle-K + geometry on Pure novel tracks (private diagnostic)
    gt_pure = load_gt("pure")
    private = {g["sample_id"]: g for g in gt_pure}
    novel_ids = [s for s in private if private[s]["protocol_role"] == "novel"]
    novel_labels = {s: private[s]["ground_truth_category_id"] for s in novel_ids}
    n_cats = len(set(novel_labels.values()))
    off_rows = []
    for name, feats in (("dinov2_single", d2_single), ("dinov2_mean", d2_mean),
                        ("dinov3_single", d3_single), ("dinov3_mean", d3_mean)):
        ids = [s for s in novel_ids if s in feats]
        res = offline_oracle_k(feats, ids, novel_labels, min(n_cats, len(ids)))
        off_rows.append({"representation": name, "novel_tracks": len(ids),
                         "oracle_k": min(n_cats, len(ids)), **res})
        print("offline", name, res, flush=True)
    write_csv(OUT / "offline_representation.csv", off_rows)
    geo_rows = []
    for name, feats in (("dinov2_mean", d2_mean), ("dinov3_mean", d3_mean)):
        ids = [s for s in private if s in feats]
        g = geometry(feats, gt_pure, ids)
        geo_rows.append({"representation": name, **g})
        print("geometry", name, g, flush=True)
    write_csv(OUT / "geometry_diagnostics.csv", geo_rows)

    # summary + gate
    def agg(method, proto="pure", subset="full"):
        vals = [r for r in rows if r["method"] == method and r["protocol"] == proto
                and r["subset"] == subset and r["seed"] in STREAMS[1:]]
        out = {}
        for k in ("all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                  "conditional_novel_acc", "novel_only_nmi", "novel_only_ari",
                  "predicted_novel_count", "novel_count_abs_error"):
            v = [float(r[k]) for r in vals]
            out[k] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0}
        return out
    v0 = agg("V0"); v2 = agg("V2")
    summary_rows = []
    for method in ("V0", "V2", "O0", "O1"):
        for proto in ("pure", "ov_assisted"):
            for subset in (SUBSETS if proto == "pure" else ("full",)):
                a = agg(method, proto, subset)
                row = {"method": method, "protocol": proto, "subset": subset}
                for k, v in a.items():
                    row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
                summary_rows.append(row)
    write_csv(OUT / "backbone_summary.csv", summary_rows)
    write_csv(OUT / "final_summary.csv", summary_rows)

    # gate
    c = {}
    c["known_ge_v0_minus_0.03"] = v2["overall_known_acc"]["mean"] >= v0["overall_known_acc"]["mean"] - 0.03
    c["route_ge_v0_plus_0.02"] = v2["route_aware_novel_acc"]["mean"] >= v0["route_aware_novel_acc"]["mean"] + 0.02
    c["cond_ge_v0_minus_0.01"] = v2["conditional_novel_acc"]["mean"] >= v0["conditional_novel_acc"]["mean"] - 0.01
    c["nmi_ge_v0_minus_0.01"] = v2["novel_only_nmi"]["mean"] >= v0["novel_only_nmi"]["mean"] - 0.01
    c["ari_ge_v0_minus_0.02"] = v2["novel_only_ari"]["mean"] >= v0["novel_only_ari"]["mean"] - 0.02
    c["count_error_le_90"] = v2["novel_count_abs_error"]["mean"] <= 90
    non_degenerate = all(c.values())
    gain = sum([
        v2["route_aware_novel_acc"]["mean"] >= v0["route_aware_novel_acc"]["mean"] + 0.03,
        v2["conditional_novel_acc"]["mean"] >= v0["conditional_novel_acc"]["mean"] + 0.03,
        v2["novel_only_nmi"]["mean"] >= v0["novel_only_nmi"]["mean"] + 0.01,
        v2["novel_only_ari"]["mean"] >= v0["novel_only_ari"]["mean"] + 0.03,
        v2["overall_known_acc"]["mean"] >= v0["overall_known_acc"]["mean"] + 0.05,
    ])
    stability = (
        len([r for r in rows if r["method"] == "V2" and r["protocol"] == "pure" and r["subset"] == "full" and r["seed"] in STREAMS[1:]]) == 3
        and v2["predicted_novel_count"]["mean"] <= 2.0 * v0["predicted_novel_count"]["mean"]
    )
    passed = non_degenerate and gain >= 2 and stability
    gate = {
        "status": "PASS_DINOV3_BACKBONE" if passed else "NO_CLEAR_DINOV3_GAIN",
        "continue_transformer": bool(passed),
        "criteria": c,
        "gain_count": gain,
        "non_degenerate": non_degenerate,
        "stability": stability,
        "v0": {k: v["mean"] for k, v in v0.items()},
        "v2": {k: v["mean"] for k, v in v2.items()},
    }
    (RUNS / "backbone_gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))


def build_protos(feats, labels):
    from src.ocd_v2.common import build_prototypes
    return build_prototypes(feats, labels, set(labels.values()))


if __name__ == "__main__":
    main()
