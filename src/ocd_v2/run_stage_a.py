#!/usr/bin/env python3
"""Architecture 1.5 Stage A: GT-track OCD-v2 experiments."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_predictions, hungarian_acc
from src.ocd_v2.common import (
    build_prototypes,
    evaluate_rows,
    load_mean_features,
    load_stream,
    load_train_known,
    load_train_known_meta,
    load_val_labels,
    proxy_split,
    row_meta,
    seed_label,
    stream_names,
)
from src.ocd_v2.gates import KnownMatcher, build_gate, calibrate_gate
from src.ocd_v2.online_clustering import (
    CandidateBuffer,
    MultiPrototypeMemory,
    OnlineDPMeans,
    OnlineSphericalKMeans,
)

OUT = PROJECT_ROOT / "outputs" / "arch1_5" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "arch1_5"


def simulate_b2(rows, feats, known_protos, thr=0.45):
    novel = {}
    counts = {}
    next_id = 100000
    preds = []
    log = []
    for i, row in enumerate(rows):
        x = feats[row["sample_id"]]
        best_k, best_s = None, -1.0
        for cid, p in known_protos.items():
            s = float(np.dot(x, p))
            if s > best_s:
                best_k, best_s = cid, s
        if best_s >= thr:
            vid = best_k
        else:
            best_n, best_p = -1.0, None
            for cid, p in novel.items():
                s = float(np.dot(x, p))
                if s > best_n:
                    best_n, best_p = s, cid
            if best_n >= thr:
                vid = best_p
                novel[vid] = (novel[vid] * counts[vid] + x) / (counts[vid] + 1)
                novel[vid] /= np.linalg.norm(novel[vid]) + 1e-12
                counts[vid] += 1
            else:
                vid = next_id
                next_id += 1
                novel[vid] = x.copy()
                counts[vid] = 1
        preds.append(vid)
        log.append({"stream_order": i, "sample_id": row["sample_id"], "virtual_category_id": vid})
    return np.array(preds), log


def run_b2_repro(encoder="dinov2", subset="full"):
    feats, labels = load_train_known(encoder)
    protos = build_prototypes(feats, labels, set(labels.values()))
    val_feats = load_mean_features(encoder, "gt_tracks_mean")
    private = load_val_labels()
    rows_out = []
    for sn in stream_names():
        rows = load_stream(sn)
        preds, log = simulate_b2(rows, val_feats, protos)
        y_true = np.array([private[r["sample_id"]]["ground_truth_category_id"] for r in rows])
        known_mask = np.array([private[r["sample_id"]]["is_known"] for r in rows])
        res = evaluate_rows(rows, y_true, preds, known_mask, subset, private)
        res["method"] = "b2_repro"
        res["seed"] = seed_label(sn)
        res["subset"] = subset
        rows_out.append(res)
        (RUNS / f"b2_repro_{seed_label(sn)}_{subset}.json").write_text(
            json.dumps(res, indent=1, default=str)
        )
    return rows_out


def calibrate_novel(method, dinov2_feats, labels):
    """Calibrate clustering thresholds on proxy-novel train-known tracks."""
    proxy_known, proxy_novel = proxy_split(labels, seed=1027)
    ids = sorted(s for s in labels if s in dinov2_feats and labels[s] in proxy_novel)
    X = np.stack([dinov2_feats[s] for s in ids])
    y = np.array([labels[s] for s in ids])
    rows = [{"sample_id": s} for s in ids]

    def acc_for(preds):
        # map proxy class ids to contiguous for Hungarian
        uniq = np.unique(preds)
        remap = {int(p): i for i, p in enumerate(uniq)}
        preds_c = np.array([remap[int(p)] for p in preds])
        return hungarian_acc(y, preds_c)[0]

    if method in ("spherical_kmeans", "dpmeans"):
        best = None
        for thr in np.arange(0.40, 0.86, 0.025):
            model = OnlineSphericalKMeans(attach_thr=float(thr))
            preds = np.array([model.predict_one(X[i], ids[i], i) for i in range(len(X))])
            a = acc_for(preds)
            if best is None or a > best[1]:
                best = ({"attach_thr": float(thr)}, a)
        return best
    if method == "candidate_buffer":
        best = None
        for attach in np.arange(0.50, 0.86, 0.025):
            create = max(0.30, attach - 0.15)
            model = CandidateBuffer(attach_thr=float(attach), create_thr=float(create))
            preds = np.array([model.predict_one(X[i], ids[i], i) for i in range(len(X))])
            a = acc_for(preds)
            if best is None or a > best[1]:
                best = ({"attach_thr": float(attach), "create_thr": float(create)}, a)
        return best
    if method == "ocd_v2":
        best = None
        for attach in np.arange(0.50, 0.86, 0.025):
            create = max(0.30, attach - 0.15)
            new_proto = max(0.40, attach - 0.05)
            model = MultiPrototypeMemory(
                attach_thr=float(attach),
                create_thr=float(create),
                new_proto_thr=float(new_proto),
                max_proto=4,
            )
            preds = np.array(
                [
                    model.predict_one(X[i], ids[i], i, num_frames=1, video_id=0)
                    for i in range(len(X))
                ]
            )
            a = acc_for(preds)
            if best is None or a > best[1]:
                best = ({"attach_thr": float(attach), "create_thr": float(create), "new_proto_thr": float(new_proto)}, a)
        return best
    raise ValueError(method)


def run_val(method, gate_type, gate_params, cluster_params, subset, encoder="dinov2", matcher_w=0.5):
    dino_feats, labels = load_train_known("dinov2")
    clip_feats, _ = load_train_known("clip")
    gate = build_gate(gate_type, clip_feats, labels, dino_feats, gate_params)
    matcher = KnownMatcher(
        build_prototypes(clip_feats, labels, set(labels.values())),
        build_prototypes(dino_feats, labels, set(labels.values())),
        w=matcher_w,
    )
    val_dino = load_mean_features("dinov2", "gt_tracks_mean")
    val_clip = load_mean_features("clip", "gt_tracks_mean")
    private = load_val_labels()
    rows_out = []
    for sn in stream_names():
        rows = load_stream(sn)
        preds = []
        log = []
        if method == "b2":
            model = None
        elif method == "spherical_kmeans":
            model = OnlineSphericalKMeans(**cluster_params)
        elif method == "dpmeans":
            model = OnlineDPMeans(**cluster_params)
        elif method == "candidate_buffer":
            model = CandidateBuffer(**cluster_params)
        elif method == "ocd_v2":
            model = MultiPrototypeMemory(**cluster_params)
        else:
            raise ValueError(method)

        for i, row in enumerate(rows):
            sid = row["sample_id"]
            is_k, kcid, score = gate.decide(val_clip[sid], val_dino[sid], row_meta(row))
            if is_k:
                vid, _ = matcher.classify(val_clip[sid], val_dino[sid])
                preds.append(vid)
            else:
                if method == "b2":
                    # placeholder, not used in run_val
                    vid = 0
                else:
                    vid = model.predict_one(
                        val_dino[sid], sid, i, num_frames=row.get("num_frames", 8), video_id=row["video_id"]
                    )
                preds.append(vid)
            log.append({"stream_order": i, "sample_id": sid, "virtual_category_id": vid})
        preds = np.array(preds)
        y_true = np.array([private[r["sample_id"]]["ground_truth_category_id"] for r in rows])
        known_mask = np.array([private[r["sample_id"]]["is_known"] for r in rows])
        res = evaluate_rows(rows, y_true, preds, known_mask, subset, private)
        res["method"] = method
        res["gate"] = gate_type
        res["seed"] = seed_label(sn)
        res["subset"] = subset
        res["known_false_absorption"] = int(
            ((~known_mask) & np.isin(preds, sorted(set(labels.values())))).sum()
        )
        if isinstance(model, (MultiPrototypeMemory, CandidateBuffer)):
            res["memory_stats"] = model.memory_stats()
        res["prediction_log"] = log
        rows_out.append(res)
        (RUNS / f"{method}_{gate_type}_{seed_label(sn)}_{subset}.json").write_text(
            json.dumps(res, indent=1, default=str)
        )
    return rows_out


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    compact_fields = [
        "method", "gate", "seed", "subset", "num_samples",
        "acc_all", "acc_known", "acc_novel", "nmi", "ari",
        "predicted_categories", "true_categories", "category_count_abs_error",
        "mean_fragmentation", "mean_purity", "merge_error",
        "duplicate_creation_rate", "mean_assignment_delay",
        "known_false_absorption", "memory_stats",
    ]
    if fieldnames is None:
        fieldnames = [f for f in compact_fields if any(f in r for r in rows)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            if "memory_stats" in out and isinstance(out["memory_stats"], dict):
                out["memory_stats"] = json.dumps(out["memory_stats"])
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--gates", default="clip,dino,dual,dual_lr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "b2"):
        rows = run_b2_repro("dinov2", "full")
        write_csv(OUT / "b2_reproduced.csv", rows)
        for r in rows:
            print(r["seed"], r["acc_all"], r["acc_novel"], r["predicted_categories"])

    if args.stage in ("all", "gates"):
        dino_feats, labels = load_train_known("dinov2")
        clip_feats, _ = load_train_known("clip")
        meta = load_train_known_meta()
        gate_rows = []
        calib = {}
        for gt in [g.strip() for g in args.gates.split(",") if g.strip()]:
            params, metrics, pk, pn = calibrate_gate(clip_feats, dino_feats, labels, gt, meta=meta)
            metrics.update({"gate": gt, "params": params})
            gate_rows.append(metrics)
            print(gt, params, metrics)
            for method in ("spherical_kmeans", "dpmeans", "candidate_buffer", "ocd_v2"):
                cp, ca = calibrate_novel(method, dino_feats, labels)
                calib.setdefault(gt, {})[method] = cp
                print("calib", method, cp, ca)
                gate_rows_all = []
                for subset in ("full", "repeated", "balanced"):
                    rows = run_val(method, gt, params, cp, subset)
                    gate_rows_all.extend(rows)
                write_csv(OUT / f"{method}_{gt}.csv", gate_rows_all)
        write_csv(OUT / "learned_gate.csv", gate_rows)
        old = {}
        old_path = RUNS / "calibrated_params.json"
        if old_path.exists():
            old = json.loads(old_path.read_text())
        old.update(calib)
        old_path.write_text(json.dumps(old, indent=2, default=str))

    if args.stage in ("all", "oracle"):
        # Oracle gate: route with GT is_known, cluster novel with calibrated OCD-v2
        dino_feats, labels = load_train_known("dinov2")
        cp, ca = calibrate_novel("ocd_v2", dino_feats, labels)
        (RUNS / "calibrated_params_oracle.json").write_text(
            json.dumps({"ocd_v2": cp}, indent=2, default=str)
        )
        val_dino = load_mean_features("dinov2", "gt_tracks_mean")
        private = load_val_labels()
        rows_out = []
        for subset in ("full", "repeated", "balanced"):
            for sn in stream_names():
                rows = load_stream(sn)
                model = MultiPrototypeMemory(**cp)
                preds = []
                for i, row in enumerate(rows):
                    sid = row["sample_id"]
                    if private[sid]["is_known"]:
                        preds.append(private[sid]["ground_truth_category_id"])
                    else:
                        preds.append(
                            model.predict_one(
                                val_dino[sid], sid, i, num_frames=row.get("num_frames", 8), video_id=row["video_id"]
                            )
                        )
                preds = np.array(preds)
                y_true = np.array([private[r["sample_id"]]["ground_truth_category_id"] for r in rows])
                known_mask = np.array([private[r["sample_id"]]["is_known"] for r in rows])
                res = evaluate_rows(rows, y_true, preds, known_mask, subset, private)
                res["method"] = "oracle_gate_ocdv2"
                res["seed"] = seed_label(sn)
                res["subset"] = subset
                res["memory_stats"] = model.memory_stats()
                rows_out.append(res)
        write_csv(OUT / "oracle_gate.csv", rows_out)
        for r in rows_out:
            if r["subset"] == "full" and r["seed"] == "main":
                print("oracle", r["acc_all"], r["acc_novel"], r["predicted_categories"])


if __name__ == "__main__":
    main()
