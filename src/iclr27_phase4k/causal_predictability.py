"""Phase 4K time-conditioned causal predictability audit.

For every novel prototype we evaluate whether online-available evidence
available at a *past* checkpoint (strictly no future events, no GT)
separates prototypes whose retrospective outcome is USEFUL from those
whose outcome is POLLUTING.

Checkpoints:
  birth + {1,2,4,8,16} frames (absolute event-clock within/beyond the
  birth video), after 2 supports, after 4 supports, after first
  cross-track reuse.

Models are deliberately simple: standardized logistic regression and a
depth-3 decision tree, both 5-fold stratified cross-validated.  GT enters
only through the retrospective outcome label, never as a feature.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4k" / "audit"

FEATURES = [
    "age_frames", "support", "n_updates", "n_reuses",
    "same_track_updates", "cross_track_updates",
    "distinct_physical_tracks", "distinct_videos",
    "cross_track_reuses", "cross_track_agreement",
    "mean_compat", "max_compat", "dispersion", "drift",
    "mean_p_known", "mean_score", "assoc_helpful",
    "assoc_harmful", "assoc_net",
    "birth_p_known", "birth_known_margin", "birth_novel_conf",
    "birth_det_score", "birth_track_age",
]


def load_gt_join_rows(prov_root):
    """(video, frame, track) -> semantic log row (for p_known / score)."""
    out = {}
    for p in sorted((prov_root / "semantic_logs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(int(r["video_id"]), int(r["frame_id"]),
                 int(r["physical_track_id"]))] = r
    return out


def load_assoc_rows():
    p = OUT_ROOT / "association_interventions.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["j0", "j1b", "m1"])
    args = ap.parse_args()
    tag = args.tag
    prov_root = ROOT / "outputs" / "iclr27_phase4k" / "audit" / \
        f"prov_{tag}"

    events = []
    for line in (prov_root / f"prototype_event_log_{tag}.jsonl").read_text() \
            .splitlines():
        if line.strip():
            events.append(json.loads(line))
    embs = {}
    pz = prov_root / f"embeddings_{tag}.npz"
    if pz.exists():
        z = np.load(pz)["embeddings"]
        embs = {i: np.asarray(x, dtype=np.float32) for i, x in enumerate(z)}
    track_rows = load_gt_join_rows(prov_root)
    assoc = load_assoc_rows()

    # absolute event clock: video processing order (first-seen) * 1e6 + frame
    vorder = {}
    for e in events:
        vorder.setdefault(int(e["video_id"]), len(vorder))
    for e in events:
        e["_abs"] = vorder[int(e["video_id"])] * 1_000_000 + int(e["frame_id"])
    events.sort(key=lambda e: (e["_abs"], e["sem_id"]))

    protos = {}
    for e in events:
        protos.setdefault(int(e["sem_id"]), []).append(e)

    assoc_by_proto = defaultdict(list)
    for r in assoc:
        pid = r.get("prototype_id")
        if pid not in ("", None):
            try:
                assoc_by_proto[int(pid)].append(
                    (int(r["frame_id"]), int(vorder.get(
                        int(r["video_id"]), -1)), r["effect"]))
            except (ValueError, KeyError):
                pass

    def ev_abs(ev):
        return vorder[int(ev["video_id"])] * 1_000_000 + int(ev["frame_id"])

    def feats(sem, upto_abs, label):
        es = [e for e in protos[sem] if ev_abs(e) <= upto_abs]
        es.sort(key=lambda e: ev_abs(e))
        b = next((e for e in es if e["kind"] == "birth"), None)
        us = [e for e in es if e["kind"] == "update"]
        rs = [e for e in es if e["kind"] == "reuse"]
        if b is None:
            return None
        creator = (int(b["video_id"]), int(b["track_key"][1]))
        cross_tracks = sorted({(int(e["video_id"]), int(e["track_key"][1]))
                               for e in us + rs})
        cross_evs = [e for e in us + rs
                     if (int(e["video_id"]), int(e["track_key"][1]))
                     != creator]
        same_up = sum(1 for e in us if int(e["same_track"]))
        cross_up = len(us) - same_up
        zidx = [b["z_idx"]] + [e["z_idx"] for e in us]
        zidx = [i for i in zidx if i >= 0]
        Z = np.stack([embs[i] for i in zidx]) if zidx and all(
            i in embs for i in zidx) else None
        dispersion = 0.0
        drift = 1.0
        if Z is not None and len(Z) >= 2:
            Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
            cos = Zn @ Zn.T
            dispersion = float(1.0 - cos[np.triu_indices(len(Z), 1)].mean())
            drift = float(Zn[0] @ Zn[-1])
        compats = [float(e["compat_best"]) for e in us] + \
            [float(e["compat"]) for e in rs]
        scores, pks = [], []
        for e in us + rs:
            r = track_rows.get((int(e["video_id"]), int(e["frame_id"]),
                                int(e["track_key"][1])))
            if r is not None:
                scores.append(float(r.get("score", 0.0)))
                pks.append(float(r.get("p_known", 0.5)))
        a = assoc_by_proto.get(sem, [])
        vid_order = vorder.get(int(b["video_id"]), -1)
        a_upto = [x for x in a if x[1] * 1_000_000 + x[0] <= upto_abs]
        helpful = sum(1 for x in a_upto if x[2] == "helpful")
        harmful = sum(1 for x in a_upto if x[2] == "harmful")
        support = max([int(b["support_after"])] +
                      [int(e["support_after"]) for e in us], default=0)
        cross_agr = float(np.mean([
            float(e.get("compat_best", e.get("compat", 0.0)))
            for e in cross_evs])) if cross_evs else -1.0
        return {
            "age_frames": int(upto_abs - ev_abs(b)),
            "support": support, "n_updates": len(us), "n_reuses": len(rs),
            "same_track_updates": same_up, "cross_track_updates": cross_up,
            "distinct_physical_tracks": len(cross_tracks),
            "distinct_videos": len({int(e["video_id"]) for e in us + rs}),
            "cross_track_reuses": len(cross_evs),
            "cross_track_agreement": cross_agr,
            "mean_compat": float(np.mean(compats)) if compats else 0.0,
            "max_compat": float(np.max(compats)) if compats else 0.0,
            "dispersion": dispersion, "drift": drift,
            "mean_p_known": float(np.mean(pks)) if pks else 0.5,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "assoc_helpful": helpful, "assoc_harmful": harmful,
            "assoc_net": helpful - harmful,
            "birth_p_known": float(b["p_known"]),
            "birth_known_margin": float(b["known_margin"]),
            "birth_novel_conf": float(b["novel_conf"]),
            "birth_det_score": float(b["det_score"]),
            "birth_track_age": int(b["track_age"]),
            "_label": label,
        }

    # retrospective labels from the provenance CSV (diagnostic-only)
    labels = {}
    with open(OUT_ROOT / f"prototype_provenance_{tag}.csv") as f:
        for r in csv.DictReader(f):
            labels[int(r["sem_id"])] = r["outcome_group"]

    checkpoints = []
    for sem, es in protos.items():
        b = next((e for e in es if e["kind"] == "birth"), None)
        if b is None:
            continue
        label = labels.get(sem)
        if label not in ("USEFUL", "POLLUTING"):
            continue
        birth_abs = ev_abs(b)
        for delta in (1, 2, 4, 8, 16):
            checkpoints.append(("birth+%df" % delta, birth_abs + delta,
                                sem, label))
        # after supports
        sup_events = sorted(es, key=ev_abs)
        for target in (2, 4):
            ev = next((e for e in sup_events
                       if int(e.get("support_after", 0)) >= target), None)
            if ev is not None:
                checkpoints.append(("support%d" % target, ev_abs(ev), sem,
                                    label))
        # first cross-track reuse / update
        creator = (int(b["video_id"]), int(b["track_key"][1]))
        cev = next((e for e in es if e["kind"] in ("reuse", "update")
                    and (int(e["video_id"]), int(e["track_key"][1]))
                    != creator), None)
        if cev is not None:
            checkpoints.append(("first_cross_track", ev_abs(cev), sem, label))

    rows = []
    for cp_name, upto, sem, label in checkpoints:
        f = feats(sem, upto, label)
        if f is not None:
            f["_checkpoint"] = cp_name
            f["_sem_id"] = sem
            rows.append(f)

    # feature CSV (transparency)
    feat_fields = FEATURES + ["_checkpoint", "_sem_id", "_label"]
    with open(OUT_ROOT / "causal_checkpoint_features.csv", "w",
              newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=feat_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in feat_fields})
    with open(OUT_ROOT / f"causal_checkpoint_features_{tag}.csv", "w",
              newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=feat_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in feat_fields})

    # metrics per checkpoint
    metrics = []
    feature_auroc_rows = []
    for cp in sorted({r["_checkpoint"] for r in rows}):
        sub = [r for r in rows if r["_checkpoint"] == cp]
        y = np.array([1 if r["_label"] == "USEFUL" else 0 for r in sub])
        if len(set(y)) < 2 or len(y) < 8:
            metrics.append({
                "tag": tag, "checkpoint": cp, "n": len(sub),
                "n_useful": int(y.sum()),
                "n_polluting": int((1 - y).sum()),
                "auroc_logistic": "", "auprc_logistic": "",
                "auroc_tree": "",
                "top_single_feature": "", "top_single_auroc": "",
            })
            continue
        X = np.array([[r[f] for f in FEATURES] for r in sub],
                     dtype=float)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        logi = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=1.0))
        try:
            yp = cross_val_predict(logi, X, y, cv=skf, method="predict_proba")
            auroc = roc_auc_score(y, yp[:, 1])
            auprc = average_precision_score(y, yp[:, 1])
        except ValueError:
            auroc = auprc = float("nan")
        tree = DecisionTreeClassifier(max_depth=3, random_state=7)
        try:
            yt = cross_val_predict(tree, X, y, cv=skf,
                                   method="predict_proba")
            auroc_t = roc_auc_score(y, yt[:, 1])
        except ValueError:
            auroc_t = float("nan")
        # single-feature logistic AUROC (leave-one-out per feature)
        best_f, best_a = None, -1.0
        for i, f in enumerate(FEATURES):
            Xi = X[:, [i]]
            try:
                ya = cross_val_predict(logi, Xi, y, cv=skf,
                                       method="predict_proba")[:, 1]
                va = roc_auc_score(y, ya)
            except ValueError:
                continue
            feature_auroc_rows.append({
                "tag": tag, "checkpoint": cp, "feature": f,
                "auroc": round(float(va), 4),
                "n": len(y), "n_useful": int(y.sum()),
                "n_polluting": int((1 - y).sum()),
            })
            if va > best_a:
                best_a, best_f = va, f
        metrics.append({
            "tag": tag, "checkpoint": cp, "n": len(y),
            "n_useful": int(y.sum()), "n_polluting": int((1 - y).sum()),
            "auroc_logistic": round(float(auroc), 4),
            "auprc_logistic": round(float(auprc), 4),
            "auroc_tree": round(float(auroc_t), 4),
            "top_single_feature": best_f or "",
            "top_single_auroc": round(float(best_a), 4) if best_f else "",
        })

    def append_csv(name, rows):
        p = OUT_ROOT / name
        old = []
        if p.exists():
            with open(p) as f:
                old = [r for r in csv.DictReader(f) if r.get("tag") != tag]
        with open(p, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())
                               if rows else ["tag"])
            w.writeheader()
            w.writerows(old + rows)
        with open(OUT_ROOT / name.replace(".csv", f"_{tag}.csv"), "w",
                  newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())
                               if rows else ["tag"])
            w.writeheader()
            w.writerows(rows)

    append_csv("causal_predictability.csv", metrics)
    append_csv("causal_feature_auroc.csv", feature_auroc_rows)
    print(json.dumps(metrics, indent=1))
    print("CAUSAL_PREDICTABILITY_DONE", tag)


if __name__ == "__main__":
    main()
