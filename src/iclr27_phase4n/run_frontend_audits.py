"""Phase 4N frontend + gate audits from the detection population."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4n" / "audit"


def num(v, default=-1.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    u = ranks[y].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def auprc(y, s):
    """Average precision (area under PR curve)."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    # integrate precision over recall (VOC-style AP)
    ap = 0.0
    prev = 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev) * p
        prev = r
    return float(ap)


def quantiles(vals):
    if not vals:
        return [""] * 7
    a = np.asarray(vals, dtype=float)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return [round(float(a.mean()), 4), round(float(a.std()), 4),
            round(float(q[0]), 4), round(float(q[1]), 4),
            round(float(q[2]), 4), round(float(q[3]), 4),
            round(float(q[4]), 4)]


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = math.sqrt(((len(a) - 1) * a.std() ** 2 +
                    (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
    if sp == 0:
        return None
    return float((a.mean() - b.mean()) / sp)


def ks(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def feat_sets():
    return {
        "D0_detector": ["score"],
        "D1_tracking": ["track_age", "assoc_appearance_best",
                        "assoc_final_best", "app_norm", "mask_frac"],
        "D2_semantic": ["p_known", "best_known", "best_novel", "z_norm",
                        "dino_norm", "known_margin"],
        "D3_det_track": ["score", "track_age", "assoc_appearance_best",
                         "assoc_final_best", "app_norm", "mask_frac"],
        "D4_all": ["score", "track_age", "assoc_appearance_best",
                   "assoc_final_best", "app_norm", "mask_frac", "p_known",
                   "best_known", "best_novel", "z_norm", "dino_norm",
                   "known_margin"],
    }


def feats_for(row, cols):
    out = []
    for c in cols:
        out.append(num(row.get(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", nargs=2, required=True,
                    help="dev csv heldout csv")
    ap.add_argument("--out", type=Path, default=AUDIT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pop = {}
    for mode, p in zip(("dev", "heldout"), args.population):
        pop[mode] = list(csv.DictReader(open(p)))

    # ---- detection population (combined) ----
    all_rows = []
    for mode, rows in pop.items():
        for r in rows:
            r["mode"] = mode
            all_rows.append(r)
    with open(args.out / "detection_population.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # ---- detector score distributions ----
    dist_rows = []
    for mode, rows in pop.items():
        by_role = defaultdict(list)
        for r in rows:
            by_role[r["gt_role"]].append(num(r["score"]))
        for role, vals in by_role.items():
            q = quantiles(vals)
            dist_rows.append({
                "mode": mode, "role": role, "n": len(vals),
                "mean": q[0], "std": q[1], "min": q[2], "q25": q[3],
                "median": q[4], "q75": q[5], "max": q[6],
                "AUROC": "", "AUPRC": "",
            })
        valid = [num(r["score"]) for r in rows if r["gt_role"] != "fp"]
        fp = [num(r["score"]) for r in rows if r["gt_role"] == "fp"]
        novel = [num(r["score"]) for r in rows
                 if r["gt_role"] == "novel"]
        dist_rows.append({
            "mode": mode, "role": "valid-vs-FP_AUROC",
            "n": len(valid) + len(fp),
            "mean": "", "std": "", "min": "", "q25": "", "median": "",
            "q75": "", "max": "",
            "AUROC": round(auroc([1] * len(valid) + [0] * len(fp),
                                 valid + fp), 4) if valid and fp else "",
            "AUPRC": round(auprc([1] * len(valid) + [0] * len(fp),
                                 valid + fp), 4) if valid and fp else "",
        })
        dist_rows.append({
            "mode": mode, "role": "novel-vs-FP_AUROC",
            "n": len(novel) + len(fp),
            "mean": "", "std": "", "min": "", "q25": "", "median": "",
            "q75": "", "max": "",
            "AUROC": round(auroc([1] * len(novel) + [0] * len(fp),
                                 novel + fp), 4) if novel and fp else "",
            "AUPRC": round(auprc([1] * len(novel) + [0] * len(fp),
                                 novel + fp), 4) if novel and fp else "",
        })
    with open(args.out / "detector_score_distributions.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)

    # ---- detector threshold oracle curve ----
    thr_rows = []
    for mode, rows in pop.items():
        scores = sorted({num(r["score"]) for r in rows}, reverse=True)
        # percentile-based thresholds
        arr = np.asarray([num(r["score"]) for r in rows])
        thrs = np.percentile(arr, [10, 20, 30, 40, 50, 60, 70, 80, 90,
                                   95, 99])
        for thr in sorted(set(round(float(t), 4) for t in thrs)):
            keep = [r for r in rows if num(r["score"]) >= thr]
            known = sum(1 for r in keep if r["gt_role"] == "known")
            novel = sum(1 for r in keep if r["gt_role"] == "novel")
            fp = sum(1 for r in keep if r["gt_role"] == "fp")
            tot_known = sum(1 for r in rows if r["gt_role"] == "known")
            tot_novel = sum(1 for r in rows if r["gt_role"] == "novel")
            thr_rows.append({
                "mode": mode, "threshold": thr,
                "valid_known_recall": round(known / max(tot_known, 1), 4),
                "valid_novel_recall": round(novel / max(tot_novel, 1), 4),
                "fp_count": fp, "valid_precision": round(
                    (known + novel) / max(len(keep), 1), 4),
                "fp_share_in_kept": round(fp / max(len(keep), 1), 4),
            })
    with open(args.out / "detector_threshold_curve.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(thr_rows[0].keys()))
        w.writeheader()
        w.writerows(thr_rows)

    # ---- persistent FP features ----
    pers_rows = []
    for mode, rows in pop.items():
        tracks = defaultdict(list)
        for r in rows:
            tid = r.get("track_id", "")
            if tid not in ("", None):
                tracks[(r["video_id"], tid)].append(r)
        for (vid, tid), rs in tracks.items():
            roles = [r["gt_role"] for r in rs]
            role = Counter(roles).most_common(1)[0][0]
            if role == "fp":
                role = "FP"
            elif role == "known":
                role = "KNOWN"
            else:
                role = "NOVEL"
            ages = [num(r.get("track_age")) for r in rs]
            pers_rows.append({
                "mode": mode, "video_id": vid, "track_id": tid,
                "role": role, "frames": len(rs),
                "max_age": int(max(ages)) if ages else "",
                "mean_score": round(np.mean([num(r["score"])
                                             for r in rs]), 4),
                "mean_mask_frac": round(np.mean([num(r.get("mask_frac"))
                                                 for r in rs]), 4),
                "mean_app_norm": round(np.mean([num(r.get("app_norm"))
                                                for r in rs]), 4),
                "mean_p_known": round(np.mean([num(r.get("p_known"))
                                               for r in rs]), 4),
                "mean_best_known": round(np.mean(
                    [num(r.get("best_known")) for r in rs]), 4),
                "mean_best_novel": round(np.mean(
                    [num(r.get("best_novel")) for r in rs]), 4),
                "mean_assoc": round(np.mean(
                    [num(r.get("assoc_final_best")) for r in rs]), 4),
            })
    with open(args.out / "persistent_fp_features.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pers_rows[0].keys()))
        w.writeheader()
        w.writerows(pers_rows)

    # ---- gate score datasets + shift ----
    gate_rows = {"dev": [], "heldout": []}
    for mode, rows in pop.items():
        for r in rows:
            if r["gt_role"] not in ("known", "novel"):
                continue
            if r.get("p_known") in ("", None):
                continue
            gate_rows[mode].append({
                "mode": mode, "gt_role": r["gt_role"],
                "p_known": num(r["p_known"]),
                "gate_logit": num(r.get("gate_logit")),
                "best_known": num(r["best_known"]),
                "best_novel": num(r["best_novel"]),
                "known_margin": num(r.get("known_margin")),
                "dino_norm": num(r.get("dino_norm")),
                "z_norm": num(r.get("z_norm")),
                "track_age": num(r.get("track_age")),
                "score": num(r["score"]),
            })
    for mode in ("dev", "heldout"):
        with open(args.out / f"gate_scores_{mode}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(
                gate_rows[mode][0].keys()))
            w.writeheader()
            w.writerows(gate_rows[mode])

    shift_rows = []
    feats = ["p_known", "gate_logit", "best_known", "best_novel",
             "known_margin", "dino_norm", "z_norm", "track_age", "score"]
    for role in ("known", "novel"):
        for f in feats:
            a = [r[f] for r in gate_rows["dev"] if r["gt_role"] == role]
            b = [r[f] for r in gate_rows["heldout"]
                 if r["gt_role"] == role]
            shift_rows.append({
                "role": role, "feature": f, "mode": "",
                "dev_mean": round(float(np.mean(a)), 4) if a else "",
                "dev_std": round(float(np.std(a)), 4) if a else "",
                "held_mean": round(float(np.mean(b)), 4) if b else "",
                "held_std": round(float(np.std(b)), 4) if b else "",
                "ks": (round(ks(a, b), 4)
                       if a and b and ks(a, b) is not None else ""),
                "cohens_d": (round(cohens_d(a, b), 4)
                             if a and b and cohens_d(a, b) is not None
                             else ""),
                "AUROC": "", "AUPRC": "",
            })
    # known-vs-novel separability per split
    for mode in ("dev", "heldout"):
        rows = gate_rows[mode]
        y = [1 if r["gt_role"] == "known" else 0 for r in rows]
        for f in ("gate_logit", "p_known", "best_known", "known_margin"):
            s = [r[f] for r in rows]
            shift_rows.append({
                "role": "known-vs-novel", "feature": f, "mode": mode,
                "dev_mean": "", "dev_std": "", "held_mean": "",
                "held_std": "", "ks": "", "cohens_d": "",
                "AUROC": (round(auroc(y, s), 4)
                          if len(set(s)) > 1 and
                          auroc(y, s) is not None else ""),
                "AUPRC": (round(auprc(y, s), 4)
                          if len(set(s)) > 1 else ""),
            })
    with open(args.out / "gate_shift_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
        w.writeheader()
        w.writerows(shift_rows)

    # ---- track-age gate shift ----
    def age_bucket(a):
        if a <= 1:
            return "1"
        if a == 2:
            return "2"
        if a <= 4:
            return "3-4"
        if a <= 8:
            return "5-8"
        if a <= 16:
            return "9-16"
        return "17+"

    age_rows = []
    for mode, rows in pop.items():
        by_age = defaultdict(list)
        for r in rows:
            if r["gt_role"] not in ("known", "novel"):
                continue
            if r.get("p_known") in ("", None):
                continue
            a = age_bucket(int(num(r.get("track_age"))))
            by_age[a].append(r)
        for bucket in ("1", "2", "3-4", "5-8", "9-16", "17+"):
            rs = by_age.get(bucket, [])
            if not rs:
                continue
            known = [r for r in rs if r["gt_role"] == "known"]
            novel = [r for r in rs if r["gt_role"] == "novel"]
            k2n = sum(1 for r in known if num(r["p_known"]) < 0.30)
            n2k = sum(1 for r in novel if num(r["p_known"]) >= 0.30)
            correct = (len(known) - k2n) + (len(novel) - n2k)
            age_rows.append({
                "mode": mode, "age_bucket": bucket,
                "known_n": len(known), "novel_n": len(novel),
                "routing_acc": round(correct / max(len(rs), 1), 4),
                "k2n": round(k2n / max(len(known), 1), 4),
                "n2k": round(n2k / max(len(novel), 1), 4),
                "mean_p_known": round(float(np.mean(
                    [num(r["p_known"]) for r in rs])), 4),
            })
    with open(args.out / "gate_shift_by_age.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(age_rows[0].keys()))
        w.writeheader()
        w.writerows(age_rows)

    # ---- video-level shift ----
    vid_rows = []
    for mode, rows in pop.items():
        by_vid = defaultdict(list)
        for r in rows:
            by_vid[r["video_id"]].append(r)
        for vid, rs in by_vid.items():
            roles = Counter(r["gt_role"] for r in rs)
            matched = [r for r in rs if r["gt_role"] in ("known", "novel")
                       and r.get("p_known") not in ("", None)]
            known = [r for r in matched if r["gt_role"] == "known"]
            novel = [r for r in matched if r["gt_role"] == "novel"]
            k2n = sum(1 for r in known if num(r["p_known"]) < 0.30)
            n2k = sum(1 for r in novel if num(r["p_known"]) >= 0.30)
            correct = (len(known) - k2n) + (len(novel) - n2k)
            vid_rows.append({
                "mode": mode, "video_id": vid,
                "known_count": roles.get("known", 0),
                "novel_count": roles.get("novel", 0),
                "fp_count": roles.get("fp", 0),
                "routing_acc": round(correct / max(len(matched), 1), 4),
                "k2n": round(k2n / max(len(known), 1), 4),
                "n2k": round(n2k / max(len(novel), 1), 4),
                "mean_p_known": round(float(np.mean(
                    [num(r["p_known"]) for r in matched])), 4)
                if matched else "",
                "mean_detector_score": round(float(np.mean(
                    [num(r["score"]) for r in rs])), 4),
            })
    with open(args.out / "gate_shift_by_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vid_rows[0].keys()))
        w.writeheader()
        w.writerows(vid_rows)

    # ---- detector x gate interaction ----
    int_rows = []
    for mode, rows in pop.items():
        buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                   (0.8, 0.9), (0.9, 1.01)]
        for lo, hi in buckets:
            rs = [r for r in rows if lo <= num(r["score"]) < hi]
            if not rs:
                continue
            known = [r for r in rs if r["gt_role"] == "known"]
            novel = [r for r in rs if r["gt_role"] == "novel"]
            fp = [r for r in rs if r["gt_role"] == "fp"]
            novel_route = [r for r in rs if num(r.get("p_known")) < 0.30]
            int_rows.append({
                "mode": mode, "score_bucket": f"{lo:.1f}-{hi:.1f}",
                "n": len(rs), "known": len(known), "novel": len(novel),
                "fp": len(fp),
                "fraction_routed_novel": round(
                    len(novel_route) / max(len(rs), 1), 4),
                "mean_p_known": round(float(np.mean(
                    [num(r.get("p_known")) for r in rs])), 4),
                "fp_routed_novel": round(
                    sum(1 for r in fp if num(r.get("p_known")) < 0.30) /
                    max(len(fp), 1), 4) if fp else "",
                "novel_routed_known": round(
                    sum(1 for r in novel if num(r.get("p_known")) >= 0.30) /
                    max(len(novel), 1), 4) if novel else "",
            })
    with open(args.out / "detector_gate_interaction.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(int_rows[0].keys()))
        w.writeheader()
        w.writerows(int_rows)

    # ---- validity predictability (logistic; dev fit -> dev+heldout) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pred_rows = []
    for name, cols in feat_sets().items():
        def matrix(rows):
            X = []
            y = []
            for r in rows:
                if r["gt_role"] in ("", None):
                    continue
                X.append(feats_for(r, cols))
                y.append(1 if r["gt_role"] != "fp" else 0)
            return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        Xd, yd = matrix(pop["dev"])
        Xh, yh = matrix(pop["heldout"])
        if len(set(yd)) < 2 or len(set(yh)) < 2:
            continue
        sc = StandardScaler().fit(Xd)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xd), yd)
        for mode, X, y in (("dev", Xd, yd), ("heldout", Xh, yh)):
            s = clf.predict_proba(sc.transform(X))[:, 1]
            rows = pop[mode]
            n_novel = int(sum(1 for r in rows if r["gt_role"] == "novel"))
            n_fp = int(sum(1 for r in rows if r["gt_role"] == "fp"))
            order = np.argsort(-s)
            recall_at_p3 = None
            tp = fp = novel_tp = 0
            for i in order:
                if y[i] == 1:
                    tp += 1
                    if rows[i]["gt_role"] == "novel":
                        novel_tp += 1
                else:
                    fp += 1
                if tp + fp > 0 and tp / (tp + fp) >= 0.3 and \
                        recall_at_p3 is None:
                    recall_at_p3 = novel_tp / max(n_novel, 1)
            fp_rej = None
            tp = fp = novel_tp = 0
            for i in order:
                if y[i] == 1:
                    tp += 1
                    if rows[i]["gt_role"] == "novel":
                        novel_tp += 1
                else:
                    fp += 1
                if n_novel and novel_tp / n_novel >= 0.7:
                    fp_rej = 1.0 - fp / max(n_fp, 1)
                    break
            pred_rows.append({
                "features": name, "mode": mode,
                "AUROC": round(auroc(y, s), 4),
                "AUPRC": round(auprc(y, s), 4),
                "valid_novel_recall_at_prec0.3": (
                    round(recall_at_p3, 4) if recall_at_p3 else ""),
                "fp_rejection_at_novel_recall0.7": (
                    round(fp_rej, 4) if fp_rej is not None else ""),
            })
    with open(args.out / "validity_predictability.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    print("FRONTEND_AUDITS_DONE")


if __name__ == "__main__":
    main()
