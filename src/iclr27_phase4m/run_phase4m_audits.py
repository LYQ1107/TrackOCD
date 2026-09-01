"""Phase 4M audit pipeline.

Runs on the corrected forced-decision datasets (prefix-P1 geometry,
validated against the online `compat` values) and produces:

  outputs/iclr27_phase4m/audit/ambiguity_features.csv
  outputs/iclr27_phase4m/audit/time_to_resolution.csv
  outputs/iclr27_phase4m/audit/deferral_pareto.csv
  outputs/iclr27_phase4m/audit/audit_summary.json
  docs/iclr27_phase4m/*.md

The retrospective oracle is a strict counterfactual replay: a deferred
track contributes no global memory writes at/after t (its own evidence
cannot be self-confirming), other tracks' events are held fixed, and each
horizon is evaluated at the exact causal position of that track's own
next decision event.  GT is used only for offline scoring.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PROV_ROOT = ROOT / "outputs" / "iclr27_phase4l" / "audit"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / \
    "detection_features"
Z_CACHE = ROOT / "outputs" / "iclr27_phase4m" / "audit" / "det_z_cache"
EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
TAO_JSON = EXPORT / "tao_subset" / "validation_20.json"
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
OUT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DOCS = ROOT / "docs" / "iclr27_phase4m"
NOVEL_UPDATE_RATE = 0.2
PREFIX_LEN = 8
PRUNE_GAP = 10
THETA = 0.6
TAGS = ("j1b", "b1", "b2")
IDENTITY_OUTCOMES = ("CORRECT_EXISTING", "WRONG_EXISTING",
                     "CORRECT_NEW", "OVERBIRTH")
ERROR_OUTCOMES = ("WRONG_EXISTING", "OVERBIRTH")


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def _num(v):
    if v in ("", None):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _maj(cats):
    cnt = Counter(c for c in cats if c is not None)
    return cnt.most_common(1)[0][0] if cnt else None


def load_gt():
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        b = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": int(ann["category_id"]),
            "role": "known" if int(ann["category_id"]) in known else "novel",
        })
    return out


def _iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def match_gt(gt, image_id, bbox):
    best, bi = None, 0.5
    for g in gt.get(image_id, []):
        v = _iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def load_decision_rows(tag):
    rows = list(csv.DictReader(open(OUT / f"identity_decisions_{tag}.csv")))
    for r in rows:
        for k in ("video_id", "frame_id", "track_id", "sem_id", "best_id",
                  "second_id", "n_gt_cat_mem", "support_causal",
                  "member_count", "proto_distinct_tracks", "near_count",
                  "rule_existing", "absolute_existing", "same_track"):
            r[k] = "" if r.get(k) in ("", None) else int(float(r[k]))
        for k in ("best_cos", "second_cos", "margin", "top1", "top2",
                  "top3", "top4", "top5", "local_entropy",
                  "member_mean_cos", "member_std_cos", "member_min_cos",
                  "member_max_cos", "query_zscore", "routing_confidence",
                  "decision_threshold", "best_known", "novel_minus_known",
                  "online_compat", "gt_cat_mem_cos"):
            r[k] = _num(r.get(k))
        r["det_gt_category"] = ("" if r.get("det_gt_category") in ("", None)
                                else int(r["det_gt_category"]))
        r["proto_majority_category"] = (
            "" if r.get("proto_majority_category") in ("", None)
            else int(r["proto_majority_category"]))
        r["gt_cat_mem_id"] = ("" if r.get("gt_cat_mem_id") in ("", None)
                              else int(r["gt_cat_mem_id"]))
    rows.sort(key=lambda r: (r["video_id"], r["frame_id"], r["sem_id"]))
    return rows


class TagData:
    def __init__(self, tag, gt):
        self.tag = tag
        self.gt = gt
        prov = PROV_ROOT / f"prov_dev_{tag}"
        self.rows = load_decision_rows(tag)
        self.track_rows = {}
        for p in sorted((prov / "semantic_logs").glob("*.jsonl")):
            vid = int(p.stem)
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                tid = r.get("physical_track_id")
                if tid is not None and tid >= 0:
                    self.track_rows[(vid, int(r["frame_id"]), int(tid))] = r
        vids = sorted({k[0] for k in self.track_rows})
        self.zs = {}
        self.fpos = {}
        for vid in vids:
            f = np.load(FEAT_ROOT / str(vid) / "feats.npz")
            fo = f["frame_orders"]
            self.zs[vid] = np.load(Z_CACHE / f"{vid}.npz")[
                "z"].astype(np.float32)
            self.fpos[vid] = {}
            for fid in np.unique(fo):
                self.fpos[vid][int(fid)] = np.where(fo == fid)[0]
        self.embs = {i: np.asarray(x, dtype=np.float32) for i, x in
                     enumerate(np.load(prov / f"embeddings_{tag}.npz")[
                         "embeddings"])}
        self.events = []
        for line in (prov / f"prototype_event_log_{tag}.jsonl").read_text() \
                .splitlines():
            if line.strip():
                self.events.append(json.loads(line))
        vorder = {}
        for e in self.events:
            vorder.setdefault(int(e["video_id"]), len(vorder))
        for e in self.events:
            e["_abs"] = vorder[int(e["video_id"])] * 1_000_000 + \
                int(e["frame_id"])
        self.events.sort(key=lambda e: (e["_abs"], e["sem_id"]))
        self.birth_frame = {}
        self.creator = {}
        for e in self.events:
            sid = int(e["sem_id"])
            if e["kind"] == "birth" and sid not in self.birth_frame:
                self.birth_frame[sid] = int(e["frame_id"])
                self.creator[sid] = (int(e["video_id"]),
                                     int(e["track_key"][1]))
        self._build_prefix_cache()
        self._build_event_cat()
        self.track_frames = defaultdict(list)
        for (vid, frame, tid) in self.track_rows:
            self.track_frames[(vid, tid)].append(frame)
        for k in self.track_frames:
            self.track_frames[k].sort()
        self.by_key = defaultdict(list)
        for r in self.rows:
            self.by_key[(r["video_id"], r["track_id"])].append(r)
        for k in self.by_key:
            self.by_key[k].sort(key=lambda r: r["frame_id"])
        self.identity = [r for r in self.rows
                         if r["outcome"] in IDENTITY_OUTCOMES]
        self.errors = [r for r in self.identity
                       if r["outcome"] in ERROR_OUTCOMES]

    def z_at(self, vid, frame, det_idx):
        rows = self.fpos.get(int(vid), {}).get(int(frame))
        if rows is None:
            return None
        k = int(det_idx)
        return self.zs[int(vid)][rows[k]] if 0 <= k < len(rows) else None

    def _build_prefix_cache(self):
        by_track = defaultdict(list)
        for (vid, frame, tid), r in self.track_rows.items():
            by_track[(vid, tid)].append((frame, r))
        self.prefix_cache = {}
        self.track_gt_cat = {}
        for (vid, tid), items in by_track.items():
            items.sort(key=lambda x: x[0])
            hist = []
            last = None
            cats = Counter()
            for frame, r in items:
                if last is not None and frame - last >= PRUNE_GAP:
                    hist = []
                z = self.z_at(vid, frame, int(r["det_idx"]))
                if z is not None:
                    hist.append(z)
                    if len(hist) > PREFIX_LEN:
                        hist.pop(0)
                if hist:
                    self.prefix_cache[(vid, frame, tid)] = _norm(
                        np.mean(hist, axis=0).astype(np.float32))
                last = frame
                g = match_gt(self.gt, int(r["image_id"]), r["bbox"])
                if g is not None and g["role"] == "novel":
                    cats[g["category_id"]] += 1
            if cats:
                self.track_gt_cat[(vid, tid)] = cats.most_common(1)[0][0]

    def _build_event_cat(self):
        self.event_cat = {}
        for e in self.events:
            key = (int(e["video_id"]), int(e["frame_id"]),
                   int(e["track_key"][1]))
            r = self.track_rows.get(key)
            cat = None
            if r is not None:
                g = match_gt(self.gt, int(r["image_id"]), r["bbox"])
                cat = g["category_id"] if g is not None else None
            self.event_cat[(int(e["sem_id"]), int(e["video_id"]),
                            int(e["frame_id"]),
                            int(e["track_key"][1]))] = cat

    def next_track_key(self, vid, tid, min_frame):
        keys = [k for k in self.track_rows if k[0] == vid and k[2] == tid]
        if not keys:
            return None
        keys.sort(key=lambda k: k[1])
        i = bisect.bisect_left([k[1] for k in keys], min_frame)
        return keys[i] if i < len(keys) else None

    def decision_event_for(self, key):
        """The birth/reuse event (abs, sem_id) for a track observation."""
        vid, frame, tid = key
        for e in self.events:
            if int(e["video_id"]) == vid and int(e["frame_id"]) == frame \
                    and int(e["track_key"][1]) == tid and \
                    e["kind"] in ("birth", "reuse"):
                return e["_abs"], int(e["sem_id"])
        return None


def rule_existing(tag, geo):
    if geo["best_cos"] < THETA:
        return False
    if tag == "j1b":
        return True
    if tag == "b1":
        return geo["margin"] >= 0.05
    if tag == "b2":
        return geo["local_entropy"] <= 1.6
    raise ValueError(tag)


def geometry(z, protos, members):
    if not protos:
        return None
    P = np.stack(list(protos.values())).astype(np.float32)
    ids = list(protos)
    cos = (P @ z).astype(np.float64)
    order = np.argsort(cos)[::-1]
    best_id = int(ids[order[0]])
    best_cos = float(cos[order[0]])
    second_cos = float(cos[order[1]]) if len(order) >= 2 else -1.0
    top5 = [float(cos[order[k]]) if k < len(order) else -1.0
            for k in range(5)]
    soft = np.exp(np.clip(top5, -30, 30))
    soft = soft / soft.sum()
    entropy = float(-(soft * np.log(soft + 1e-12)).sum())
    ms = members.get(best_id, [])
    if ms:
        M = np.stack(ms)
        Mc = M @ z
        m_mean = float(Mc.mean())
        m_std = float(Mc.std()) + 1e-6
        zscore = float((best_cos - m_mean) / m_std)
    else:
        m_mean = m_std = zscore = 0.0
    return {
        "best_cos": best_cos, "second_cos": second_cos,
        "margin": best_cos - second_cos,
        "local_entropy": entropy,
        "query_zscore": round(zscore, 4),
        "member_mean_cos": round(m_mean, 4),
        "support_causal": len(ms) + 1,
        "best_id": best_id,
    }


def counterfactual_horizons(td, vid, tid, fid):
    """Oracle: replay without this track's global writes at/after fid and
    score the same rule at this track's next decision at/after t+1/2/4/8."""
    results = {}
    targets = []          # (abs, sem_id, k, key)
    max_abs = -1
    for k in (1, 2, 4, 8):
        key = td.next_track_key(vid, tid, fid + k)
        if key is None:
            results[k] = "NO_OBS"
            continue
        r = td.track_rows[key]
        if float(r["p_known"]) >= float(r["decision_threshold"]):
            results[k] = "KNOWN_ROUTING"
            continue
        de = td.decision_event_for(key)
        if de is None:
            results[k] = "NO_DECISION"
            continue
        targets.append((de[0], de[1], k, key))
        max_abs = max(max_abs, de[0])
    if not targets:
        return results
    targets.sort()
    protos = {}
    members = defaultdict(list)
    mgc = defaultdict(list)
    ti = 0
    for e in td.events:
        e_abs = e["_abs"]
        e_sem = int(e["sem_id"])
        while ti < len(targets) and (
                targets[ti][0] < e_abs or
                (targets[ti][0] == e_abs and targets[ti][1] <= e_sem)):
            _evaluate_target(td, results, targets[ti], protos, members, mgc)
            ti += 1
        if e_abs > max_abs:
            break
        if int(e["video_id"]) == vid and \
                int(e["track_key"][1]) == tid and \
                int(e["frame_id"]) >= fid:
            continue  # deferred track contributes no global writes from t
        sid = int(e["sem_id"])
        zi = int(e.get("z_idx", -1))
        z = td.embs.get(zi) if zi >= 0 else None
        if z is None:
            continue
        if e["kind"] == "birth":
            if sid in protos:
                continue
            protos[sid] = _norm(z.astype(np.float32))
            members[sid].append(z)
            mgc[sid].append(td.event_cat.get(
                (sid, int(e["video_id"]), int(e["frame_id"]),
                 int(e["track_key"][1]))))
        elif e["kind"] == "update":
            if sid not in protos:
                continue
            p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                NOVEL_UPDATE_RATE * z
            protos[sid] = _norm(p.astype(np.float32))
            members[sid].append(z)
            mgc[sid].append(td.event_cat.get(
                (sid, int(e["video_id"]), int(e["frame_id"]),
                 int(e["track_key"][1]))))
    while ti < len(targets):
        _evaluate_target(td, results, targets[ti], protos, members, mgc)
        ti += 1
    return results


def _evaluate_target(td, results, target, protos, members, mgc):
    _, _, k, key = target
    if results.get(k) is not None:
        return
    vid, frame, tid = key
    z = td.prefix_cache.get(key)
    geo = geometry(z, protos, members) if z is not None else None
    if geo is None:
        results[k] = "NO_MEMORY"
        return
    tcat = td.track_gt_cat.get((vid, tid))
    bmaj = _maj(mgc.get(geo["best_id"], []))
    ok = rule_existing(td.tag, geo)
    if ok and bmaj == tcat:
        results[k] = "CORRECT_EXISTING"
    elif ok:
        results[k] = "WRONG_EXISTING"
    elif bmaj == tcat:
        results[k] = "OVERBIRTH"
    else:
        results[k] = "CORRECT_NEW"


def oracle_rows(td, rows):
    """One row per (decision, k) with per-decision eventual flags."""
    out = []
    for d in rows:
        vid, tid, fid = d["video_id"], d["track_id"], d["frame_id"]
        res = counterfactual_horizons(td, vid, tid, fid)
        last = td.track_frames.get((vid, tid))
        term = last[-1] if last else fid
        eventual = any(res.get(k) in ("CORRECT_EXISTING", "CORRECT_NEW")
                       for k in (1, 2, 4, 8))
        for k in (1, 2, 4, 8):
            fk = td.next_track_key(vid, tid, fid + k)
            out.append({
                "tag": td.tag, "video_id": vid, "frame_id": fid,
                "track_id": tid, "sem_id": d["sem_id"],
                "outcome": d["outcome"], "k": k,
                "future_frame": fk[1] if fk is not None else "",
                "oracle": res.get(k, "NO_OBS"),
                "correct": int(res.get(k, "NO_OBS") in
                              ("CORRECT_EXISTING", "CORRECT_NEW")),
                "track_terminated": int(term < fid + k),
                "eventual_correct": int(eventual),
            })
    return out


def fit_ambiguity(amb):
    """Logistic ambiguity model fit on j1b identity rows (dev-only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    feats = ["best_cos", "margin", "local_entropy", "novel_minus_known",
             "support_causal", "query_zscore"]
    rows = [r for r in amb if all(r[k] == r[k] for k in feats)]
    X = np.asarray([[r[k] for k in feats] for r in rows], dtype=np.float32)
    y = np.asarray([r["error"] for r in rows], dtype=np.int64)
    X = np.column_stack([X[:, :4], np.log1p(X[:, 4]), X[:, 5]])
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260808)
    for tr, va in skf.split(X, y):
        m = LogisticRegression(max_iter=1000)
        m.fit(X[tr], y[tr])
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    m = LogisticRegression(max_iter=1000)
    m.fit(X, y)
    return {
        "feats": feats,
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "auc_fit": float(roc_auc_score(y, m.predict_proba(X)[:, 1])),
        "coef": {f: float(c) for f, c in zip(feats, m.coef_[0])},
        "intercept": float(m.intercept_[0]),
        "model": m,
    }


def apply_score(model, r):
    feats = model["feats"]
    x = np.asarray([[r[k] for k in feats]], dtype=np.float32)
    x = np.column_stack([x[:, :4], np.log1p(x[:, 4]), x[:, 5]])
    return float(model["model"].predict_proba(x)[0, 1])


def pareto_for_rule(td, rule, defer_mask, oracle_map):
    """Metrics for one deferral rule on the identity set."""
    n = len(td.identity)
    nondef = [r for r, df in zip(td.identity, defer_mask) if not df]
    deferred = [r for r, df in zip(td.identity, defer_mask) if df]
    n_nd = len(nondef)
    n_df = len(deferred)
    imm_correct = [r for r in nondef
                   if r["outcome"] in ("CORRECT_EXISTING", "CORRECT_NEW")]
    df_correct_later = []
    for r in deferred:
        o = oracle_map.get((r["sem_id"], r["video_id"], r["frame_id"],
                            r["track_id"]))
        df_correct_later.append(bool(o) and o["eventual_correct"])
    n_df_resolved = sum(df_correct_later)
    n_prevent_wrong = sum(1 for r, ok in zip(deferred, df_correct_later)
                          if ok and r["outcome"] == "WRONG_EXISTING")
    n_prevent_over = sum(1 for r, ok in zip(deferred, df_correct_later)
                         if ok and r["outcome"] == "OVERBIRTH")
    lat = []
    for r, ok in zip(deferred, df_correct_later):
        if ok:
            o = oracle_map[(r["sem_id"], r["video_id"], r["frame_id"],
                            r["track_id"])]
            fs = sorted((k, v["future_frame"]) for k, v in
                        o["by_k"].items() if v["correct"])
            lat.append(fs[0][1] - r["frame_id"])
    total_correct = len(imm_correct) + n_df_resolved
    return {
        "tag": td.tag, "rule": rule,
        "n_identity": n, "n_deferred": n_df,
        "defer_fraction": round(n_df / n, 4) if n else 0.0,
        "immediate_correct_coverage": round(len(imm_correct) / n, 4),
        "immediate_error_rate": round(
            sum(1 for r in nondef if r["outcome"] in ERROR_OUTCOMES) /
            max(n_nd, 1), 4),
        "eventual_coverage": round(total_correct / n, 4) if n else 0.0,
        "resolution_precision": round(
            total_correct / max(n_nd + n_df_resolved, 1), 4),
        "prevented_wrong_reuse": n_prevent_wrong,
        "prevented_overbirth": n_prevent_over,
        "unresolved_at_termination": n_df - n_df_resolved,
        "latency_mean": round(float(np.mean(lat)), 2) if lat else "",
        "latency_median": round(float(np.median(lat)), 2) if lat else "",
        "latency_p90": round(float(np.percentile(lat, 90)), 2)
        if lat else "",
        "deferred_correct_later": int(n_df_resolved),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    summary = {}
    all_amb = []
    all_ttr = []
    all_pareto = []
    tds = {}

    for tag in tags:
        td = TagData(tag, gt)
        tds[tag] = td
        summary[tag] = {
            "n_decisions": len(td.rows),
            "outcome_counts": dict(Counter(r["outcome"] for r in td.rows)),
            "n_identity": len(td.identity),
            "n_errors": len(td.errors),
            "overbirth_rate": round(
                sum(1 for r in td.identity
                    if r["outcome"] == "OVERBIRTH") /
                max(sum(1 for r in td.identity
                        if r["outcome"] in ("OVERBIRTH", "CORRECT_NEW")),
                    1), 4),
            "wrong_reuse_rate": round(
                sum(1 for r in td.identity
                    if r["outcome"] == "WRONG_EXISTING") /
                max(sum(1 for r in td.identity if r["outcome"] in (
                    "WRONG_EXISTING", "CORRECT_EXISTING")), 1), 4),
        }
        amb = []
        for r in td.identity:
            if r["second_cos"] in (-1, "") or r["second_cos"] != r[
                    "second_cos"]:
                continue
            amb.append({
                "tag": tag, "sem_id": r["sem_id"],
                "video_id": r["video_id"], "frame_id": r["frame_id"],
                "track_id": r["track_id"], "outcome": r["outcome"],
                "error": int(r["outcome"] in ERROR_OUTCOMES),
                "best_cos": r["best_cos"], "margin": r["margin"],
                "local_entropy": r["local_entropy"],
                "novel_minus_known": r["novel_minus_known"],
                "support_causal": r["support_causal"],
                "query_zscore": r["query_zscore"],
                "near_count": r["near_count"],
                "member_mean_cos": r["member_mean_cos"],
                "member_std_cos": r["member_std_cos"],
                "n_gt_cat_mem": r["n_gt_cat_mem"],
            })
        all_amb.extend(amb)

    # fit ambiguity model on j1b only
    model = None
    j1b_amb = [r for r in all_amb if r["tag"] == "j1b"]
    if not args.skip_oracle and j1b_amb:
        model = fit_ambiguity(j1b_amb)
        summary["ambiguity_model"] = {
            "cv_auc_mean": model["cv_auc_mean"],
            "cv_auc_std": model["cv_auc_std"],
            "auc_fit": model["auc_fit"],
            "coef": model["coef"], "intercept": model["intercept"],
        }
    for r in all_amb:
        r["ambiguity_score"] = apply_score(model, r) if model else ""

    # oracle + pareto per tag
    for tag in tags:
        td = tds[tag]
        # counterfactual oracle for errors
        if not args.skip_oracle and td.errors:
            ttr = oracle_rows(td, td.errors)
            all_ttr.extend(ttr)
            by_dec = defaultdict(list)
            for row in ttr:
                by_dec[(row["sem_id"], row["video_id"], row["frame_id"],
                        row["track_id"])].append(row)
            summary[tag]["oracle_errors"] = {
                "n": len(td.errors),
                "eventual_correct": sum(
                    v[0]["eventual_correct"] for v in by_dec.values()),
                "terminated_before_t8": sum(
                    v[-1]["track_terminated"] for v in by_dec.values()),
                "resolved_by_k": {
                    str(k): sum(1 for v in by_dec.values()
                                if any(x["k"] == k and x["correct"]
                                       for x in v))
                    for k in (1, 2, 4, 8)},
            }
        # deferral rules on identity set
        if not args.skip_oracle:
            score_by_key = {}
            for r in all_amb:
                if r["tag"] == tag:
                    score_by_key[(r["sem_id"], r["video_id"],
                                  r["frame_id"], r["track_id"])] = r[
                        "ambiguity_score"]
            s66 = np.nan
            scores66 = [score_by_key[(r["sem_id"], r["video_id"],
                                      r["frame_id"], r["track_id"])]
                        for r in td.identity
                        if (r["sem_id"], r["video_id"], r["frame_id"],
                            r["track_id"]) in score_by_key]
            if scores66:
                s66 = float(np.percentile(scores66, 66))
            rules = {
                "M0_none": [False] * len(td.identity),
                "M1_margin_lt_0.05": [
                    r["margin"] < 0.05 for r in td.identity],
                "M2_entropy_gt_1.6": [
                    r["local_entropy"] > 1.6 for r in td.identity],
                "M3_ambiguity_top33": [
                    (s66 == s66 and
                     score_by_key.get((r["sem_id"], r["video_id"],
                                       r["frame_id"], r["track_id"]),
                                      np.nan) > s66)
                    for r in td.identity],
            }
            # oracle for all deferred identity rows
            deferred_rows = []
            for r, m1, m2, m3 in zip(td.identity, rules[
                    "M1_margin_lt_0.05"], rules["M2_entropy_gt_1.6"],
                    rules["M3_ambiguity_top33"]):
                if m1 or m2 or m3:
                    deferred_rows.append(r)
            oracle_map = {}
            if deferred_rows:
                for row in oracle_rows(td, deferred_rows):
                    key = (row["sem_id"], row["video_id"], row["frame_id"],
                           row["track_id"])
                    oracle_map.setdefault(key, {}).setdefault(
                        "by_k", {})[row["k"]] = {
                        "future_frame": row["future_frame"],
                        "correct": bool(row["correct"]),
                        "oracle": row["oracle"],
                    }
                    oracle_map[key]["eventual_correct"] = bool(
                        row["eventual_correct"])
            for rule, mask in rules.items():
                p = pareto_for_rule(td, rule, mask, oracle_map)
                all_pareto.append(p)
                summary[tag].setdefault("pareto", {})[rule] = p

    # write outputs
    if all_amb:
        fields = list(all_amb[0].keys())
        with open(OUT / "ambiguity_features.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_amb)
    if all_ttr:
        fields = list(all_ttr[0].keys())
        with open(OUT / "time_to_resolution.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_ttr)
    if all_pareto:
        fields = list(all_pareto[0].keys())
        with open(OUT / "deferral_pareto.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_pareto)
    with open(OUT / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("PHASE4M_AUDIT_DONE", {t: summary[t]["n_decisions"]
                                 for t in tags})


if __name__ == "__main__":
    main()
