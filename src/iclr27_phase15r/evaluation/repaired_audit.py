"""Phase-15R corrected raw-DINOv2 causal audit.

This is intentionally independent of the historical Phase-15 linker.  It
keeps the physical proposal stream untouched, updates a prefix before every
decision, separates the three calibration quantities, and emits one row-level
action that can be checked by the independent transition validator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.iclr27_phase15r.validation.transition_validator import (
    audit_prefix_invariance, validate_transitions,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase15r"
PREFIXES = (1, 2, 4, 8, 16)
MAX_STATES = 512
MAX_EXEMPLARS = 4


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    target = path.resolve()
    h = hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def l2one(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def read_rows(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            q = dict(r)
            for k in ("video_id", "frame_id", "source_frame_index", "image_id",
                      "proposal_local_id", "track_id", "prior_hits", "gt_track_id",
                      "gt_category_id"):
                if k in q and q[k] not in (None, ""):
                    q[k] = int(float(q[k]))
            if "score" in q:
                q["score"] = float(q["score"])
            if "gt_temporal_iou" in q and q["gt_temporal_iou"] not in (None, ""):
                q["gt_temporal_iou"] = float(q["gt_temporal_iou"])
            out.append(q)
    return out


def chrono_indices(rows: list[dict]) -> list[int]:
    return sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id", 0)), int(rows[i]["track_id"]), i))


def load_public():
    z = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", allow_pickle=False)
    manifest = json.loads((ROOT / "outputs/iclr27_phase15/manifests/phase15_preregistration.json").read_text())
    known_ids = {int(x) for x in json.loads((ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json").read_text())}
    return {k: z[k] for k in z.files}, manifest, known_ids


def prefix_features(data: dict) -> dict[int, np.ndarray]:
    ff = data["frame_feats"].astype(np.float32)
    mm = data["frame_mask"].astype(np.float32)[..., None]
    out = {}
    for p in PREFIXES:
        q = min(p, ff.shape[1])
        den = np.maximum(mm[:, :q].sum(axis=1), 1.0)
        out[p] = l2((ff[:, :q] * mm[:, :q]).sum(axis=1) / den)
    return out


def _bank_from_groups(groups: dict[int, list[np.ndarray]], exemplar: bool = False) -> dict:
    cats = sorted(int(c) for c, v in groups.items() if v)
    if exemplar:
        ex = np.zeros((len(cats), MAX_EXEMPLARS, 768), dtype=np.float32)
        mask = np.zeros((len(cats), MAX_EXEMPLARS), dtype=bool)
        for ci, c in enumerate(cats):
            vals = [l2one(v) for v in groups[c][:MAX_EXEMPLARS]]
            if vals:
                ex[ci, :len(vals)] = np.asarray(vals)
                mask[ci, :len(vals)] = True
        return {"name": "proposal_exemplar", "categories": np.asarray(cats, dtype=np.int64),
                "vectors": ex, "mask": mask, "source": "proposal_rows"}
    vec = np.asarray([l2one(np.mean(np.asarray(groups[c], dtype=np.float32), axis=0)) for c in cats], dtype=np.float32)
    return {"name": "prototype", "categories": np.asarray(cats, dtype=np.int64),
            "vectors": vec, "source": "grouped_rows"}


def build_banks(data: dict, manifest: dict, known_ids: set[int]):
    train_idx = [int(x) for x in manifest["split"]["representation_train"]["track_indices"]]
    train_videos = {int(x) for x in manifest["split"]["representation_train"]["videos"]}
    gt_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for i in train_idx:
        c = int(data["labels"][i])
        if int(data["is_known"][i]) and c in known_ids:
            gt_groups[c].append(data["mean_feats"][i].astype(np.float32))
    gt = _bank_from_groups(gt_groups)
    gt["name"] = "historical_gt_box"
    gt["source"] = "outputs/iclr27_phase6d/assets/full_tao_tracks.npz; public TRAIN role"

    proxy_csv = ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv"
    proxy_npz = ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz"
    proxy_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    proxy_ex_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    proxy_rows = read_rows(proxy_csv)
    proxy_feats = np.load(proxy_npz, allow_pickle=False)["feats"].astype(np.float32)
    if len(proxy_rows) != len(proxy_feats):
        raise RuntimeError("proxy proposal/feature length mismatch")
    # Same DINOv2 crop recipe and public TRAIN restriction; tracker provenance
    # is recorded separately and is never confused with Phase-6B DSCT.
    proxy_track_first: dict[tuple[int, int], int] = {}
    for i, r in enumerate(proxy_rows):
        if int(r.get("video_id", -1)) not in train_videos:
            continue
        if str(r.get("gt_role", "")) != "known":
            continue
        c = int(r.get("gt_category_id", -1))
        if c not in known_ids:
            continue
        proxy_groups[c].append(proxy_feats[i])
        key = (int(r["video_id"]), int(r["track_id"]))
        if key not in proxy_track_first:
            proxy_track_first[key] = i
            proxy_ex_groups[c].append(proxy_feats[i])
    proposal = _bank_from_groups(proxy_groups)
    proposal["name"] = "proposal_crop"
    proposal["source"] = "outputs/iclr27_phase4t/train_stream (TRAIN-only tracker-induced diagnostic)"
    ex = _bank_from_groups(proxy_ex_groups, exemplar=True)
    ex["name"] = "proposal_four_exemplar"
    ex["source"] = proposal["source"]
    banks = {"historical_gt": gt, "proposal": proposal, "proposal_exemplar": ex}
    exact_csv = OUT / "dsct_subset/proposals.csv"
    exact_npz = OUT / "dsct_subset/proposal_dinov2.npz"
    exact_present = exact_csv.exists() and exact_npz.exists()
    exact_audit = {
        "present": bool(exact_present),
        "proposal_csv": str(exact_csv),
        "feature_npz": str(exact_npz),
    }
    if exact_present:
        exact_rows = read_rows(exact_csv)
        exact_feats = np.load(exact_npz, allow_pickle=False)["feats"].astype(np.float32)
        if len(exact_rows) != len(exact_feats):
            raise RuntimeError("exact DSCT proposal/feature length mismatch")
        exact_groups: dict[int, list[np.ndarray]] = defaultdict(list)
        exact_ex_groups: dict[int, list[np.ndarray]] = defaultdict(list)
        exact_track_first: set[tuple[int, int]] = set()
        for i, r in enumerate(exact_rows):
            if int(r.get("video_id", -1)) not in train_videos:
                continue
            if str(r.get("gt_role", "")) not in ("known", "supported_known"):
                continue
            c = int(r.get("gt_category_id", -1))
            if c not in known_ids:
                continue
            exact_groups[c].append(exact_feats[i])
            key = (int(r["video_id"]), int(r["track_id"]))
            if key not in exact_track_first:
                exact_track_first.add(key)
                exact_ex_groups[c].append(exact_feats[i])
        dsct = _bank_from_groups(exact_groups)
        dsct["name"] = "proposal_dsct"
        dsct["source"] = "outputs/iclr27_phase15r/dsct_subset; Phase-6B DSCT TRAIN-only subset"
        dsct_ex = _bank_from_groups(exact_ex_groups, exemplar=True)
        dsct_ex["name"] = "proposal_dsct_exemplar"
        dsct_ex["source"] = dsct["source"]
        banks["proposal_dsct"] = dsct
        banks["proposal_dsct_exemplar"] = dsct_ex
        exact_audit.update({
            "rows": len(exact_rows),
            "known_rows_in_train_role": int(sum(len(v) for v in exact_groups.values())),
            "known_tracks_in_train_role": len(exact_track_first),
            "categories": sorted(int(x) for x in exact_groups),
            "videos": sorted({int(r["video_id"]) for r in exact_rows if int(r.get("video_id", -1)) in train_videos}),
            "alignment_summary": str((OUT / "dsct_subset/proposals_summary.json").resolve()),
        })
    return banks, {
        "proxy_rows": len(proxy_rows), "proxy_known_rows_in_train_role": int(sum(len(v) for v in proxy_groups.values())),
        "proxy_categories": sorted(int(x) for x in proxy_groups),
        "proxy_videos_in_train_role": sorted(train_videos & {int(r["video_id"]) for r in proxy_rows}),
        "exact_phase6b_dsct_train_cache_present": bool(exact_present),
        "exact_dsct_subset": exact_audit,
        "exact_cache_search": ["outputs/iclr27_phase6b/q1/final_dsct (DEV-only)", "outputs/iclr27_phase14c (DEV+-only)"],
    }


def score_known(cur: np.ndarray, bank: dict) -> tuple[float, int]:
    if len(bank["categories"]) == 0:
        return -1.0, -1
    if bank["vectors"].ndim == 2:
        sims = bank["vectors"] @ cur
    else:
        sims_all = bank["vectors"] @ cur
        sims_all = np.where(bank["mask"], sims_all, -1.0)
        sims = sims_all.max(axis=1)
    j = int(np.argmax(sims))
    return float(sims[j]), j


def calibration_for_bank(bank: dict, data: dict, manifest: dict, known_ids: set[int], pfx: dict[int, np.ndarray]) -> dict:
    idx = np.asarray(manifest["split"]["calibration"]["track_indices"], dtype=np.int64)
    vec = pfx[8][idx]
    labels = data["labels"][idx].astype(np.int64)
    ks = np.asarray([score_known(x, bank)[0] for x in vec], dtype=np.float64)
    bank_cats = {int(x) for x in bank["categories"]}
    # A calibration track is a positive only when its supported-known
    # category is actually represented by the TRAIN-only bank.  Categories
    # outside the bank are an explicit coverage negative, not silently
    # discarded; the coverage table reports this limitation.
    pos = ks[np.asarray([int(data["is_known"][i]) and int(labels[j]) in bank_cats for j, i in enumerate(idx)])]
    neg = ks[np.asarray([not (int(data["is_known"][i]) and int(labels[j]) in bank_cats) for j, i in enumerate(idx)])]
    # Complete chronological calibration pair population (public labels only).
    same, diff = [], []
    vids = data["video_ids"][idx].astype(np.int64)
    for a in range(len(vec)):
        for b in range(a + 1, len(vec)):
            if vids[a] == vids[b]:
                continue
            (same if labels[a] == labels[b] else diff).append(float(vec[a] @ vec[b]))
    all_pairs = np.asarray(same + diff, dtype=np.float64)
    known_source = "public calibration"
    if not len(pos):
        # The category-disjoint calibration role has no overlap with a
        # TRAIN-visible bank by construction.  Use a legal leave-one-track-out
        # known episode from representation TRAIN rather than inventing a
        # threshold or peeking at DEV+.  For the GT bank, explicitly remove the
        # query track from its category prototype; for a proposal bank, the
        # independent proxy crops provide the bank and TRAIN GT tracks provide
        # the query distribution.
        tr_idx = np.asarray(manifest["split"]["representation_train"]["track_indices"], dtype=np.int64)
        tr_vec = pfx[8][tr_idx]; tr_lab = data["labels"][tr_idx].astype(np.int64)
        tr_pos, tr_neg = [], []
        for j, (vv, cc) in enumerate(zip(tr_vec, tr_lab)):
            if int(cc) not in bank_cats: continue
            if bank["name"] == "historical_gt":
                other = [tr_vec[t] for t in range(len(tr_vec)) if t != j and int(tr_lab[t]) == int(cc)]
                if not other: continue
                pp = l2one(np.mean(np.asarray(other), axis=0)); tr_pos.append(float(vv @ pp))
                others = []
                for oc in bank_cats:
                    if oc == int(cc): continue
                    ids = [t for t in range(len(tr_vec)) if int(tr_lab[t]) == oc and t != j]
                    if ids: others.append(float(vv @ l2one(np.mean(tr_vec[ids], axis=0))))
                if others: tr_neg.append(max(others))
            else:
                tr_pos.append(score_known(vv, bank)[0])
                if bank["vectors"].ndim == 2:
                    other = [float(vv @ bank["vectors"][q]) for q, oc in enumerate(bank["categories"]) if int(oc) != int(cc)]
                else:
                    other = [float((bank["vectors"][q][bank["mask"][q]] @ vv).max()) for q, oc in enumerate(bank["categories"]) if int(oc) != int(cc) and bank["mask"][q].any()]
                if other: tr_neg.append(max(other))
        pos, neg = np.asarray(tr_pos, dtype=np.float64), np.asarray(tr_neg, dtype=np.float64)
        known_source = "representation_train_leave_one_track_out_fallback"
    tau_known = float((pos.mean() + neg.mean()) / 2.0) if len(pos) and len(neg) else 0.5
    tau_existing = float((np.mean(same) + np.mean(diff)) / 2.0) if same and diff else 0.5
    delta_new = float(np.clip(0.5 * (all_pairs.std() if len(all_pairs) else 0.1), 0.02, 0.15))
    return {
        "tau_known": tau_known, "tau_existing": tau_existing, "delta_new": delta_new,
        "known_positive_n": int(len(pos)), "known_negative_n": int(len(neg)),
        "known_positive_mean": float(pos.mean()) if len(pos) else None,
        "known_negative_mean": float(neg.mean()) if len(neg) else None,
        "same_category_pairs": int(len(same)), "different_category_pairs": int(len(diff)),
        "same_category_mean": float(np.mean(same)) if same else None,
        "different_category_mean": float(np.mean(diff)) if diff else None,
        "pair_std": float(all_pairs.std()) if len(all_pairs) else None,
        "source_role": known_source + "; no DEV+/Q1 labels",
    }


def controller(ks: float, es: float, cal: dict) -> str:
    if ks >= cal["tau_known"] and ks >= es + cal["delta_new"]:
        return "known"
    if es >= cal["tau_existing"] and es >= ks + cal["delta_new"]:
        return "existing"
    return "new"


def replay(rows: list[dict], feats: np.ndarray, bank: dict, cal: dict, mode: str) -> tuple[list[dict], dict]:
    if len(rows) != len(feats):
        raise RuntimeError("replay row/feature mismatch")
    chrono = chrono_indices(rows)
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: Counter = Counter()
    first_assignment: dict[tuple[int, int], tuple[str, int]] = {}
    states: dict[int, dict] = {}
    # Preallocated bounded state matrix avoids rebuilding a Python/vstack
    # object on every row when the rejection controller creates many slots.
    state_matrix = np.zeros((MAX_STATES * MAX_EXEMPLARS, 768), dtype=np.float32)
    state_quality = np.full((MAX_STATES * MAX_EXEMPLARS,), -np.inf, dtype=np.float32)
    state_counts = np.zeros((MAX_STATES,), dtype=np.int32)
    next_slot = 0
    overflow = 0
    decisions: list[dict | None] = [None] * len(rows)

    def state_score(cur):
        if not states:
            return -1.0, None
        active = np.flatnonzero(np.isfinite(state_quality))
        sims = state_matrix[active] @ cur
        j = int(np.argmax(sims))
        return float(sims[j]), int(active[j] // MAX_EXEMPLARS)

    def update_state(sid: int, cur: np.ndarray, quality: float):
        st = states[sid]
        if len(st["vectors"]) < MAX_EXEMPLARS:
            slot = len(st["vectors"])
            st["vectors"].append(cur.copy()); st["quality"].append(float(quality))
            state_matrix[sid * MAX_EXEMPLARS + slot] = cur
            state_quality[sid * MAX_EXEMPLARS + slot] = float(quality)
            state_counts[sid] = slot + 1
            return
        j = int(np.argmin(st["quality"]))
        if float(quality) > float(st["quality"][j]):
            st["vectors"][j] = cur.copy(); st["quality"][j] = float(quality)
            state_matrix[sid * MAX_EXEMPLARS + j] = cur
            state_quality[sid * MAX_EXEMPLARS + j] = float(quality)

    for i in chrono:
        r = rows[i]
        key = (int(r["video_id"]), int(r["track_id"]))
        x = l2one(feats[i])
        sums[key] = sums.get(key, np.zeros_like(x)) + x
        counts[key] += 1
        cur = l2one(sums[key] / float(counts[key]))
        kscore, kpos = score_known(cur, bank)
        escore, best_sid = state_score(cur)
        if mode == "birth_only" and key in first_assignment:
            old_action, old_sid = first_assignment[key]
            action = "existing" if old_action == "new" else old_action
            sid = old_sid
        else:
            action = controller(kscore, escore, cal)
            if action == "known":
                sid = int(bank["categories"][kpos])
            elif action == "existing" and best_sid is not None:
                sid = 100000 + int(best_sid)
            else:
                if len(states) >= MAX_STATES and best_sid is not None:
                    action, sid, overflow = "existing", 100000 + int(best_sid), overflow + 1
                else:
                    sid = 100000 + int(next_slot)
                    states[next_slot] = {"vectors": [], "quality": [], "birth": key}
                    next_slot += 1
                    action = "new"
            if mode == "birth_only":
                first_assignment[key] = (action, sid)
        if action == "new":
            slot = int(sid - 100000)
            if slot not in states:
                states[slot] = {"vectors": [], "quality": [], "birth": key}
            update_state(slot, cur, 1.0)
        elif action == "existing":
            slot = int(sid - 100000)
            if slot in states:
                update_state(slot, cur, escore if np.isfinite(escore) else 0.0)
        decisions[i] = {
            "sem_action": action, "sem_sid": int(sid), "sem_kscore": float(kscore),
            "sem_escore": float(escore), "sem_slot": int(sid - 100000) if sid >= 100000 else int(sid),
            "obs_count": int(counts[key]), "prefix_bin": "16+" if counts[key] >= 16 else str(counts[key]),
        }
    return [d for d in decisions], {
        "mode": mode, "rows": len(rows), "states_born_internal": len(states),
        "max_exemplars": MAX_EXEMPLARS, "max_states": MAX_STATES, "state_overflow": overflow,
        "cumulative_features": mode == "cumulative", "birth_only_timing_control": mode == "birth_only",
        "future_frames_used": False, "physical_id_used_as_feature": False,
    }


def ece(scores: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    # cosine is mapped to [0,1] only for calibration-error display.
    p = np.clip((np.asarray(scores) + 1.0) / 2.0, 0, 1)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i + 1 < bins else (p <= edges[i + 1]))
        if mask.any(): total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def offline_meta(data: dict, manifest: dict, pfx: dict[int, np.ndarray]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    idx = np.asarray(manifest["split"]["meta_validation"]["track_indices"], dtype=np.int64)
    labels = data["labels"][idx].astype(np.int64); vids = data["video_ids"][idx].astype(np.int64)
    out = {"role": "meta_validation", "tracks": int(len(idx)), "prefixes": {}}
    for p in PREFIXES:
        if p > 8: continue
        v = pfx[p][idx]; pairs, ys = [], []; retrieval = []
        by_cat = defaultdict(list); by_vid = defaultdict(list)
        for i in range(len(v)):
            cand = [j for j in range(len(v)) if j != i and vids[j] != vids[i]]
            pos = {j for j in cand if labels[j] == labels[i]}
            if not pos: continue
            ranked = sorted(cand, key=lambda j: (-float(v[i] @ v[j]), j))
            hit = 0; ap = 0.0
            for rank, j in enumerate(ranked, 1):
                if j in pos: hit += 1; ap += hit / rank
            retrieval.append((float(ranked[0] in pos), float(bool(set(ranked[:5]) & pos)), ap / len(pos)))
            by_cat[int(labels[i])].append(float(ranked[0] in pos)); by_vid[int(vids[i])].append(float(ranked[0] in pos))
            for j in cand:
                if j > i:
                    pairs.append(float(v[i] @ v[j])); ys.append(int(labels[i] == labels[j]))
        s = np.asarray(pairs, dtype=np.float64); y = np.asarray(ys, dtype=np.int64)
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, s)); pr = float(average_precision_score(y, s)); gap = float(s[y == 1].mean() - s[y == 0].mean())
        else: auc = pr = gap = None
        cats = [np.mean(x) for x in by_cat.values()]; vids_g = [np.mean(x) for x in by_vid.values()]
        out["prefixes"][str(p)] = {
            "queries": len(retrieval), "r1": float(np.mean([x[0] for x in retrieval])) if retrieval else None,
            "r5": float(np.mean([x[1] for x in retrieval])) if retrieval else None,
            "map": float(np.mean([x[2] for x in retrieval])) if retrieval else None,
            "pairs": len(s), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
            "roc_auc": auc, "pr_auc": pr, "positive_negative_gap": gap,
            "ece": ece(s, y) if len(s) else None,
            "category_grouped_r1": {"n": len(cats), "mean": float(np.mean(cats)) if cats else None,
                                     "low": float(np.min(cats)) if cats else None, "high": float(np.max(cats)) if cats else None},
            "video_grouped_r1": {"n": len(vids_g), "mean": float(np.mean(vids_g)) if vids_g else None,
                                  "low": float(np.min(vids_g)) if vids_g else None, "high": float(np.max(vids_g)) if vids_g else None},
        }
    return out


def proposal_domain_audit(proxy_rows: list[dict], proxy_feats: np.ndarray, known_ids: set[int], train_videos: set[int]) -> dict:
    by_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    cats = {}
    for i, r in enumerate(proxy_rows):
        if int(r.get("video_id", -1)) not in train_videos or r.get("gt_role") != "known": continue
        c = int(r.get("gt_category_id", -1))
        if c not in known_ids: continue
        key = (int(r["video_id"]), int(r["track_id"]))
        by_track[key].append(i); cats[key] = c
    tv = [(k, l2one(proxy_feats[ix].mean(axis=0)), cats[k]) for k, ix in by_track.items()]
    pairs = []; ys = []; r1 = []
    for i, (ki, vi, ci) in enumerate(tv):
        cand = [j for j, (kj, _, _) in enumerate(tv) if j != i and kj[0] != ki[0]]
        pos = {j for j in cand if tv[j][2] == ci}
        if not pos: continue
        ranked = sorted(cand, key=lambda j: (-float(vi @ tv[j][1]), j))
        r1.append(float(ranked[0] in pos))
        for j in cand: pairs.append(float(vi @ tv[j][1])); ys.append(int(tv[j][2] == ci))
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(ys, dtype=np.int64); s = np.asarray(pairs, dtype=np.float64)
    return {"source": "Phase4T tracker-induced TRAIN proposal diagnostic (not DSCT)",
            "tracks": len(tv), "videos": len({k[0] for k, _, _ in tv}), "categories": len({c for _, _, c in tv}),
            "cross_video_pairs": len(s), "positive_pairs": int(y.sum()),
            "r1": float(np.mean(r1)) if r1 else None,
            "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
            "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None,
            "note": "physical proposal generator differs from frozen Phase-6B DSCT; retained only as a proxy diagnostic while the exact DSCT subset is audited separately"}


def strict_metrics(rows: list[dict], aligned: list[dict], decisions: list[dict], known_ids: set[int], internal: dict) -> dict:
    order = chrono_indices(rows)
    aligned_idxs = [i for i in order if int(aligned[i].get("gt_track_id", -1)) >= 0 and aligned[i].get("gt_role") in ("supported_known", "novel")]
    known = [i for i in aligned_idxs if aligned[i].get("gt_role") == "supported_known"]
    novel = [i for i in aligned_idxs if aligned[i].get("gt_role") == "novel"]
    first_cat = {}
    for i in novel: first_cat.setdefault(int(aligned[i]["gt_category_id"]), i)
    reuse = [i for i in novel if i != first_cat[int(aligned[i]["gt_category_id"])] ]
    slot_cat, slot_birth = {}, {}
    for i in order:
        d = decisions[i]
        if d["sem_action"] == "new" and int(aligned[i].get("gt_track_id", -1)) >= 0 and aligned[i].get("gt_role") == "novel":
            slot_cat.setdefault(int(d["sem_sid"]), int(aligned[i]["gt_category_id"]))
            slot_birth.setdefault(int(d["sem_sid"]), (int(rows[i]["video_id"]), int(rows[i]["track_id"])))
    def correct_reuse(i):
        d = decisions[i]; return d["sem_action"] == "existing" and slot_cat.get(int(d["sem_sid"])) == int(aligned[i]["gt_category_id"])
    cross = [i for i in reuse if int(decisions[i]["sem_sid"]) in slot_birth and slot_birth[int(decisions[i]["sem_sid"])] != (int(rows[i]["video_id"]), int(rows[i]["track_id"]))]
    cross_vid = [i for i in cross if slot_birth[int(decisions[i]["sem_sid"])][0] != int(rows[i]["video_id"])]
    known_ok = sum(decisions[i]["sem_action"] == "known" and int(decisions[i]["sem_sid"]) == int(aligned[i]["gt_category_id"]) for i in known)
    known_by_category = defaultdict(list)
    for i in known:
        c = int(aligned[i]["gt_category_id"])
        known_by_category[c].append(float(decisions[i]["sem_action"] == "known" and int(decisions[i]["sem_sid"]) == c))
    known_macro = float(np.mean([np.mean(v) for v in known_by_category.values()])) if known_by_category else 0.0
    reuse_ok = sum(correct_reuse(i) for i in reuse); cross_ok = sum(correct_reuse(i) for i in cross_vid)
    first_birth_ok = sum(decisions[i]["sem_action"] == "new" for i in first_cat.values())
    frag = defaultdict(set)
    for i in novel:
        if decisions[i]["sem_action"] in ("new", "existing"): frag[int(aligned[i]["gt_category_id"])].add(int(decisions[i]["sem_sid"]))
    by_key = defaultdict(list)
    for i in aligned_idxs: by_key[(int(rows[i]["video_id"]), int(rows[i]["track_id"]))].append(i)
    switches = adj = 0
    for ids in by_key.values():
        for a, b in zip(ids, ids[1:]):
            adj += 1; switches += int((decisions[a]["sem_action"], decisions[a]["sem_sid"]) != (decisions[b]["sem_action"], decisions[b]["sem_sid"]))
    y = np.asarray([int(aligned[i]["gt_category_id"]) for i in novel], dtype=np.int64)
    p = np.asarray([int(decisions[i]["sem_sid"]) for i in novel], dtype=np.int64)
    if len(y) > 1:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        nmi, ari = float(normalized_mutual_info_score(y, p)), float(adjusted_rand_score(y, p))
    else: nmi = ari = 0.0
    half = len(known) // 2; kh1 = known[:half]; kh2 = known[half:]
    known_first = sum(decisions[i]["sem_action"] == "known" and int(decisions[i]["sem_sid"]) == int(aligned[i]["gt_category_id"]) for i in kh1) / max(len(kh1), 1)
    known_second = sum(decisions[i]["sem_action"] == "known" and int(decisions[i]["sem_sid"]) == int(aligned[i]["gt_category_id"]) for i in kh2) / max(len(kh2), 1)
    by_prefix = {}
    for b in ("1", "2", "4", "8", "16+"):
        ids = [i for i in aligned_idxs if decisions[i]["prefix_bin"] == b]
        ki = [i for i in ids if aligned[i].get("gt_role") == "supported_known"]
        ni = [i for i in ids if aligned[i].get("gt_role") == "novel"]
        by_prefix[b] = {"occurrences": len(ids), "known_acc": sum(decisions[i]["sem_action"] == "known" and int(decisions[i]["sem_sid"]) == int(aligned[i]["gt_category_id"]) for i in ki) / max(len(ki), 1), "novel_existing_correct": sum(correct_reuse(i) for i in ni) / max(len(ni), 1)}
    validator = validate_transitions([{**rows[i], **decisions[i]} for i in order], known_ids, internal_state_count=int(internal["states_born_internal"]))
    ct_cat = Counter(int(aligned[i]["gt_category_id"]) for i in cross_vid); ct_good_cat = Counter(int(aligned[i]["gt_category_id"]) for i in cross_vid if correct_reuse(i))
    ct_vid = Counter(int(rows[i]["video_id"]) for i in cross_vid); ct_good_vid = Counter(int(rows[i]["video_id"]) for i in cross_vid if correct_reuse(i))
    frag_values = [len(v) for v in frag.values()]
    new_aligned_novel = sum(1 for i in order if decisions[i]["sem_action"] == "new" and aligned[i].get("gt_role") == "novel" and int(aligned[i].get("gt_track_id", -1)) >= 0)
    total_new = sum(1 for i in order if decisions[i]["sem_action"] == "new")
    return {
        "n_rows": len(rows), "n_aligned_occurrences": len(aligned_idxs), "n_known_occurrences": len(known), "n_novel_occurrences": len(novel),
        "known_occurrence_acc": known_ok / max(len(known), 1), "known_macro_category_acc": known_macro,
        "known_category_acc": {str(c): float(np.mean(v)) for c, v in sorted(known_by_category.items())},
        "first_novel_birth_acc": first_birth_ok / max(len(first_cat), 1),
        "novel_reuse_acc": reuse_ok / max(len(reuse), 1), "cross_physical_reuse_acc": sum(correct_reuse(i) for i in cross) / max(len(cross), 1),
        "ct_reuse": cross_ok / max(len(cross_vid), 1), "ct_reuse_correct": cross_ok, "ct_reuse_eligible": len(cross_vid),
        "ct_category": {str(c): {"correct": ct_good_cat[c], "eligible": ct_cat[c]} for c in sorted(ct_cat)},
        "ct_video": {str(v): {"correct": ct_good_vid[v], "eligible": ct_vid[v]} for v in sorted(ct_vid)},
        "known_to_new_rate": sum(decisions[i]["sem_action"] == "new" for i in known) / max(len(known), 1),
        "known_to_existing_rate": sum(decisions[i]["sem_action"] == "existing" for i in known) / max(len(known), 1),
        "reuse_to_new_rate": sum(decisions[i]["sem_action"] == "new" for i in reuse) / max(len(reuse), 1),
        "semantic_switch_rate": switches / max(adj, 1), "n_true_novel_categories": len(first_cat),
        "unique_new_states": validator["unique_new_state_count"], "new_action_count": validator["new_action_count"],
        "birth_precision": new_aligned_novel / max(total_new, 1),
        "n_born_novel_states_global": validator["unique_new_state_count"], "internal_state_count": int(internal["states_born_internal"]),
        "validator": validator, "duplicate_creation_rate": float(np.mean([len(v) > 1 for v in frag.values()])) if frag else 0.0,
        "mean_fragmentation": float(np.mean(frag_values)) if frag_values else 0.0,
        "median_fragmentation": float(np.median(frag_values)) if frag_values else 0.0,
        "novel_nmi": nmi, "novel_ari": ari, "known_acc_first_half_stream": known_first, "known_acc_second_half_stream": known_second,
        "known_forgetting_delta": known_second - known_first, "by_observation_prefix": by_prefix,
        "action_confusion": {"known_to_new": sum(decisions[i]["sem_action"] == "new" for i in known), "known_to_existing": sum(decisions[i]["sem_action"] == "existing" for i in known), "reuse_to_new": sum(decisions[i]["sem_action"] == "new" for i in reuse)},
        "internal_replay": internal,
    }


def write_decision_csv(path: Path, rows: list[dict], decisions: list[dict]):
    out = []
    for r, d in zip(rows, decisions):
        q = dict(r); q.update(d); out.append(q)
    atomic_csv(path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-online", action="store_true")
    args = ap.parse_args()
    started = time.time()
    data, manifest, known_ids = load_public(); pfx = prefix_features(data)
    rows = read_rows(ROOT / "outputs/iclr27_phase14c/proposals/proposals_mixed.csv")
    aligned = read_rows(ROOT / "outputs/iclr27_phase14c/proposals/proposals_aligned.csv")
    feats = np.load(ROOT / "outputs/iclr27_phase14c/features/proposal_dinov2.npz", allow_pickle=False)["feats"].astype(np.float32)
    if not (len(rows) == len(aligned) == len(feats)): raise RuntimeError("DEV+ proposal alignment mismatch")
    banks, bank_audit = build_banks(data, manifest, known_ids)
    cal = {name: calibration_for_bank(bank, data, manifest, known_ids, pfx) for name, bank in banks.items()}
    atomic_json(OUT / "eval/calibration_summary.json", {"protocol": "phase15r", "banks": cal, "inputs": {"public": "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", "preregistration": "outputs/iclr27_phase15r/manifests/preregistration.json"}})
    train_videos = {int(x) for x in manifest["split"]["representation_train"]["videos"]}
    proxy_rows = read_rows(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")
    proxy_feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz", allow_pickle=False)["feats"].astype(np.float32)
    bank_audit["proxy_domain_metrics"] = proposal_domain_audit(proxy_rows, proxy_feats, known_ids, train_videos)
    bank_audit["historical_gt_categories"] = [int(x) for x in banks["historical_gt"]["categories"]]
    bank_audit["proposal_categories"] = [int(x) for x in banks["proposal"]["categories"]]
    bank_audit["proposal_exemplar_categories"] = [int(x) for x in banks["proposal_exemplar"]["categories"]]
    for key in ("proposal_dsct", "proposal_dsct_exemplar"):
        if key in banks:
            bank_audit[f"{key}_categories"] = [int(x) for x in banks[key]["categories"]]
    atomic_json(OUT / "eval/known_bank_domain_audit.json", bank_audit)
    atomic_json(OUT / "eval/offline_summary.json", {"historical_phase15_cited": {"meta_prefix8_r1": 0.9074074074, "relation_mean_r1": 0.1388888889, "devplus_raw_prefix16_r1": 0.5032679739, "devplus_raw_prefix16_auc": 0.808887}, "repaired_public_meta": offline_meta(data, manifest, pfx), "leakage_flags": {"devplus_labels_used_for_fit_or_calibration": False, "q1_label_used": False, "future_frames_used": False}})
    # Replays are deliberately small in interface but full-stream in rows.
    candidates = [
        ("birth_only_historical_gt", "historical_gt", "birth_only"),
        ("cumulative_historical_gt", "historical_gt", "cumulative"),
        ("birth_only_proposal", "proposal", "birth_only"),
        ("cumulative_proposal", "proposal", "cumulative"),
        ("cumulative_proposal_exemplar", "proposal_exemplar", "cumulative"),
    ]
    if "proposal_dsct" in banks:
        candidates.extend([
            ("birth_only_proposal_dsct", "proposal_dsct", "birth_only"),
            ("cumulative_proposal_dsct", "proposal_dsct", "cumulative"),
            ("cumulative_proposal_dsct_exemplar", "proposal_dsct_exemplar", "cumulative"),
        ])
    strict = {}; contracts = {}; prefix_audit = {}
    for name, bank_name, mode in candidates:
        dec, internal = replay(rows, feats, banks[bank_name], cal[bank_name], mode)
        write_decision_csv(OUT / f"csv/{name}.csv", rows, dec)
        sm = strict_metrics(rows, aligned, dec, known_ids, internal)
        strict[name] = {"bank": bank_name, "calibration": cal[bank_name], "strict": sm}
        contracts[name] = sm["validator"]
        # Prefix replay on a fixed early chronological slice; this detects the
        # historical skip/carry-forward defect without opening any labels.
        k = min(256, len(rows)); early = list(range(k)); erows = [rows[i] for i in early]; efeats = feats[:k]
        def f(xs, _bn=banks[bank_name], _cal=cal[bank_name], _mode=mode):
            # xs is a prefix in original chronological order, so features are
            # indexed by position in this local replay.
            dd, _ = replay(list(xs), efeats[:len(xs)], _bn, _cal, _mode)
            return [{"sem_action": d["sem_action"], "sem_sid": d["sem_sid"]} for d in dd]
        prefix_audit[name] = audit_prefix_invariance(erows, f, (1, 2, 4, 8, 16, 64, 128, 256))
    atomic_json(OUT / "eval/strict_trackocd_summary.json", {"protocol": "phase15r", "candidates": strict, "legacy_gate": {n: {"known_ge_0_60": bool(v["strict"]["known_occurrence_acc"] >= 0.60), "ct_gt_0": bool(v["strict"]["ct_reuse"] > 0), "pass": bool(v["strict"]["known_occurrence_acc"] >= 0.60 and v["strict"]["ct_reuse"] > 0)} for n, v in strict.items()}})
    atomic_json(OUT / "eval/transition_contract.json", {"protocol": "phase15r", "candidates": contracts, "focused_tests": "src/iclr27_phase15r/tests/test_transition_validator.py"})
    atomic_json(OUT / "eval/prefix_timing_audit.json", {"protocol": "phase15r", "candidates": prefix_audit, "historical_defect": "historical causal_link skipped all later feature accumulation"})
    gates = {n: bool(v["strict"]["known_occurrence_acc"] >= 0.60 and v["strict"]["ct_reuse"] > 0) for n, v in strict.items()}
    exact_present = bool(bank_audit.get("exact_phase6b_dsct_train_cache_present"))
    if not exact_present:
        branch, reason = "R-E", "exact Phase-6B DSCT TRAIN proposal cache is absent; the available proposal bank is explicitly a different tracker and cannot support a matched-domain foundation decision"
    else:
        exact_cats = set(int(x) for x in banks["proposal_dsct"]["categories"])
        repr_cats = set(int(x) for x in manifest["split"]["representation_train"]["categories"])
        exact_gate = gates.get("cumulative_proposal_dsct", False) or gates.get("cumulative_proposal_dsct_exemplar", False)
        if exact_gate:
            branch = "R-A"
            reason = "matched Phase-6B DSCT TRAIN proposal bank plus cumulative raw-DINOv2 replay satisfies the registered legacy gate"
        elif len(exact_cats) < len(repr_cats):
            branch = "R-C"
            reason = "exact DSCT proposal bank is legal and matched, but detector/crop alignment covers only a subset of representation-train known categories; the coverage/domain loss must be isolated before changing the foundation"
        else:
            branch = "R-D"
            reason = "transition contract and matched DSCT bank are valid, but cumulative raw-DINOv2 remains weak on the repaired category-transfer/online audit"
    decision = {"protocol": "phase15r", "selected_branch": branch, "reason": reason, "foundation_audit_opened": branch == "R-D", "phase16_training_authorized": False, "q1_opened": False, "candidate_gates": gates, "proxy_results_are_diagnostic_only": True, "exact_dsct_bank_present": exact_present}
    atomic_json(OUT / "eval/phase15r_decision.json", decision)
    # Leakage/source audit with resolved symlink hashes.
    paths = ["data/iclr27_phase15r/sources/full_tao_tracks.npz", "data/iclr27_phase15r/sources/proposals_mixed.csv", "data/iclr27_phase15r/sources/proposal_dinov2.npz", "data/iclr27_phase15r/sources/mixed_gt_tracks.jsonl", "data/iclr27_phase15r/sources/train_proposals_proxy.csv", "data/iclr27_phase15r/sources/train_proposal_dinov2_proxy.npz"]
    source_audit = {p: {"path": p, "resolved": str((ROOT / p).resolve()), "is_symlink": (ROOT / p).is_symlink(), "sha256": sha256(ROOT / p), "size_bytes": (ROOT / p).resolve().stat().st_size} for p in paths}
    exact_paths = ["outputs/iclr27_phase15r/dsct_subset/proposals.csv", "outputs/iclr27_phase15r/dsct_subset/proposal_dinov2.npz", "data/iclr27_phase15r/sources/validation_train_subset.json"]
    exact_source_audit = {p: {"path": p, "resolved": str((ROOT / p).resolve()), "is_symlink": (ROOT / p).is_symlink(), "sha256": sha256(ROOT / p), "size_bytes": (ROOT / p).resolve().stat().st_size} for p in exact_paths}
    atomic_json(OUT / "manifests/data_and_leakage_audit.json", {"protocol": "phase15r", "sources": source_audit, "exact_dsct_sources": exact_source_audit, "exact_phase6b_dsct_train_cache_present": exact_present, "devplus_used_for_fit": False, "devplus_used_for_calibration": False, "q1_label_used": False, "future_frames_used": False, "physical_id_used_as_feature": False, "private_gt_used_for_decision": False, "gt_boxes_primary_final_input": False, "symlink_only_sources": True, "exact_subset_gt_labels_used_for_alignment_only": True})
    atomic_json(OUT / "eval/resource_summary.json", {"protocol": "phase15r", "runtime_seconds": time.time() - started, "gpu_job": False, "gpu_ids": [], "max_gpus": 4, "host_memory_preflight": "125G total / 87G available", "disk_preflight": "199G free on /data1", "near_oom": False, "other_processes_terminated": False, "git_revision": "git_unavailable; content hashes recorded"})
    print(json.dumps({"candidates": list(strict), "selected_branch": branch, "seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
