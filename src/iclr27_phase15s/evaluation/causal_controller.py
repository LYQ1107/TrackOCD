"""Decoupled causal semantic controller for matched DINOv2 features."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase15s.evaluation.fixed_ct import fixed_ct_metrics
from src.iclr27_phase15s.validation.transition_validator import validate_transitions

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
MAX_STATES = 2048


def norm(x):
    x = np.asarray(x, dtype=np.float32); return x / max(float(np.linalg.norm(x)), 1e-12)


def chrono(rows):
    return sorted(range(len(rows)), key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]), int(rows[i].get("proposal_local_id", 0)), int(rows[i].get("track_id", 0)), i))


def build_bank(rows, feats, role_videos, known_ids, mode="cls"):
    vecs = defaultdict(list); ex = defaultdict(list); first = set()
    for i, r in enumerate(rows):
        if int(r.get("video_id", -1)) not in set(role_videos): continue
        if str(r.get("gt_role", "")) not in ("known", "supported_known"): continue
        c = int(r.get("gt_category_id", -1))
        if c not in known_ids: continue
        v = norm(feats[i]); vecs[c].append(v)
        key = (int(r["video_id"]), int(r["track_id"]))
        if key not in first: first.add(key); ex[c].append(v)
    cats = sorted(vecs)
    prototypes = {c: norm(np.mean(vecs[c], axis=0)) for c in cats}
    exemplars = {c: [norm(x) for x in ex[c][:8]] for c in cats}
    ex_rows = []; ex_cats = []
    for c in cats:
        for x in exemplars[c]: ex_rows.append(x); ex_cats.append(c)
    return {"categories": cats, "prototypes": prototypes, "exemplars": exemplars,
            "prototype_matrix": np.asarray([prototypes[c] for c in cats], dtype=np.float32),
            "exemplar_matrix": np.asarray(ex_rows, dtype=np.float32) if ex_rows else np.zeros((0, 768), dtype=np.float32),
            "exemplar_categories": np.asarray(ex_cats, dtype=np.int64),
            "rows": int(sum(len(v) for v in vecs.values())), "tracks": len(first), "mode": mode}


def known_score(v, bank):
    if not bank["categories"]: return -1.0, -1
    p = bank["prototype_matrix"] @ v
    vals = p.copy()
    if len(bank["exemplar_matrix"]):
        es = bank["exemplar_matrix"] @ v
        for j, c in enumerate(bank["categories"]):
            hit = bank["exemplar_categories"] == int(c)
            if hit.any(): vals[j] = max(float(vals[j]), float(es[hit].max()))
    j = int(np.argmax(vals)); return float(vals[j]), int(bank["categories"][j])


def cross_score(v, state, current_key):
    sims = [float(v @ x) for x, key in state["exemplars"] if key != current_key and key[0] != current_key[0]]
    return max(sims, default=-1.0)


def replay(rows, feats, bank, thresholds, known_ids, *, return_internal=False):
    tk = float(thresholds["tau_known"]); tc = float(thresholds["tau_cross_physical_reuse"]); margin = float(thresholds.get("margin_new", 0.0))
    order = chrono(rows); prefix = {}; states = {}; local = {}; next_sid = 100000; out = [None] * len(rows); overflow = False
    for i in order:
        r = rows[i]; key = (int(r["video_id"]), int(r["track_id"])); x = np.asarray(feats[i], dtype=np.float32)
        total, n = prefix.get(key, (np.zeros_like(x), 0)); total = total + x; n += 1; prefix[key] = (total, n); v = norm(total / n)
        ks, kc = known_score(v, bank); local_sid = local.get(key)
        best_sid, es = None, -1.0
        # One batched dot product replaces the old state-by-state Python
        # loop.  Provenance filtering is still explicit and causal.
        sid_list, vec_list, valid_list = [], [], []
        for sid0, st in states.items():
            for x0, source_key in st["exemplars"]:
                sid_list.append(sid0); vec_list.append(x0); valid_list.append(source_key != key and source_key[0] != key[0])
        if vec_list:
            vv = np.asarray(vec_list, dtype=np.float32); ss = vv @ v; valid = np.asarray(valid_list, dtype=bool)
            if valid.any():
                # Max per state, then choose the deterministic highest score.
                best_by = {}
                for sid0, score0 in zip(np.asarray(sid_list)[valid], ss[valid]):
                    best_by[int(sid0)] = max(float(score0), best_by.get(int(sid0), -1.0))
                if best_by:
                    best_sid, es = max(best_by.items(), key=lambda z: (z[1], -z[0]))
        if ks >= tk:
            action, sid, source = "known", int(kc), "known"
            local.pop(key, None)
        elif local_sid is not None and local_sid in states:
            action, sid, source = "existing", int(local_sid), "local"
        elif best_sid is not None and es >= tc and es >= ks + margin:
            action, sid, source = "existing", int(best_sid), "cross_physical"
        else:
            if len(states) >= MAX_STATES:
                overflow = True; out[i] = {"sem_action": "invalid_overflow", "sem_sid": -1, "sem_kscore": ks, "sem_escore": es, "evidence_source": "overflow"}; continue
            sid = next_sid; next_sid += 1; states[sid] = {"exemplars": [], "birth": key}; local[key] = sid; action, source = "new", "new"
        out[i] = {"sem_action": action, "sem_sid": int(sid), "sem_kscore": float(ks), "sem_escore": float(es), "evidence_source": source, "obs_count": n, "physical_video": key[0], "physical_track": key[1]}
        if action in ("new", "existing"):
            st = states[int(sid)]
            if not st["exemplars"] or source in ("new", "cross_physical"):
                st["exemplars"].append((v.copy(), key)); st["exemplars"] = st["exemplars"][-8:]
            local[key] = int(sid)
    if overflow: return out, {"valid": False, "overflow": True, "states": len(states)}
    return out, {"valid": True, "overflow": False, "states": len(states), "internal_state_count": len(states)}


def strict_metrics(rows, decisions, known_ids, selected_novel):
    known_idx = [i for i, r in enumerate(rows) if str(r.get("gt_role", "")) in ("known", "supported_known")]
    known_ok = sum(str(decisions[i]["sem_action"]) == "known" and int(decisions[i]["sem_sid"]) == int(rows[i]["gt_category_id"]) for i in known_idx)
    by_cat = defaultdict(list)
    for i in known_idx: by_cat[str(int(rows[i]["gt_category_id"]))].append(int(str(decisions[i]["sem_action"]) == "known" and int(decisions[i]["sem_sid"]) == int(rows[i]["gt_category_id"])))
    novel_idx = [i for i, r in enumerate(rows) if str(r.get("gt_role", "")) == "novel"]
    new_novel = sum(decisions[i]["sem_action"] == "new" for i in novel_idx); total_new = sum(d["sem_action"] == "new" for d in decisions)
    # Fragmentation is measured by states attached to true novel categories;
    # it is descriptive and does not alter fixed CT.
    frag = defaultdict(set)
    for i in novel_idx:
        if decisions[i]["sem_action"] in ("new", "existing"): frag[int(rows[i]["gt_category_id"])].add(int(decisions[i]["sem_sid"]))
    ct = fixed_ct_metrics(rows, decisions, selected_novel)
    # Historical diagnostic only: its denominator is prediction-conditioned
    # by available births/states and therefore is not CT recall.
    birth_cat = {}; birth_src = {}; reuse = []
    for i, (r, d) in enumerate(zip(rows, decisions)):
        if str(r.get("gt_role", "")) != "novel": continue
        sid = int(d["sem_sid"])
        if d["sem_action"] == "new" and sid not in birth_cat:
            birth_cat[sid] = int(r.get("gt_category_id", -1)); birth_src[sid] = (int(r.get("video_id", -1)), int(r.get("track_id", -1)))
        if d["sem_action"] == "existing": reuse.append(i)
    legacy = []
    for i in reuse:
        sid = int(decisions[i]["sem_sid"]); src = birth_src.get(sid)
        if src is not None and src[0] != int(rows[i].get("video_id", -1)):
            legacy.append(int(birth_cat.get(sid, -2) == int(rows[i].get("gt_category_id", -1))))
    ct["legacy_prediction_conditioned_ct"] = {"correct": int(sum(legacy)), "eligible": len(legacy), "rate": sum(legacy) / max(len(legacy), 1), "is_recall": False}
    valid = validate_transitions(decisions, known_ids, internal_state_count=None)
    return {"n_rows": len(rows), "known_occurrences": len(known_idx), "known_occurrence_acc": known_ok / max(len(known_idx), 1), "known_macro_category_acc": sum(sum(v)/len(v) for v in by_cat.values()) / max(len(by_cat), 1), "known_category_acc": {k: sum(v)/len(v) for k, v in sorted(by_cat.items())}, "new_novel": new_novel, "birth_precision": new_novel / max(total_new, 1), "novel_categories": len(frag), "fragmentation_mean_states": sum(len(v) for v in frag.values()) / max(len(frag), 1), "duplicate_creation_rate": sum(len(v) > 1 for v in frag.values()) / max(len(frag), 1), "fixed_ct": ct, "transition_contract": valid}


def calibration_grid(rows, feats, bank, known_ids, selected_novel, seeds=(20260824, 20260825)):
    # The registered grid is evaluated on complete public calibration rows,
    # with two deterministic video orders.  A seed changes episode order only;
    # no labels enter the controller, only the public objective.
    # Keep complete physical episodes while bounding the public calibration
    # replay.  We retain the first four observations of every track, then
    # fill deterministic false-proposal rows to the registered 2,000-row cap.
    if len(rows) > 2000:
        keep = set(); by_track = defaultdict(list)
        for i, r in enumerate(rows): by_track[(int(r["video_id"]), int(r["track_id"]))].append(i)
        for key in sorted(by_track): keep.update(by_track[key][:4])
        for i, r in enumerate(rows):
            if len(keep) >= 2000: break
            if str(r.get("gt_role", "")) == "fp": keep.add(i)
        keep = sorted(keep)[:2000]; rows = [rows[i] for i in keep]; feats = feats[keep]
    vals = []
    for s in seeds:
        # Chronology is always preserved; the cheap seed only changes the
        # deterministic tie order of proposals sharing one frame.
        order = sorted(range(len(rows)), key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]), (i + int(s)) % max(len(rows), 1)))
        rr = [rows[i] for i in order]; ff = feats[order]
        for tk in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
            for tc in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
                for margin in (0.0, 0.05, 0.1, 0.15):
                    ds, internal = replay(rr, ff, bank, {"tau_known": tk, "tau_cross_physical_reuse": tc, "margin_new": margin}, known_ids)
                    if not internal["valid"]: continue
                    met = strict_metrics(rr, ds, known_ids, selected_novel)
                    score = met["known_occurrence_acc"] + met["fixed_ct"]["recall"] + met["birth_precision"] - 0.1 * met["fragmentation_mean_states"]
                    vals.append({"seed": int(s), "tau_known": tk, "tau_cross_physical_reuse": tc, "margin_new": margin, "objective": score, "metrics": {"known": met["known_occurrence_acc"], "ct": met["fixed_ct"]["recall"], "birth_precision": met["birth_precision"], "fragmentation": met["fragmentation_mean_states"]}})
    vals.sort(key=lambda x: (-x["objective"], x["tau_known"], x["tau_cross_physical_reuse"], x["margin_new"], x["seed"]))
    best = vals[0] if vals else {"tau_known": 0.45, "tau_cross_physical_reuse": 0.45, "margin_new": 0.05, "objective": None}
    thresholds = {k: best[k] for k in ("tau_known", "tau_cross_physical_reuse", "margin_new")}
    return thresholds, {"seeds": list(seeds), "grid_size": 6 * 6 * 4, "evaluated": len(vals), "calibration_rows_used": len(rows), "max_calibration_rows": 2000, "episode_prefix_observations": 4, "best": best, "top10": vals[:10]}
