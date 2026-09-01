"""Phase 4K offline provenance audit.

GT (TAO validation subset) is used ONLY for offline diagnostic labeling:
never for online method inputs.  For every novel prototype we build
provenance (birth + updates + reuses), and for every association decision
we classify whether the semantic contribution was helpful / harmful /
neutral / no-effect, attributed to the prototype that supplied it.

Outputs (per tag):
  audit/prototype_provenance_<tag>.csv
  audit/prototype_utility.csv            (one row per tag, aggregated)
  audit/association_interventions.csv    (one row per decision)
  audit/cross_track_support.csv          (per prototype)
  audit/pollution_hubs.csv               (top useful / harmful cases)
  audit/offline_summary_<tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PROV_ROOT = Path(os.environ.get(
    "PHASE4L_PROV_ROOT",
    ROOT / "outputs" / "iclr27_phase4k" / "audit"))
OUT_ROOT = Path(os.environ.get(
    "PHASE4L_AUDIT_OUT", PROV_ROOT))
TAO_JSON = Path(os.environ.get(
    "PHASE4L_TAO_JSON",
    ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" /
    "validation_20.json"))
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
PRE_ASSOC = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / \
    "pre_assoc_detections"
MATCH_THR = 0.5          # tracker match_score_thr (frozen IDOL config)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt():
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        b = ann["bbox"]
        cat = int(ann["category_id"])
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": cat,
            "role": "known" if cat in known else "novel",
        })
    return out


def match_gt(gt, image_id, bbox):
    best, bi = None, MATCH_THR
    for g in gt.get(image_id, []):
        v = iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def load_pre_assoc():
    """(video_id, frame_id, det_local_id) -> det dict; frame_id -> image_id."""
    dets, frame_img = {}, {}
    for p in sorted(PRE_ASSOC.glob("*.jsonl")):
        vid = int(p.stem)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (vid, int(r["frame_order"]), int(r["det_local_id"]))
            dets[key] = r
            frame_img[(vid, int(r["frame_order"]))] = int(r["image_id"])
    return dets, frame_img


def load_semantic_logs(prov_root):
    """Row indexes used by the offline join."""
    rows = []
    track_row = {}     # (video, frame, track) -> row
    det_row = {}       # (video, frame, det_idx) -> row
    for p in sorted((prov_root / "semantic_logs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            vid, fid, tid = int(r["video_id"]), int(r["frame_id"]), \
                int(r["physical_track_id"])
            key = (vid, fid, tid)
            if key not in track_row:
                track_row[key] = r
            det_row[(vid, fid, int(r["det_idx"]))] = r
    return rows, track_row, det_row


def load_events(prov_root, tag):
    evs = []
    p = prov_root / f"prototype_event_log_{tag}.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                evs.append(json.loads(line))
    embs = {}
    pz = prov_root / f"embeddings_{tag}.npz"
    if pz.exists():
        z = np.load(pz)["embeddings"]
        embs = {i: np.asarray(x, dtype=np.float32) for i, x in enumerate(z)}
    return evs, embs


def load_assoc(prov_root, tag):
    out = []
    p = prov_root / f"association_decisions_{tag}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def track_gt_history(track_rows, gt):
    """(video, track) -> sorted [(frame, gt_track_id, role, category)]."""
    hist = defaultdict(list)
    for (vid, fid, tid), r in track_rows.items():
        g = match_gt(gt, int(r["image_id"]), r["bbox"])
        if g is not None:
            hist[(vid, tid)].append((fid, g["track_id"], g["role"],
                                     g["category_id"]))
    for k in hist:
        hist[k].sort(key=lambda x: x[0])
    return hist


def majority_gt(hist, video_id, track_id, before_frame):
    """Majority GT physical track of a track before a frame (causal)."""
    vals = [x[1] for x in hist.get((video_id, track_id), [])
            if x[0] < before_frame]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def classify_intervention(r, det_gt, ap_gt, fn_gt, chosen_gt):
    """Returns effect label and correctness flags."""
    ap_ok = ap_gt is not None and ap_gt == det_gt
    fn_ok = fn_gt is not None and fn_gt == det_gt
    chosen_ok = chosen_gt is not None and chosen_gt == det_gt
    if det_gt is None:
        ap_ok = fn_ok = chosen_ok = False   # no correct association exists
    ap_best_score = r["appearance_best_score"]
    fn_best_score = r["final_best_score"]
    argmax_switch = int(r["appearance_best_idx"]) != int(r["final_best_idx"])
    thr_switch = (ap_best_score > MATCH_THR) != (fn_best_score > MATCH_THR)
    decision_effect = argmax_switch or thr_switch
    if not decision_effect:
        effect = "no_effect"
    elif chosen_ok and not ap_ok:
        effect = "helpful"
    elif ap_ok and not chosen_ok:
        effect = "harmful"
    else:
        effect = "neutral_switch"
    return effect, argmax_switch, thr_switch, chosen_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag
    prov_root = PROV_ROOT / f"prov_{tag}"
    out_root = OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    gt = load_gt()
    dets, frame_img = load_pre_assoc()
    log_rows, track_row, det_row = load_semantic_logs(prov_root)
    events, embs = load_events(prov_root, tag)
    decisions = load_assoc(prov_root, tag)
    hist = track_gt_history(track_row, gt)

    # ---- detection GT labels for semantic-log rows (diagnostic only) ----
    row_gt = {}
    for (vid, fid, _), r in track_row.items():
        row_gt[(vid, fid, r["physical_track_id"])] = match_gt(
            gt, int(r["image_id"]), r["bbox"])

    # ---- association intervention rows ----
    inter_rows = []
    proto_assoc = defaultdict(lambda: Counter())
    for r in decisions:
        vid, fid = int(r["video_id"]), int(r["frame_id"])
        det = dets.get((vid, fid, int(r["raw_det_idx"])))
        det_gt = None
        if det is not None:
            image_id = frame_img.get((vid, fid))
            det_gt = match_gt(gt, image_id, det["bbox_xyxy_original"])
        ap_gt = majority_gt(hist, vid, r["ap_track_id"], fid)
        fn_gt = majority_gt(hist, vid, r["fn_track_id"], fid)
        chosen_gt = majority_gt(hist, vid, r["chosen_track_id"], fid)
        effect, argmax_switch, thr_switch, chosen_ok = classify_intervention(
            r, (det_gt or {}).get("track_id"),
            ap_gt, fn_gt, chosen_gt)
        det_track = (det_gt or {}).get("track_id")
        det_role = (det_gt or {}).get("role", "fp")
        det_cat = (det_gt or {}).get("category_id")
        proto = r.get("chosen_novel_id")
        if not isinstance(proto, int):
            proto = r.get("det_novel_id")
        if not isinstance(proto, int):
            proto = None
        if proto is not None:
            proto_assoc[proto][effect] += 1
        inter_rows.append({
            "video_id": vid, "frame_id": fid, "raw_det_idx":
            int(r["raw_det_idx"]),
            "ap_track_id": int(r["ap_track_id"]),
            "fn_track_id": int(r["fn_track_id"]),
            "chosen_track_id": int(r["chosen_track_id"]),
            "assigned_id": int(r["assigned_id"]),
            "det_gt_track": det_track if det_track is not None else "",
            "det_role": det_role, "det_category": det_cat or "",
            "ap_gt_track": ap_gt if ap_gt is not None else "",
            "fn_gt_track": fn_gt if fn_gt is not None else "",
            "chosen_gt_track": chosen_gt if chosen_gt is not None else "",
            "sem_delta_appearance": r["sem_delta_appearance"],
            "sem_delta_final": r["sem_delta_final"],
            "argmax_switch": int(argmax_switch),
            "threshold_switch": int(thr_switch),
            "effect": effect,
            "prototype_id": proto if proto is not None else "",
            "det_novel_id": r.get("det_novel_id") or "",
            "ap_novel_id": r.get("ap_novel_id") or "",
            "fn_novel_id": r.get("fn_novel_id") or "",
            "chosen_novel_id": r.get("chosen_novel_id") or "",
            "chosen_ok": int(chosen_ok),
        })

    # ---- per-prototype provenance ----
    births = {}
    updates = defaultdict(list)
    reuses = defaultdict(list)
    for e in events:
        sem = int(e["sem_id"])
        if e["kind"] == "birth":
            births[sem] = e
        elif e["kind"] == "update":
            updates[sem].append(e)
        elif e["kind"] == "reuse":
            reuses[sem].append(e)

    joined_missing = 0
    event_total = len(events)
    for e in events:
        key = (int(e["video_id"]), int(e["frame_id"]),
               int(e["track_key"][1]))
        if track_row.get(key) is None:
            joined_missing += 1

    protos = []
    for sem in sorted(set(births) | set(updates) | set(reuses)):
        b = births.get(sem)
        us = updates.get(sem, [])
        rs = reuses.get(sem, [])
        # join GT labels to every event via the semantic log track row
        def ev_gt(e):
            key = (int(e["video_id"]), int(e["frame_id"]),
                   int(e["track_key"][1]))
            r = track_row.get(key)
            if r is None:
                return None
            return match_gt(gt, int(r["image_id"]), r["bbox"])

        birth_gt = ev_gt(b) if b is not None else None
        reuse_gts = [ev_gt(e) for e in rs]
        upd_gts = [ev_gt(e) for e in us]
        novel_cats = [g["category_id"] for g in reuse_gts
                      if g is not None and g["role"] == "novel"]
        maj_cat = Counter(novel_cats).most_common(1)[0][0] if novel_cats \
            else None
        same_real = sum(1 for g in reuse_gts
                        if g is not None and g["role"] == "novel"
                        and g["category_id"] == maj_cat)
        cross_real = sum(1 for g in reuse_gts
                         if g is not None and g["role"] == "novel"
                         and g["category_id"] != maj_cat)
        known_abs = sum(1 for g in reuse_gts
                        if g is not None and g["role"] == "known")
        fp_abs = sum(1 for g in reuse_gts if g is None)
        novel_abs = same_real + cross_real
        n_reuse = len(rs)
        assoc = proto_assoc.get(sem, Counter())
        helpful, harmful = assoc.get("helpful", 0), assoc.get("harmful", 0)
        # embedding stats (updates only; birth embedding included via z_idx)
        zidx = []
        if b is not None and b["z_idx"] >= 0:
            zidx.append(b["z_idx"])
        for e in us:
            if e["z_idx"] >= 0:
                zidx.append(e["z_idx"])
        Z = np.stack([embs[i] for i in zidx]) if zidx and all(
            i in embs for i in zidx) else None
        dispersion = drift = -1.0
        if Z is not None and len(Z) >= 2:
            Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
            cos = Zn @ Zn.T
            dispersion = float(1.0 - cos[np.triu_indices(len(Z), 1)].mean())
            drift = float(Zn[0] @ Zn[-1])
        cross_tracks = sorted({(int(e["video_id"]), int(e["track_key"][1]))
                               for e in us + rs})
        creator = (int(b["video_id"]), int(b["track_key"][1])) \
            if b is not None else None
        cross_track_reuses = sum(
            1 for e in rs
            if creator is not None and
            (int(e["video_id"]), int(e["track_key"][1])) != creator)
        same_track_updates = sum(
            1 for e in us if int(e["same_track"]))
        cross_track_updates = len(us) - same_track_updates
        support = max([int(b["support_after"]) for b in [b] if b] +
                      [int(e["support_after"]) for e in us], default=0)
        # fixed transparent retrospective groups (diagnostic only)
        if n_reuse >= 2:
            if (known_abs + fp_abs) / n_reuse >= 0.5 or cross_real > same_real:
                group = "POLLUTING"
            elif same_real / n_reuse >= 0.5 and helpful >= harmful:
                group = "USEFUL"
            else:
                group = "MIXED"
        else:
            group = "LOW_EVIDENCE"
        protos.append({
            "sem_id": sem, "tag": tag,
            "birth_video": int(b["video_id"]) if b else "",
            "birth_frame": int(b["frame_id"]) if b else "",
            "birth_track": int(b["track_key"][1]) if b else "",
            "birth_track_age": int(b["track_age"]) if b else "",
            "birth_track_len": int(b["track_len"]) if b else "",
            "birth_det_score": float(b["det_score"]) if b else "",
            "birth_p_known": float(b["p_known"]) if b else "",
            "birth_known_margin": float(b["known_margin"]) if b else "",
            "birth_novel_conf": float(b["novel_conf"]) if b else "",
            "birth_gt_role": (birth_gt or {}).get("role", "fp") if b else "",
            "birth_gt_category": (birth_gt or {}).get("category_id", "")
            if b else "",
            "birth_gt_track": (birth_gt or {}).get("track_id", "")
            if b else "",
            "n_updates": len(us), "n_reuses": n_reuse,
            "final_support": support,
            "same_track_updates": same_track_updates,
            "cross_track_updates": cross_track_updates,
            "distinct_physical_tracks": len(cross_tracks),
            "distinct_videos": len({int(e["video_id"]) for e in us + rs}),
            "cross_track_reuses": cross_track_reuses,
            "same_real_class_reuses": same_real,
            "cross_real_class_reuses": cross_real,
            "known_absorptions": known_abs, "fp_absorptions": fp_abs,
            "novel_absorptions": novel_abs,
            "majority_gt_category": maj_cat if maj_cat is not None else "",
            "semantic_purity": round(same_real / max(n_reuse, 1), 4),
            "embedding_dispersion": round(dispersion, 4)
            if dispersion >= 0 else "",
            "prototype_drift": round(drift, 4) if drift >= -1 else "",
            "mean_compat": round(float(np.mean(
                [float(e["compat_best"]) for e in us] +
                [float(e["compat"]) for e in rs])), 4)
            if us or rs else "",
            "max_compat": round(float(max(
                [float(e["compat_best"]) for e in us] +
                [float(e["compat"]) for e in rs])), 4)
            if us or rs else "",
            "assoc_helpful": helpful, "assoc_harmful": harmful,
            "assoc_neutral": assoc.get("neutral_switch", 0),
            "assoc_no_effect": assoc.get("no_effect", 0),
            "assoc_net_utility": helpful - harmful,
            "outcome_group": group,
        })

    # ---- CSVs ----
    prov_fields = list(protos[0].keys()) if protos else ["sem_id"]
    with open(out_root / f"prototype_provenance_{tag}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=prov_fields)
        w.writeheader()
        w.writerows(protos)

    if inter_rows:
        ifields = list(inter_rows[0].keys())
        with open(out_root / "association_interventions.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=ifields)
            w.writeheader()
            w.writerows(inter_rows)
        with open(out_root / f"association_interventions_{tag}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=ifields)
            w.writeheader()
            w.writerows(inter_rows)

    # prototype_utility.csv: one summary row per tag
    grp = Counter(p["outcome_group"] for p in protos)
    n_useful = grp.get("USEFUL", 0)
    n_poll = grp.get("POLLUTING", 0)
    utility_row = {
        "tag": tag, "n_prototypes": len(protos),
        "n_useful": n_useful, "n_polluting": n_poll,
        "n_mixed": grp.get("MIXED", 0),
        "n_low_evidence": grp.get("LOW_EVIDENCE", 0),
        "useful_frac": round(n_useful / max(len(protos), 1), 4),
        "polluting_frac": round(n_poll / max(len(protos), 1), 4),
        "cross_track_protos": sum(1 for p in protos
                                  if p["distinct_physical_tracks"] >= 2),
        "same_track_only_protos": sum(1 for p in protos
                                      if p["distinct_physical_tracks"] < 2),
        "mean_final_support": round(float(np.mean(
            [p["final_support"] for p in protos])), 2) if protos else 0,
        "mean_purity_useful": round(float(np.mean(
            [p["semantic_purity"] for p in protos
             if p["outcome_group"] == "USEFUL"])), 4) if n_useful else "",
        "mean_purity_polluting": round(float(np.mean(
            [p["semantic_purity"] for p in protos
             if p["outcome_group"] == "POLLUTING"])), 4) if n_poll else "",
        "total_assoc_helpful": sum(p["assoc_helpful"] for p in protos),
        "total_assoc_harmful": sum(p["assoc_harmful"] for p in protos),
        "net_assoc_utility": sum(p["assoc_net_utility"] for p in protos),
        "mean_support_useful": round(float(np.mean(
            [p["final_support"] for p in protos
             if p["outcome_group"] == "USEFUL"])), 2) if n_useful else "",
        "mean_support_polluting": round(float(np.mean(
            [p["final_support"] for p in protos
             if p["outcome_group"] == "POLLUTING"])), 2) if n_poll else "",
        "mean_cross_track_tracks_useful": round(float(np.mean(
            [p["distinct_physical_tracks"] for p in protos
             if p["outcome_group"] == "USEFUL"])), 2) if n_useful else "",
        "mean_cross_track_tracks_polluting": round(float(np.mean(
            [p["distinct_physical_tracks"] for p in protos
             if p["outcome_group"] == "POLLUTING"])), 2) if n_poll else "",
    }
    util_path = out_root / "prototype_utility.csv"
    util_existed = util_path.exists()
    util_rows = []
    if util_existed:
        with open(util_path) as f:
            util_rows = list(csv.DictReader(f))
    util_rows = [r for r in util_rows if r.get("tag") != tag] + [utility_row]
    with open(util_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(utility_row.keys()))
        w.writeheader()
        w.writerows(util_rows)

    # cross_track_support.csv: per prototype (accumulated across tags)
    def read_old(name):
        p = out_root / name
        if not p.exists():
            return []
        with open(p) as f:
            return [r for r in csv.DictReader(f) if r.get("tag") != tag]

    cts_rows = []
    for p in protos:
        cts_rows.append({
            "tag": tag, "sem_id": p["sem_id"],
            "cross_track": int(p["distinct_physical_tracks"] >= 2),
            "distinct_physical_tracks": p["distinct_physical_tracks"],
            "cross_track_reuses": p["cross_track_reuses"],
            "same_track_updates": p["same_track_updates"],
            "cross_track_updates": p["cross_track_updates"],
            "semantic_purity": p["semantic_purity"],
            "outcome_group": p["outcome_group"],
            "assoc_helpful": p["assoc_helpful"],
            "assoc_harmful": p["assoc_harmful"],
            "assoc_net_utility": p["assoc_net_utility"],
            "final_support": p["final_support"],
        })
    cts_fields = list(cts_rows[0].keys()) if cts_rows else ["tag"]
    old_cts = read_old("cross_track_support.csv")
    with open(out_root / "cross_track_support.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tag", "sem_id", "cross_track", "distinct_physical_tracks",
            "cross_track_reuses", "same_track_updates",
            "cross_track_updates", "semantic_purity", "outcome_group",
            "assoc_helpful", "assoc_harmful", "assoc_net_utility",
            "final_support"])
        w.writeheader()
        for r in old_cts + cts_rows:
            w.writerow({k: r.get(k, "") for k in cts_fields})

    # pollution_hubs.csv: top useful / top harmful cases
    harmful_key = lambda p: ((p["known_absorptions"] + p["fp_absorptions"])
                             + p["assoc_harmful"],
                             p["distinct_physical_tracks"])
    useful_key = lambda p: (p["same_real_class_reuses"] + p["assoc_helpful"],
                            p["distinct_physical_tracks"])
    top_harm = sorted([p for p in protos], key=harmful_key, reverse=True)[:20]
    top_use = sorted([p for p in protos], key=useful_key, reverse=True)[:20]
    hub_rows = []
    for rank, p in enumerate(top_harm, 1):
        hub_rows.append({k: p.get(k, "") for k in [
            "sem_id", "outcome_group", "distinct_physical_tracks",
            "known_absorptions", "fp_absorptions",
            "same_real_class_reuses", "cross_real_class_reuses",
            "assoc_helpful", "assoc_harmful", "assoc_net_utility",
            "final_support", "birth_gt_role"]} | {
            "tag": tag, "rank": rank, "hub_type": "harmful"})
    for rank, p in enumerate(top_use, 1):
        hub_rows.append({k: p.get(k, "") for k in [
            "sem_id", "outcome_group", "distinct_physical_tracks",
            "known_absorptions", "fp_absorptions",
            "same_real_class_reuses", "cross_real_class_reuses",
            "assoc_helpful", "assoc_harmful", "assoc_net_utility",
            "final_support", "birth_gt_role"]} | {
            "tag": tag, "rank": rank, "hub_type": "useful"})
    hub_fields = list(hub_rows[0].keys()) if hub_rows else ["tag"]
    old_hubs = read_old("pollution_hubs.csv")
    with open(out_root / "pollution_hubs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hub_fields)
        w.writeheader()
        for r in old_hubs + hub_rows:
            w.writerow({k: r.get(k, "") for k in hub_fields})

    summary = {
        "tag": tag,
        "events": len(events), "births": len(births),
        "updates": sum(len(v) for v in updates.values()),
        "reuses": sum(len(v) for v in reuses.values()),
        "decisions": len(decisions),
        "prototypes": len(protos),
        "birth_gt_role_distribution": dict(Counter(
            p["birth_gt_role"] for p in protos)),
        "event_join_missing": joined_missing,
        "event_join_missing_rate": round(
            joined_missing / max(event_total, 1), 4),
        "outcome_groups": dict(grp),
        "utility": utility_row,
        "intervention_effects": dict(Counter(r["effect"] for r in inter_rows)),
    }
    (out_root / f"offline_summary_{tag}.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("OFFLINE_AUDIT_DONE", tag)


if __name__ == "__main__":
    main()
