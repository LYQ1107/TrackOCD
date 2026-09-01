"""Phase 4M corrected forced-decision retrospective dataset.

The Phase 4L provenance stream labels *code branches*, not memory
actions:
  - a sticky reuse is a same-track continuation, NOT a new identity
    decision;
  - a "birth" event only creates a global prototype the first time its
    sem_id appears; later "birth" events with an existing sem_id were
    routed by NovelSemanticMemory.propose back to the existing prototype
    (the is_new return was discarded), so the actual memory action is
    EXISTING while the branch label says NEW.

This builder replays the event stream in true causal order, reconstructs
the exact online quantities (track prefix P1 = mean of the last <=8
single-frame z's, memory prototype state, members, support), and emits
one row per *actual* identity decision with the geometry the online
system had at that moment.

GT is used only for offline labeling (roles / categories).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
NOVEL_UPDATE_RATE = 0.2
PREFIX_LEN = 8


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt(tao_json):
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(tao_json.read_text())
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
    best, bi = None, 0.5
    for g in gt.get(image_id, []):
        v = iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def geometry(z, protos, members, member_tracks):
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
        m_max = float(Mc.max())
    else:
        m_mean = m_std = zscore = m_max = 0.0
    near = int(np.sum(cos >= best_cos - 0.05)) - 1
    tracks = member_tracks.get(best_id, set())
    return {
        "best_cos": best_cos, "second_cos": second_cos,
        "margin": best_cos - second_cos, "entropy": entropy,
        "query_zscore": round(zscore, 4),
        "member_mean_cos": round(m_mean, 4),
        "member_std_cos": round(m_std, 4),
        "member_max_cos": round(m_max, 4),
        "support_causal": len(ms) + 1,
        "proto_distinct_tracks": len(tracks),
        "near_count": near,
        "best_id": best_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prov-root", type=Path, required=True)
    ap.add_argument("--tao-json", type=Path, required=True)
    ap.add_argument("--z-cache", type=Path, default=ROOT /
                    "outputs/iclr27_phase4m/audit/det_z_cache")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tag = args.tag
    prov = args.prov_root
    gt = load_gt(args.tao_json)

    events = []
    for line in (prov / f"prototype_event_log_{tag}.jsonl").read_text() \
            .splitlines():
        if line.strip():
            events.append(json.loads(line))
    embs = {i: np.asarray(x, dtype=np.float32) for i, x in enumerate(
        np.load(prov / f"embeddings_{tag}.npz")["embeddings"])}
    track_rows = {}
    track_order = []
    track_sem_by_frame = defaultdict(dict)
    for p in sorted((prov / "semantic_logs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["video_id"]), int(r["frame_id"]),
                   int(r["physical_track_id"]))
            track_rows[key] = r
            track_order.append(key)
            tkey = (key[0], key[2])
            sid_s = r.get("semantic_id")
            if isinstance(sid_s, str) and sid_s.startswith("N"):
                try:
                    track_sem_by_frame[tkey][key[1]] = int(sid_s[1:])
                except ValueError:
                    pass
            elif isinstance(sid_s, str) and sid_s.startswith("K"):
                track_sem_by_frame[tkey][key[1]] = None

    track_frames = {t: sorted(fr.keys()) for t, fr in
                    track_sem_by_frame.items()}

    def prior_novel_sem(vid, tid, frame_id):
        """Track's novel semantic id at the last frame strictly before
        frame_id, or None (known / no assignment)."""
        frs = track_frames.get((vid, tid))
        if not frs:
            return None
        import bisect
        i = bisect.bisect_left(frs, frame_id)
        if i == 0:
            return None
        return track_sem_by_frame[(vid, tid)].get(frs[i - 1])

    # single-frame z map: (video_id, det_idx) -> z
    zmap = {}
    vids = sorted({int(e["video_id"]) for e in events})
    for vid in vids:
        f = np.load(args.z_cache / f"{vid}.npz")
        ids = f["det_local_ids"].tolist()
        frames = f["frame_orders"].tolist() \
            if "frame_orders" in f.files else None
        zs = f["z"].astype(np.float32)
        for i, did in enumerate(ids):
            fid = frames[i] if frames is not None else -1
            zmap[(vid, fid, did)] = zs[i]

    # per-track prefix reconstruction: last <=8 associated single-frame z's
    by_track = defaultdict(list)
    for key in track_order:
        by_track[(key[0], key[2])].append(key)
    for tkey in by_track:
        by_track[tkey].sort(key=lambda k: k[1])
    prefix_cache = {}
    for tkey, keys in by_track.items():
        hist = []
        for key in keys:
            r = track_rows[key]
            z = zmap.get((key[0], key[1], int(r["det_idx"])))
            if z is not None:
                hist.append(z)
                if len(hist) > PREFIX_LEN:
                    hist.pop(0)
            if hist:
                prefix_cache[key] = _norm(
                    np.mean(hist, axis=0).astype(np.float32))

    protos = {}
    members = defaultdict(list)
    member_tracks = defaultdict(set)
    member_gt_cat = defaultdict(list)
    created_by = {}
    rows = []

    def maj_cat(sid):
        cats = [c for c in member_gt_cat[sid] if c is not None]
        return Counter(cats).most_common(1)[0][0] if cats else None

    for e in events:
        sid = int(e["sem_id"])
        vid = int(e["video_id"])
        tid = int(e["track_key"][1])
        key = (vid, int(e["frame_id"]), tid)
        r = track_rows.get(key)
        det_gt = match_gt(gt, int(r["image_id"]), r["bbox"]) \
            if r is not None else None
        q_role = (det_gt or {}).get("role", "fp")
        q_cat = (det_gt or {}).get("category_id")
        z = embs[int(e["z_idx"])] if "z_idx" in e and \
            int(e["z_idx"]) >= 0 else None

        if e["kind"] == "update":
            if sid in protos and z is not None:
                p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                    NOVEL_UPDATE_RATE * z
                protos[sid] = _norm(p.astype(np.float32))
                members[sid].append(z)
                member_tracks[sid].add((vid, tid))
                member_gt_cat[sid].append(q_cat)
            continue

        if e["kind"] == "reuse":
            prior = prior_novel_sem(vid, tid, int(e["frame_id"]))
            # sticky continuation: the track was already assigned this
            # semantic identity at an earlier frame
            if prior == sid:
                continue
            if z is None:
                z = prefix_cache.get(key)
            geo = geometry(z, protos, members, member_tracks) \
                if z is not None else None
            bmaj = maj_cat(geo["best_id"]) if geo else None
            dclass = "CORRECT_EXISTING" if (
                q_role == "novel" and bmaj is not None
                and bmaj == q_cat) else "WRONG_EXISTING"
            rows.append(_row(tag, e, key, r, geo, "EXISTING_SOFT",
                             "EXISTING", dclass, q_role, q_cat, bmaj,
                             same_track_redecision=prior is not None
                             and prior != sid))
            continue

        # birth event
        is_first = sid not in protos
        if is_first:
            geo = geometry(z, protos, members, member_tracks) \
                if z is not None else None
            bmaj = maj_cat(geo["best_id"]) if geo else None
            prior = prior_novel_sem(vid, tid, int(e["frame_id"]))
            if q_role == "novel":
                dclass = "OVERBIRTH" if (bmaj is not None and
                                         bmaj == q_cat) else "CORRECT_NEW"
            elif q_role == "known":
                dclass = "KNOWN_BIRTH"
            else:
                dclass = "FP_BIRTH"
            rows.append(_row(tag, e, key, r, geo, "NEW_BRANCH", "NEW",
                             dclass, q_role, q_cat, bmaj,
                             same_track_redecision=prior is not None
                             and prior != sid))
            if z is not None:
                protos[sid] = _norm(z.astype(np.float32))
                members[sid].append(z)
                member_tracks[sid].add((vid, tid))
                member_gt_cat[sid].append(q_cat)
                created_by[sid] = (vid, tid)
            continue

        # duplicate birth: actual memory action is EXISTING
        prior = prior_novel_sem(vid, tid, int(e["frame_id"]))
        same_creator = (created_by.get(sid) == (vid, tid)
                        or prior == sid)
        geo = geometry(z, protos, members, member_tracks) \
            if z is not None else None
        if not same_creator:
            bmaj = maj_cat(geo["best_id"]) if geo else None
            dclass = "CORRECT_EXISTING" if (
                q_role == "novel" and bmaj is not None
                and bmaj == q_cat) else "WRONG_EXISTING"
            rows.append(_row(tag, e, key, r, geo, "NEW_BRANCH",
                             "EXISTING", dclass, q_role, q_cat, bmaj,
                             same_track_redecision=prior is not None
                             and prior != sid))
        if z is not None:
            p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                NOVEL_UPDATE_RATE * z
            protos[sid] = _norm(p.astype(np.float32))
            members[sid].append(z)
            member_tracks[sid].add((vid, tid))
            member_gt_cat[sid].append(q_cat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("IDENTITY_DECISIONS_V2_DONE", tag, len(rows))


def _row(tag, e, key, r, geo, branch, actual, dclass, q_role, q_cat,
         bmaj, same_track_redecision):
    return {
        "tag": tag, "video_id": key[0], "frame_id": key[1],
        "track_id": key[2], "sem_id": int(e["sem_id"]),
        "kind": e["kind"], "branch": branch,
        "actual_action": actual, "decision_class": dclass,
        "same_track_redecision": int(same_track_redecision),
        "query_gt_role": q_role, "query_gt_category": q_cat or "",
        "best_proto_id": (geo or {}).get("best_id", ""),
        "best_proto_majority_category": bmaj if bmaj is not None else "",
        "p_known": float(r["p_known"]) if r is not None else "",
        "det_score": float(r["score"]) if r is not None else "",
        "track_age": int(r["track_age"]) if r is not None else "",
        "best_cos": round((geo or {}).get("best_cos", -1.0), 4),
        "second_cos": round((geo or {}).get("second_cos", -1.0), 4),
        "margin": round((geo or {}).get("margin", -1.0), 4),
        "entropy": round((geo or {}).get("entropy", -1.0), 4),
        "novel_minus_known": round(
            (geo or {}).get("best_cos", -1.0) -
            (float(r["best_known"]) if r is not None else 0.0), 4),
        "query_zscore": (geo or {}).get("query_zscore", ""),
        "member_mean_cos": (geo or {}).get("member_mean_cos", ""),
        "member_std_cos": (geo or {}).get("member_std_cos", ""),
        "member_max_cos": (geo or {}).get("member_max_cos", ""),
        "support_causal": (geo or {}).get("support_causal", ""),
        "proto_distinct_tracks": (geo or {}).get(
            "proto_distinct_tracks", ""),
        "near_count": (geo or {}).get("near_count", ""),
    }


if __name__ == "__main__":
    main()
