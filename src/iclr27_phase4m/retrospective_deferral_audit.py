"""Phase 4M retrospective deferral oracle.

For every real novel-query identity decision in the corrected decision
dataset we ask: if the irreversible EXISTING/NEW resolution had been
deferred at frame t, would the same physical track's later causal
prefixes (first observation at or after t+1/t+2/t+4/t+8) resolve the
identity correctly under the anchor rule (best cosine >= 0.6 ->
EXISTING, else NEW)?

Counterfactual: the deferred track contributes no global memory writes
after t (its birth/update events are skipped), so its own evidence cannot
be self-confirming through prototype updates.  Other tracks' events are
held fixed (documented approximation).  GT is used only for offline
scoring.
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
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
NOVEL_UPDATE_RATE = 0.2
PREFIX_LEN = 8
THETA = 0.6


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


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
        v = _iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


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
        "margin": best_cos - second_cos, "entropy": entropy,
        "query_zscore": round(zscore, 4),
        "member_mean_cos": round(m_mean, 4),
        "member_std_cos": round(m_std, 4),
        "support_causal": len(ms) + 1,
        "best_id": best_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prov-root", type=Path, required=True)
    ap.add_argument("--tao-json", type=Path, required=True)
    ap.add_argument("--z-cache", type=Path, default=ROOT /
                    "outputs/iclr27_phase4m/audit/det_z_cache")
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
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
    for p in sorted((prov / "semantic_logs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["video_id"]), int(r["frame_id"]),
                   int(r["physical_track_id"]))
            track_rows[key] = r

    zmap = {}
    for vid in sorted({int(e["video_id"]) for e in events}):
        f = np.load(args.z_cache / f"{vid}.npz")
        ids = f["det_local_ids"].tolist()
        frames = f["frame_orders"].tolist() \
            if "frame_orders" in f.files else None
        zs = f["z"].astype(np.float32)
        for i, did in enumerate(ids):
            fid = frames[i] if frames is not None else -1
            zmap[(vid, fid, did)] = zs[i]

    by_track = defaultdict(list)
    for key in track_rows:
        by_track[(key[0], key[2])].append(key)
    track_frames = {}
    for tkey, keys in by_track.items():
        keys.sort(key=lambda k: k[1])
        track_frames[tkey] = keys
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

    decisions = list(csv.DictReader(open(args.decisions)))
    # only real novel-query decisions (identity resolution target)
    novel = [r for r in decisions
             if r["query_gt_role"] == "novel"]
    # track GT category: use majority category of the decision's physical
    # track from all its semantic-log GT matches
    track_gt_cat = {}
    for key, r in track_rows.items():
        g = match_gt(gt, int(r["image_id"]), r["bbox"])
        if g is not None and g["role"] == "novel":
            track_gt_cat[(key[0], key[2])] = int(g["category_id"])

    def maj_cat(mgc):
        cats = [c for c in mgc if c is not None]
        return Counter(cats).most_common(1)[0][0] if cats else None

    def next_row(tkey, min_frame):
        keys = track_frames.get(tkey)
        if not keys:
            return None
        i = bisect.bisect_left([k[1] for k in keys], min_frame)
        return keys[i] if i < len(keys) else None

    out_rows = []
    summary = []
    ks = [1, 2, 4, 8]
    for nr, dec in enumerate(novel):
        vid = int(dec["video_id"])
        fid = int(dec["frame_id"])
        tid = int(dec["track_id"])
        tkey = (vid, tid)
        tcat = track_gt_cat.get(tkey)
        if tcat is None:
            continue
        # per-task counterfactual replay: skip this track's global memory
        # writes at/after t
        protos = {}
        members = defaultdict(list)
        mgc = defaultdict(list)
        resolved = {}
        vorder = {}
        for e in events:
            vorder.setdefault(int(e["video_id"]), len(vorder))

        def vkey(v, f):
            return (vorder.get(v, 10 ** 9), f)

        pending = []
        for k in ks:
            key = next_row(tkey, fid + k)
            if key is None:
                resolved[k] = "NO_OBS"
            else:
                pending.append((vkey(key[0], key[1]), k))
        pending.sort()
        cur = vkey(-1, -1)

        def evaluate_target(k):
            key = next_row(tkey, fid + k)
            if key is None:
                resolved[k] = "NO_OBS"
                return
            z = prefix_cache.get(key)
            geo = geometry(z, protos, members) if z is not None else None
            if geo is None:
                resolved[k] = "NO_MEMORY"
                return
            best = geo["best_cos"]
            bmaj = maj_cat(mgc.get(geo["best_id"], []))
            if best >= THETA and bmaj == tcat:
                resolved[k] = "CORRECT_EXISTING"
            elif best >= THETA:
                resolved[k] = "WRONG_EXISTING"
            elif bmaj == tcat:
                resolved[k] = "OVERBIRTH"
            else:
                resolved[k] = "CORRECT_NEW"

        for e in events:
            ev = vkey(int(e["video_id"]), int(e["frame_id"]))
            while pending and pending[0][0] < ev:
                evaluate_target(pending.pop(0)[1])
            if int(e["video_id"]) == vid and \
                    int(e["track_key"][1]) == tid and \
                    int(e["frame_id"]) >= fid:
                continue  # deferred track contributes no writes from t on
            sid = int(e["sem_id"])
            z = embs[int(e["z_idx"])] if "z_idx" in e and \
                int(e["z_idx"]) >= 0 else None
            if e["kind"] == "birth":
                if sid not in protos and z is not None:
                    protos[sid] = _norm(z.astype(np.float32))
                    members[sid].append(z)
                    mgc[sid].append(_cat_for(e, track_rows, gt))
            elif e["kind"] == "update":
                if sid in protos and z is not None:
                    p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                        NOVEL_UPDATE_RATE * z
                    protos[sid] = _norm(p.astype(np.float32))
                    members[sid].append(z)
                    mgc[sid].append(_cat_for(e, track_rows, gt))
        while pending:
            evaluate_target(pending.pop(0)[1])
        # track termination / no future observation
        last_row = track_frames.get(tkey)
        term_frame = last_row[-1][1] if last_row else fid
        for k in ks:
            out = resolved.get(k, "NO_OBS")
            obs = next_row(tkey, fid + k)
            row = {
                "tag": tag, "video_id": vid, "frame_id": fid,
                "track_id": tid, "decision_class": dec["decision_class"],
                "actual_action": dec["actual_action"],
                "k": k, "gt_category": tcat,
                "future_frame": obs[1] if obs else "",
                "outcome": out,
                "track_terminated": int(term_frame < fid + k),
                "correct": int(out in ("CORRECT_EXISTING",
                                       "CORRECT_NEW")),
            }
            out_rows.append(row)
        correct_later = any(
            resolved.get(k) in ("CORRECT_EXISTING", "CORRECT_NEW")
            for k in ks)
        summary.append({
            "tag": tag, "video_id": vid, "frame_id": fid,
            "track_id": tid, "decision_class": dec["decision_class"],
            "gt_category": tcat,
            "resolved_correctly_by_t8": int(correct_later),
            "terminated_before_t8": int(term_frame < fid + 8),
        })
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.out_prefix) + f"_k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(str(args.out_prefix) + f"_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("RETRO_DEFERRAL_DONE", tag, len(novel), len(out_rows))


def _cat_for(e, track_rows, gt):
    key = (int(e["video_id"]), int(e["frame_id"]),
           int(e["track_key"][1]))
    r = track_rows.get(key)
    if r is None:
        return None
    g = match_gt(gt, int(r["image_id"]), r["bbox"])
    return g["category_id"] if g is not None else None


if __name__ == "__main__":
    main()
