"""Phase 4M: forced-decision retrospective audit dataset on corrected streams.

Decision geometry is reconstructed with the exact online quantity used by
Phase 4L: the track-local causal prefix z = P1 mean of the last <=8
single-frame M2 embeddings (TrackSemState.prefix), NOT the single-frame z.
Reuse/update events carry the online prefix cosine (`compat` /
`compat_best`) which we use to validate the replay.

Replays a corrected Phase 4L provenance event stream (j1b / b1 / b2) in
causal order and, for every global novel identity decision (birth =
NEW_NOVEL, reuse = EXISTING_NOVEL), records the decision-time geometry and
attaches an offline GT diagnostic outcome:

  reuse  + novel + majority category matches  -> CORRECT_EXISTING
  reuse  + novel + majority category differs  -> WRONG_EXISTING
  reuse  + known                              -> KNOWN_COLLISION
  reuse  + fp                                 -> FP_QUERY
  birth  + novel + same category already in causal memory -> OVERBIRTH
  birth  + novel + no matching existing category -> CORRECT_NEW
  birth  + known                              -> KNOWN_BIRTH
  birth  + fp                                 -> FP_BIRTH

GT is used only offline for diagnosis. No online method consumes it here.

Outputs:
  outputs/iclr27_phase4m/audit/identity_decisions_{tag}.csv
  outputs/iclr27_phase4m/audit/overbirth_events_{tag}.csv
  outputs/iclr27_phase4m/audit/wrong_reuse_events_{tag}.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PROV_ROOT = ROOT / "outputs" / "iclr27_phase4l" / "audit"
EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / \
    "detection_features"
TAO_JSON = EXPORT / "tao_subset" / "validation_20.json"
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
OUT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DET_Z_CACHE = OUT / "det_z_cache"
THETA = 0.6
NOVEL_UPDATE_RATE = 0.2


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
    best, bi = None, 0.5
    for g in gt.get(image_id, []):
        v = iou(bbox, g["bbox"])
        if v >= bi:
            bi, best = v, g
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="j1b",
                    choices=["j1b", "b1", "b2", "m0", "m1", "m2", "m3"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prov-root", type=Path, default=None)
    args = ap.parse_args()
    tag = args.tag
    if args.prov_root is not None:
        prov_root = args.prov_root
    elif tag.startswith("m"):
        prov_root = ROOT / "outputs" / "iclr27_phase4m" / "audit"
    else:
        prov_root = PROV_ROOT
    PROV = prov_root / f"prov_dev_{tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    prov = PROV
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    sem_mgr = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0")
    dino_cache = {}

    events = []
    for line in (prov / f"prototype_event_log_{tag}.jsonl").read_text().splitlines():
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
            track_rows[(int(r["video_id"]), int(r["frame_id"]),
                        int(r["physical_track_id"]))] = r
    vorder = {}
    for e in events:
        vorder.setdefault(int(e["video_id"]), len(vorder))
    for e in events:
        e["_abs"] = vorder[int(e["video_id"])] * 1_000_000 + int(e["frame_id"])
    events.sort(key=lambda e: (e["_abs"], e["sem_id"]))

    z_cache = {}

    def cache_z(vid, frame, det_idx):
        """z for a detection from the precomputed single-frame M2 cache.

        `det_idx` is the per-frame raw detection position
        (last_raw_positions / det_local_id within that frame).  The
        cache stores rows in global per-video JSONL order, so we map
        (frame, det_idx) -> global row via frame_orders; det_local_ids
        repeat every frame and must NOT be used as global positions.
        """
        if vid not in z_cache:
            f = np.load(DET_Z_CACHE / f"{vid}.npz")
            fo = np.load(FEAT_ROOT / str(vid) / "feats.npz")["frame_orders"]
            fpos = {}
            for frame_id in np.unique(fo):
                fpos[int(frame_id)] = np.where(fo == frame_id)[0]
            z_cache[vid] = (f["z"].astype(np.float32), fpos)
        zs, fpos = z_cache[vid]
        rows = fpos.get(int(frame))
        if rows is None:
            return None
        k = int(det_idx)
        return zs[rows[k]] if 0 <= k < len(rows) else None

    # Per-track observation timelines from the semantic logs.  Rows are
    # (frame_id, det_idx) in causal order; TrackSemState keeps only the
    # last 8 observations and is pruned after a >=10-frame gap, which is
    # replicated in prefix_z() below.
    timelines = defaultdict(list)
    for p in sorted((prov / "semantic_logs").glob("*.jsonl")):
        vid = int(p.stem)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r.get("physical_track_id")
            if tid is not None and tid >= 0:
                timelines[(vid, int(tid))].append(
                    (int(r["frame_id"]), int(r["det_idx"])))
    for key in timelines:
        timelines[key].sort()
    _tl_frames = {key: [x[0] for x in rows]
                  for key, rows in timelines.items()}

    def prefix_z(video, track, frame):
        """Causal P1 prefix embedding of a physical track at `frame`."""
        tl = timelines.get((video, track))
        if not tl:
            return None
        frames = _tl_frames[(video, track)]
        idx = bisect.bisect_right(frames, frame) - 1
        if idx < 0 or frames[idx] != frame:
            return None
        zs = []
        prev = None
        for j in range(idx, -1, -1):
            f, det = tl[j]
            if prev is not None and prev - f >= 10:
                # TrackSemState was pruned before this older observation.
                break
            z = cache_z(video, f, det)
            if z is None:
                return None
            zs.append(z)
            prev = f
            if len(zs) >= 8:
                break
        if not zs:
            return None
        return _norm(np.mean(np.stack(zs), axis=0).astype(np.float32))

    # Validation counters for the prefix reconstruction.
    val_prefix_vs_embs = []
    val_compat_diff = []
    n_prefix_used = 0
    n_fallback_single = 0

    protos = {}
    members = defaultdict(list)
    members_meta = defaultdict(list)
    member_gt_cat = defaultdict(list)
    support = Counter()
    creator = {}
    decisions = []

    def geometry(z):
        if not protos:
            return None
        P = np.stack(list(protos.values())).astype(np.float32)
        ids = list(protos)
        cos = (P @ z).astype(np.float64)
        order = np.argsort(cos)[::-1]
        top5 = [float(cos[order[k]]) if k < len(order) else -1.0
                for k in range(5)]
        soft = np.exp(np.clip(top5, -30, 30))
        soft = soft / soft.sum()
        entropy = float(-(soft * np.log(soft + 1e-12)).sum())
        best = int(ids[order[0]])
        second = int(ids[order[1]]) if len(order) >= 2 else -1
        best_cos = float(cos[order[0]])
        second_cos = float(cos[order[1]]) if len(order) >= 2 else -1.0
        ms = members[best]
        if ms:
            M = np.stack(ms)
            Mc = M @ z
            m_mean = float(Mc.mean())
            m_std = float(Mc.std()) + 1e-6
            zscore = float((best_cos - m_mean) / m_std)
            m_min = float(Mc.min())
            m_max = float(Mc.max())
        else:
            m_mean = m_std = zscore = m_min = m_max = -1.0
        distinct_tracks = len({(int(m["v"]), int(m["t"]))
                               for m in members_meta[best]})
        return {
            "best_cos": best_cos, "best_id": best,
            "second_cos": second_cos, "second_id": second,
            "margin": best_cos - second_cos,
            "top1": top5[0], "top2": top5[1], "top3": top5[2],
            "top4": top5[3], "top5": top5[4],
            "local_entropy": entropy,
            "near_count": sum(1 for c in cos if float(c) >=
                              max(THETA, best_cos - 0.05)),
            "support_causal": support[best] + 1,
            "member_count": len(ms),
            "member_mean_cos": round(m_mean, 4),
            "member_std_cos": round(m_std, 4),
            "member_min_cos": round(m_min, 4),
            "member_max_cos": round(m_max, 4),
            "query_zscore": round(zscore, 4),
            "proto_distinct_tracks": distinct_tracks,
        }

    def member_gt(sem):
        cats = [c for c in member_gt_cat[sem] if c is not None]
        cnt = Counter(cats)
        return cnt.most_common(1)[0][0] if cnt else None

    def existing_categories():
        return {member_gt(s) for s in protos if member_gt(s) is not None}

    for i, e in enumerate(events):
        if i % 10000 == 0:
            print(f"PHASE4M_PROGRESS {tag} {i}/{len(events)}", flush=True)
        sid = int(e["sem_id"])
        key = (int(e["video_id"]), int(e["frame_id"]),
               int(e["track_key"][1]))
        r = track_rows.get(key)
        if "z_idx" in e:
            z = embs[int(e["z_idx"])]
            z_prefix_recon = prefix_z(
                int(e["video_id"]), int(e["track_key"][1]),
                int(e["frame_id"]))
            if z_prefix_recon is not None:
                val_prefix_vs_embs.append(float(
                    z_prefix_recon @ z / max(
                        np.linalg.norm(z_prefix_recon) *
                        np.linalg.norm(z), 1e-12)))
            z_src = "embs"
        elif r is not None:
            vid = int(e["video_id"])
            z = prefix_z(vid, int(e["track_key"][1]), int(e["frame_id"]))
            z_src = "prefix"
            if z is not None:
                n_prefix_used += 1
            if z is None:
                # fallback: live single-frame M2 embed (rare)
                z = cache_z(vid, int(e["frame_id"]), int(r["det_idx"]))
                z_src = "single"
                if z is not None:
                    n_fallback_single += 1
            if z is None:
                if vid not in dino_cache:
                    f = np.load(FEAT_ROOT / str(vid) / "feats.npz")
                    fo = f["frame_orders"]
                    fpos = {}
                    for frame_id in np.unique(fo):
                        fpos[int(frame_id)] = np.where(fo == frame_id)[0]
                    dino_cache[vid] = (f["feats"].astype(np.float32), fpos)
                feats, fpos = dino_cache[vid]
                rows = fpos.get(int(e["frame_id"]))
                zi = int(r["det_idx"])
                if rows is not None and 0 <= zi < len(rows):
                    z = sem_mgr.embed(feats[rows[zi]])[0]
                else:
                    z = None
                z_src = "embed"
        else:
            z = None
        if e["kind"] in ("reuse", "update"):
            online = e.get("compat" if e["kind"] == "reuse" else "compat_best")
            if isinstance(online, (int, float)) and z is not None and protos:
                P = np.stack(list(protos.values())).astype(np.float32)
                val_compat_diff.append(float((P @ z).max()) - float(online))
        online_compat = (e.get("compat") if e["kind"] == "reuse"
                         else e.get("compat_best"))
        det_gt = None
        if r is not None:
            det_gt = match_gt(gt, int(r["image_id"]), r["bbox"])
        # Geometry is only needed for actual identity decisions; update
        # events only mutate memory (this removes ~30k redundant stacks).
        geo = geometry(z) if z is not None and e["kind"] in ("birth", "reuse") \
            else None
        if geo is not None and e["kind"] in ("birth", "reuse"):
            q_gt_role = (det_gt or {}).get("role", "fp")
            q_gt_cat = (det_gt or {}).get("category_id")
            mem_cat_ids = [s for s in protos
                           if member_gt(s) == q_gt_cat] \
                if q_gt_cat is not None else []
            gt_cat_mem_cos = ""
            gt_cat_mem_id = ""
            if mem_cat_ids:
                Pc = np.stack([protos[s] for s in mem_cat_ids]
                              ).astype(np.float32)
                cc = Pc @ z
                j = int(np.argmax(cc))
                gt_cat_mem_id = mem_cat_ids[j]
                gt_cat_mem_cos = round(float(cc[j]), 4)
            maj = member_gt(sid)
            action = "NEW_NOVEL" if e["kind"] == "birth" else "EXISTING_NOVEL"
            if e["kind"] == "reuse":
                if q_gt_role == "novel":
                    if maj is not None and maj == q_gt_cat:
                        outcome = "CORRECT_EXISTING"
                    else:
                        outcome = "WRONG_EXISTING"
                elif q_gt_role == "known":
                    outcome = "KNOWN_COLLISION"
                else:
                    outcome = "FP_QUERY"
            else:
                if q_gt_role == "novel":
                    if q_gt_cat in existing_categories():
                        outcome = "OVERBIRTH"
                    else:
                        outcome = "CORRECT_NEW"
                elif q_gt_role == "known":
                    outcome = "KNOWN_BIRTH"
                else:
                    outcome = "FP_BIRTH"
            same_track = creator.get(sid) == (
                int(e["video_id"]), int(e["track_key"][1]))
            decisions.append({
                "tag": tag,
                "sem_id": sid, "video_id": int(e["video_id"]),
                "frame_id": int(e["frame_id"]),
                "track_id": int(e["track_key"][1]),
                "kind": e["kind"], "action": action,
                "same_track": int(same_track),
                "absolute_existing": int(geo["best_cos"] >= THETA),
                "rule_existing": int(
                    (tag == "j1b" and geo["best_cos"] >= THETA) or
                    (tag == "b1" and geo["best_cos"] >= THETA and
                     geo["margin"] >= 0.05) or
                    (tag == "b2" and geo["best_cos"] >= THETA and
                     geo["local_entropy"] <= 1.6) or
                    (tag in ("m0", "m1", "m2", "m3") and
                     geo["best_cos"] >= THETA)),
                "z_src": z_src,
                "online_compat": online_compat if isinstance(
                    online_compat, (int, float)) else "",
                "det_gt_role": q_gt_role,
                "det_gt_category": q_gt_cat or "",
                "proto_majority_category": maj if maj is not None else "",
                "gt_cat_mem_id": gt_cat_mem_id,
                "gt_cat_mem_cos": gt_cat_mem_cos,
                "n_gt_cat_mem": len(mem_cat_ids),
                "outcome": outcome,
                "routing_confidence": float(r["p_known"]) if r is not None else "",
                "decision_threshold": float(r["decision_threshold"])
                if r is not None else 0.30,
                "best_known": float(r["best_known"]) if r is not None else "",
                "novel_minus_known": round(
                    geo["best_cos"] - float(r["best_known"]), 4)
                if r is not None else "",
                **{k: v for k, v in geo.items() if k != "best_id"},
                "best_id": geo["best_id"],
            })
        # causal memory mutation
        if z is None:
            print(f"PHASE4M_WARN_MISSING_Z {tag} "
                  f"event={i} kind={e['kind']} sem={sid} "
                  f"v={e['video_id']} f={e['frame_id']} "
                  f"t={e['track_key'][1]}", flush=True)
            continue
        if e["kind"] == "birth":
            creator[sid] = (int(e["video_id"]), int(e["track_key"][1]))
            protos[sid] = _norm(z.astype(np.float32))
            members[sid].append(z)
            members_meta[sid].append({
                "v": int(e["video_id"]), "t": int(e["track_key"][1]),
                "frame": int(e["frame_id"])})
            member_gt_cat[sid].append((det_gt or {}).get("category_id"))
            support[sid] += 1
        elif e["kind"] == "update":
            p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                NOVEL_UPDATE_RATE * z
            protos[sid] = _norm(p.astype(np.float32))
            members[sid].append(z)
            members_meta[sid].append({
                "v": int(e["video_id"]), "t": int(e["track_key"][1]),
                "frame": int(e["frame_id"])})
            member_gt_cat[sid].append((det_gt or {}).get("category_id"))
            support[sid] += 1

    if decisions:
        fields = list(decisions[0].keys())
        with open(OUT / f"identity_decisions_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(decisions)
        over = [d for d in decisions if d["outcome"] == "OVERBIRTH"]
        wrong = [d for d in decisions if d["outcome"] == "WRONG_EXISTING"]
        for name, sub in (("overbirth_events", over),
                          ("wrong_reuse_events", wrong)):
            with open(OUT / f"{name}_{tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(sub)
    counts = Counter(d["outcome"] for d in decisions)
    if val_prefix_vs_embs:
        a = np.asarray(val_prefix_vs_embs)
        print("PHASE4M_PREFIX_VS_EMBS_COSINE "
              f"n={len(a)} mean={a.mean():.6f} min={a.min():.6f} "
              f"p1={np.percentile(a, 1):.6f}", flush=True)
    if val_compat_diff:
        d = np.asarray([x for x in val_compat_diff if x == x])
        print("PHASE4M_REPLAY_VS_ONLINE_COMPAT_DIFF "
              f"n={len(d)} mean={d.mean():.6f} std={d.std():.6f} "
              f"min={d.min():.6f} max={d.max():.6f} "
              f"p99={np.percentile(np.abs(d), 99):.6f}", flush=True)
    print("PHASE4M_PREFIX_STATS "
          f"prefix_used={n_prefix_used} fallback_single={n_fallback_single}",
          flush=True)
    print("PHASE4M_DECISION_DATASET_DONE", tag, len(decisions), dict(counts))


if __name__ == "__main__":
    main()
