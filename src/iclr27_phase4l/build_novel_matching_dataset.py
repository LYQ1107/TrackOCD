"""Phase 4L Root Cause B: relative novel matching dataset.

Replays the Phase 4K J1b provenance event stream (causal, no future, no
GT) to reconstruct, for every novel matching query, the exact prototype
geometry the system saw at that time:

  best / second-best prototype cosine
  best-second margin
  prototype support, member count, member cosine distribution (radius)
  known-vs-novel relative evidence (novel - known)
  top-5 local density / entropy
  mutual consistency (query z-score inside member distribution)

GT is attached only offline (IoU >= 0.5) to label the query and the
causal member majority category.  No online method uses these labels.

Outputs:
  outputs/iclr27_phase4l/audit/novel_matching_pairs.csv
  outputs/iclr27_phase4l/audit/novel_similarity_distributions.csv
  outputs/iclr27_phase4l/audit/prototype_radius.csv
  outputs/iclr27_phase4l/audit/margin_analysis.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PROV = ROOT / "outputs" / "iclr27_phase4k" / "audit" / "prov_j1b"
EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
PRE_ASSOC = EXPORT / "pre_assoc_detections"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / \
    "detection_features"
TAO_JSON = EXPORT / "tao_subset" / "validation_20.json"
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
OUT = ROOT / "outputs" / "iclr27_phase4l" / "audit"
THETA = 0.6              # frozen min_birth_sim
NOVEL_UPDATE_RATE = 0.2  # frozen


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
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt()

    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    device = torch.device("cuda:0")
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    sem_mgr = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0")
    dino_cache = {}
    events = []
    for line in (PROV / "prototype_event_log_j1b.jsonl").read_text() \
            .splitlines():
        if line.strip():
            events.append(json.loads(line))
    embs = {i: np.asarray(x, dtype=np.float32) for i, x in enumerate(
        np.load(PROV / "embeddings_j1b.npz")["embeddings"])}
    track_rows = {}
    for p in sorted((PROV / "semantic_logs").glob("*.jsonl")):
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
        e["_abs"] = vorder[int(e["video_id"])] * 1_000_000 + \
            int(e["frame_id"])
    events.sort(key=lambda e: (e["_abs"], e["sem_id"]))

    # causal memory state (replicates NovelSemanticMemory exactly)
    protos = {}       # sem_id -> z (normalized running mean)
    members = defaultdict(list)   # sem_id -> list of z (float32)
    support = Counter()
    creator = {}
    pairs = []

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

    members_meta = defaultdict(list)   # sem_id -> [{"v","t","frame"}]
    member_gt_cat = defaultdict(list)  # sem_id -> [category or None]

    def member_gt(sem, frame_abs):
        """Causal majority GT-novel category of past members (offline)."""
        cats = [c for c in member_gt_cat[sem] if c is not None]
        cnt = Counter(cats)
        return cnt.most_common(1)[0][0] if cnt else None

    for e in events:
        sid = int(e["sem_id"])
        key = (int(e["video_id"]), int(e["frame_id"]),
               int(e["track_key"][1]))
        r = track_rows.get(key)
        if "z_idx" in e:
            z = embs[int(e["z_idx"])]
        elif r is not None:
            vid = int(e["video_id"])
            if vid not in dino_cache:
                f = np.load(FEAT_ROOT / str(vid) / "feats.npz")
                dino_cache[vid] = (
                    f["feats"].astype(np.float32),
                    {int(i): k for k, i in enumerate(
                        f["det_local_ids"].tolist())})
            feats, idx = dino_cache[vid]
            zi = idx.get(int(r["det_idx"]))
            z = sem_mgr.embed(feats[zi])[0] if zi is not None else None
        else:
            z = None
        if z is None:
            continue
        det_gt = None
        if r is not None:
            det_gt = match_gt(gt, int(r["image_id"]), r["bbox"])
        geo = geometry(z)
        if geo is not None and e["kind"] in ("birth", "reuse"):
            maj = member_gt(sid, e["_abs"])
            q_gt_role = (det_gt or {}).get("role", "fp")
            q_gt_cat = (det_gt or {}).get("category_id")
            if q_gt_role == "novel":
                if maj is not None and maj == q_gt_cat:
                    case = "SAME_NOVEL"
                else:
                    case = "DIFFERENT_NOVEL"
            elif q_gt_role == "known":
                case = "KNOWN_COLLISION"
            else:
                case = "FP_QUERY"
            same_track = creator.get(sid) == (
                int(e["video_id"]), int(e["track_key"][1]))
            pairs.append({
                "sem_id": sid, "video_id": int(e["video_id"]),
                "frame_id": int(e["frame_id"]),
                "track_id": int(e["track_key"][1]),
                "kind": e["kind"], "same_track": int(same_track),
                "absolute_existing": int(geo["best_cos"] >= THETA),
                "det_gt_role": q_gt_role,
                "det_gt_category": q_gt_cat or "",
                "proto_majority_category": maj if maj is not None else "",
                "case": case,
                "p_known": float(r["p_known"]) if r is not None else "",
                "best_known": float(r["best_known"]) if r is not None else "",
                "novel_minus_known": round(
                    geo["best_cos"] - float(r["best_known"]), 4)
                if r is not None else "",
                **{k: v for k, v in geo.items() if k != "best_id"},
                "best_id": geo["best_id"],
            })
        # apply the event's memory mutation (causal order)
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
        # reuse without update: no memory mutation

    # ---- write pairs ----
    if pairs:
        fields = list(pairs[0].keys())
        with open(OUT / "novel_matching_pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(pairs)

    # ---- distributions by case ----
    dist_rows = []
    for case in ("SAME_NOVEL", "DIFFERENT_NOVEL", "KNOWN_COLLISION",
                 "FP_QUERY"):
        sub = [p for p in pairs if p["case"] == case]
        if not sub:
            continue
        def stat(key):
            vals = [float(p[key]) for p in sub if p[key] != ""]
            return vals
        b = stat("best_cos"); m = stat("margin"); nm = stat(
            "novel_minus_known"); zs = stat("query_zscore")
        ent = stat("local_entropy")
        dist_rows.append({
            "case": case, "n": len(sub),
            "best_cos_mean": round(float(np.mean(b)), 4) if b else "",
            "best_cos_median": round(float(np.median(b)), 4) if b else "",
            "best_cos_p90": round(float(np.percentile(b, 90)), 4)
            if b else "",
            "margin_mean": round(float(np.mean(m)), 4) if m else "",
            "margin_median": round(float(np.median(m)), 4) if m else "",
            "novel_minus_known_mean": round(float(np.mean(nm)), 4)
            if nm else "",
            "query_zscore_mean": round(float(np.mean(zs)), 4)
            if zs else "",
            "local_entropy_mean": round(float(np.mean(ent)), 4)
            if ent else "",
        })
    with open(OUT / "novel_similarity_distributions.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)

    # ---- prototype radius (final causal member stats) ----
    radius_rows = []
    for sem in sorted(members):
        M = np.stack(members[sem])
        c = M @ M.T
        n = len(M)
        iu = np.triu_indices(n, 1)
        radius_rows.append({
            "sem_id": sem, "support": support[sem],
            "n_members": n,
            "member_pairwise_mean": round(float(c[iu].mean()), 4)
            if n >= 2 else "",
            "member_pairwise_min": round(float(c[iu].min()), 4)
            if n >= 2 else "",
            "member_pairwise_p90": round(float(np.percentile(c[iu], 90)),
                                         4) if n >= 2 else "",
            "distinct_physical_tracks": len({(m["v"], m["t"])
                                             for m in members_meta[sem]}),
            "distinct_videos": len({m["v"] for m in members_meta[sem]}),
            "gt_novel_member_share": round(
                sum(1 for c in member_gt_cat[sem] if c is not None) /
                max(n, 1), 4),
            "majority_gt_category": Counter(
                c for c in member_gt_cat[sem] if c is not None
            ).most_common(1)[0][0] if any(member_gt_cat[sem]) else "",
        })
    with open(OUT / "prototype_radius.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(radius_rows[0].keys()))
        w.writeheader()
        w.writerows(radius_rows)

    # ---- margin analysis ----
    margin_rows = []
    for case in ("SAME_NOVEL", "DIFFERENT_NOVEL"):
        sub = [p for p in pairs if p["case"] == case and
               not p["same_track"]]
        if not sub:
            continue
        m = [float(p["margin"]) for p in sub]
        for thr in (0.0, 0.02, 0.05, 0.1, 0.2):
            margin_rows.append({
                "case": case, "margin_threshold": thr,
                "n": len(sub),
                "fraction_above": round(
                    sum(1 for x in m if x > thr) / max(len(m), 1), 4),
            })
    with open(OUT / "margin_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
        w.writeheader()
        w.writerows(margin_rows)

    print("NOVEL_MATCHING_DATASET_DONE", len(pairs), "queries")


if __name__ == "__main__":
    main()
