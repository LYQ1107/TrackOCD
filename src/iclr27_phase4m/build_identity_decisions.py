"""Phase 4M forced-decision retrospective dataset.

Replays the corrected Phase 4L dev runs (j1b / b1 / b2) causally from
the provenance event stream and labels every novel identity decision:

  action EXISTING (reuse event) or NEW (birth event)
  GT diagnostic case (SAME_NOVEL / DIFFERENT_NOVEL / KNOWN_COLLISION /
                      FP_QUERY)
  decision class:
    CORRECT_EXISTING / WRONG_EXISTING / OVERBIRTH / CORRECT_NEW /
    KNOWN_BIRTH / FP_BIRTH

Geometry (best/second cosine, margin, entropy, known-relative margin,
prototype support/radius) is computed for every decision from the causal
prototype state at that time. GT is offline-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
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
    ms = members[best_id]
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embed-reuse", action="store_true")
    ap.add_argument("--feat-root", type=Path, default=ROOT /
                    "outputs/iclr27_phase4i/audit/detection_features")
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
            track_rows[(int(r["video_id"]), int(r["frame_id"]),
                        int(r["physical_track_id"]))] = r
    vorder = {}
    for e in events:
        vorder.setdefault(int(e["video_id"]), len(vorder))
    for e in events:
        e["_abs"] = vorder[int(e["video_id"])] * 1_000_000 + \
            int(e["frame_id"])
    events.sort(key=lambda e: (e["_abs"], e["sem_id"]))

    zmap = {}
    if args.embed_reuse:
        from src.orbit_mdc.evaluate_mdc import load_mdc_model
        from src.frame_online_trackocd.semantic import \
            build_semantic_manager
        device = torch.device("cuda:0")
        model, _ = load_mdc_model(
            str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"), device)
        model.eval()
        sem_mgr = build_semantic_manager(
            model, device, prefix_mode="P1", decision_threshold=0.30,
            commit_mode="M0")
        need = {}
        for e in events:
            if e["kind"] != "reuse":
                continue
            key = (int(e["video_id"]), int(e["frame_id"]),
                   int(e["track_key"][1]))
            r = track_rows.get(key)
            if r is not None:
                need.setdefault(int(e["video_id"]), set()).add(
                    int(r["det_idx"]))
        for vid, dids in need.items():
            f = np.load(args.feat_root / str(vid) / "feats.npz")
            feats = f["feats"].astype(np.float32)
            ids = f["det_local_ids"].tolist()
            idx = {int(i): k for k, i in enumerate(ids)}
            zlist = []
            for did in sorted(dids):
                zi = idx.get(did)
                zlist.append((did, feats[zi] if zi is not None else None))
            chunk = np.stack([x[1] for x in zlist if x[1] is not None])
            if len(chunk) == 0:
                continue
            zs = []
            for s in range(0, len(chunk), 512):
                x = torch.as_tensor(chunk[s:s + 512], dtype=torch.float32,
                                    device=device).view(-1, 1, 768)
                mask = torch.ones(x.shape[0], 1, dtype=torch.bool,
                                  device=device)
                with torch.no_grad():
                    out = model.aggregate(x, mask)
                zz = out["z"]
                if zz.dim() == 3:
                    zz = zz[:, 0]
                zs.append(zz.float().cpu().numpy())
            zs = np.concatenate(zs, axis=0)
            ci = 0
            for did, feat in zlist:
                if feat is not None:
                    zmap[(vid, did)] = zs[ci]
                    ci += 1

    protos = {}
    members = defaultdict(list)
    member_gt_cat = defaultdict(list)
    creator = {}
    rows = []

    def maj_cat(sid):
        cats = [c for c in member_gt_cat[sid] if c is not None]
        return Counter(cats).most_common(1)[0][0] if cats else None

    for e in events:
        sid = int(e["sem_id"])
        key = (int(e["video_id"]), int(e["frame_id"]),
               int(e["track_key"][1]))
        r = track_rows.get(key)
        det_gt = match_gt(gt, int(r["image_id"]), r["bbox"]) \
            if r is not None else None
        q_role = (det_gt or {}).get("role", "fp")
        q_cat = (det_gt or {}).get("category_id")
        z = embs[int(e["z_idx"])] if "z_idx" in e else None
        if z is None and e["kind"] == "reuse" and r is not None:
            z = zmap.get((int(e["video_id"]), int(r["det_idx"])))

        if e["kind"] == "birth":
            geo = geometry(z, protos, members) if z is not None else None
            best_id = geo["best_id"] if geo else None
            bmaj = maj_cat(best_id) if best_id is not None else None
            if q_role == "novel":
                if bmaj is not None and bmaj == q_cat:
                    dclass = "OVERBIRTH"
                else:
                    dclass = "CORRECT_NEW"
            elif q_role == "known":
                dclass = "KNOWN_BIRTH"
            else:
                dclass = "FP_BIRTH"
            creator[sid] = (int(e["video_id"]), int(e["track_key"][1]))
            protos[sid] = _norm(z.astype(np.float32))
            members[sid].append(z)
            member_gt_cat[sid].append(q_cat)
        elif e["kind"] == "update":
            # same-track continuation: mutate memory, not a new decision
            p = (1.0 - NOVEL_UPDATE_RATE) * protos[sid] + \
                NOVEL_UPDATE_RATE * z
            protos[sid] = _norm(p.astype(np.float32))
            members[sid].append(z)
            member_gt_cat[sid].append(q_cat)
            continue
        else:  # reuse
            if creator.get(sid) == (int(e["video_id"]),
                                    int(e["track_key"][1])):
                continue   # same-track sticky continuation
            bmaj = maj_cat(sid)
            geo = geometry(z, protos, members) if z is not None else None
            if q_role == "novel":
                if bmaj is not None and bmaj == q_cat:
                    dclass = "CORRECT_EXISTING"
                else:
                    dclass = "WRONG_EXISTING"
            else:
                dclass = "WRONG_EXISTING"

        rows.append({
            "tag": tag, "video_id": int(e["video_id"]),
            "frame_id": int(e["frame_id"]),
            "track_id": int(e["track_key"][1]),
            "sem_id": sid, "kind": e["kind"],
            "action": "EXISTING" if e["kind"] == "reuse" else "NEW",
            "decision_class": dclass,
            "query_gt_role": q_role,
            "query_gt_category": q_cat or "",
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
            "support_causal": (geo or {}).get("support_causal", ""),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("IDENTITY_DECISIONS_DONE", tag, len(rows))


if __name__ == "__main__":
    main()
