"""Proposal-aware CLS/ROI correspondence diagnostics (labels for audit only)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def l2(x):
    x = np.asarray(x, dtype=np.float32); return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def correspondence(rows, feat, role_videos=None):
    vids = set(role_videos or [])
    tracks = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("gt_role") not in ("known", "supported_known"): continue
        if vids and int(r["video_id"]) not in vids: continue
        c = int(r.get("gt_category_id", -1)); tracks[(int(r["video_id"]), int(r["track_id"]))].append(i)
    items = [(k, l2(feat[idx].mean(axis=0)), int(rows[idx[0]]["gt_category_id"])) for k, idx in tracks.items()]
    pairs, labels, r1 = [], [], []
    for i, (key, v, c) in enumerate(items):
        cand = [j for j, (k, _, _) in enumerate(items) if j != i and k[0] != key[0]]; pos = {j for j in cand if items[j][2] == c}
        if not pos: continue
        ranked = sorted(cand, key=lambda j: (-float(v @ items[j][1]), j)); r1.append(float(ranked[0] in pos))
        for j in cand: pairs.append(float(v @ items[j][1])); labels.append(int(items[j][2] == c))
    s, y = np.asarray(pairs), np.asarray(labels)
    valid = len(np.unique(y)) > 1
    return {"tracks": len(items), "videos": len({k[0] for k, _, _ in items}), "categories": len({c for _, _, c in items}), "cross_video_pairs": len(s), "positive_pairs": int(y.sum()) if len(y) else 0, "r1": float(np.mean(r1)) if r1 else None, "roc_auc": float(roc_auc_score(y, s)) if valid else None, "pr_auc": float(average_precision_score(y, s)) if valid else None, "positive_negative_gap": float(s[y == 1].mean() - s[y == 0].mean()) if valid else None}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--proposals", required=True); ap.add_argument("--features", required=True); ap.add_argument("--roles", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / args.proposals).open())); z = np.load(ROOT / args.features, allow_pickle=False); role = json.load((ROOT / args.roles).open())["roles"]
    value = {"protocol": "trackocd_iclr27_phase15s16", "rows": len(rows), "features": {m: correspondence(rows, z[m], role["known_bank_train"]) for m in ("cls", "roi")}, "calibration_role": {m: correspondence(rows, z[m], role["known_calibration"]) for m in ("cls", "roi")}, "audit_role": {m: correspondence(rows, z[m], role["known_audit"]) for m in ("cls", "roi")}, "gt_labels_used_for_alignment_only": True, "q1_label_used": False}
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); tmp.replace(out); print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
