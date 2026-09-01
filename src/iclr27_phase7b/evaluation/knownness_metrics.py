"""Phase 7B-specific knownness metrics on any replay CSV.

Computes, for a DEV/heldout stream with GT (evaluation only):
  - known explainability AUROC/AUPR (sem_kscore vs GT known/novel);
  - novel->known absorption rate;
  - known->novel error;
  - first-occurrence knownness calibration (mean kscore by GT role and
    reliability diagram ECE);
  - per-category first-track kscore for the absorbed novel categories.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import group_tracks, load_proposals
from src.iclr27_phase7a.evaluation.strict_eval_any import load_gt_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--video-ids", required=True)
    ap.add_argument("--proxy-auc", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = load_proposals(Path(args.proposals))
    tracks = group_tracks(all_rows)
    vids = json.loads(args.video_ids)
    stream, labels = load_gt_videos(vids)
    mapping = align_pred_to_gt(tracks, gt_track_boxes(stream))
    aligned_keys = set(mapping)

    recs = []
    for r in all_rows:
        key = (int(r["video_id"]), int(r["track_id"]))
        if key not in aligned_keys:
            continue
        sid = mapping[key]
        lab = labels[sid]
        if lab["protocol_role"] not in (
                "supported_known", "zero_shot_known", "novel"):
            continue
        recs.append({
            "key": key,
            "sample_id": sid,
            "frame": int(r["frame_id"]),
            "cat": int(lab["ground_truth_category_id"]),
            "is_novel": lab["protocol_role"] == "novel",
            "action": r.get("sem_action") or "unresolved",
            "sid": int(r["sem_sid"]) if r.get("sem_sid") not in ("", None)
            else None,
            "kscore": float(r.get("sem_kscore") or 0.0),
            "kmahal": float(r.get("sem_kmahal") or 0.0),
            "ksim": float(r.get("sem_ksim") or 0.0),
        })
    recs.sort(key=lambda x: (x["key"][0], x["frame"]))

    y = np.asarray([0 if x["is_novel"] else 1 for x in recs])
    s = np.asarray([x["kscore"] for x in recs])
    auc = aupr = None
    if len(np.unique(y)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auc = float(roc_auc_score(y, s))
        aupr = float(average_precision_score(y, s))

    novel = [x for x in recs if x["is_novel"]]
    known = [x for x in recs if not x["is_novel"]]
    absorption = (sum(1 for x in novel if x["action"] == "known")
                  / max(len(novel), 1))
    known_to_new = (sum(1 for x in known if x["action"] == "new")
                    / max(len(known), 1))
    known_to_existing = (sum(1 for x in known if x["action"] == "existing")
                         / max(len(known), 1))

    # first occurrence per GT category (chronological)
    first = {}
    for x in recs:
        first.setdefault((x["cat"], x["is_novel"]), x)
    first_known = [v for k, v in first.items() if not k[1]]
    first_novel = [v for k, v in first.items() if k[1]]

    # reliability diagram ECE over kscore bins (known==1)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.searchsorted(bins, s, side="right") - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = float(s[m].mean())
        acc = float(y[m].mean())
        ece += float((m.sum() / len(s)) * abs(conf - acc))

    absorbed_cats = defaultdict(list)
    for x in novel:
        if x["action"] == "known":
            absorbed_cats[x["cat"]].append(x)

    proxy = None
    if args.proxy_auc:
        p = json.loads(Path(args.proxy_auc).read_text())
        proxy = {
            "auc": p.get("knownness_auc", {}).get("auc_known_vs_proxy"),
            "aupr": p.get("knownness_auc", {}).get("aupr_known_vs_proxy"),
        }

    out = {
        "n_records": len(recs),
        "n_known": len(known),
        "n_novel": len(novel),
        "known_auroc": auc,
        "known_aupr": aupr,
        "novel_to_known_absorption": absorption,
        "known_to_new_rate": known_to_new,
        "known_to_existing_rate": known_to_existing,
        "ece": ece,
        "first_known_mean_kscore": float(np.mean([x["kscore"]
                                                  for x in first_known])),
        "first_novel_mean_kscore": float(np.mean([x["kscore"]
                                                  for x in first_novel])),
        "first_known_acc": float(np.mean([
            x["action"] == "known" and x["sid"] == x["cat"]
            for x in first_known])),
        "first_novel_birth_acc": float(np.mean([
            x["action"] == "new" for x in first_novel])),
        "first_known_kscore": [
            {"cat": x["cat"], "kscore": x["kscore"], "action": x["action"],
             "sid": x["sid"]} for x in first_known],
        "first_novel_kscore": [
            {"cat": x["cat"], "kscore": x["kscore"], "action": x["action"],
             "sid": x["sid"]} for x in first_novel],
        "absorbed_novel_categories": {
            str(c): {
                "n_rows": len(v),
                "first_kscore": v[0]["kscore"],
                "actions": [x["action"] for x in v[:10]],
                "sids": [x["sid"] for x in v[:10]],
            } for c, v in sorted(absorbed_cats.items())},
        "proxy_knownness": proxy,
        "transfer_gap_auroc": (
            auc - proxy["auc"] if auc is not None and proxy
            and proxy["auc"] is not None else None),
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, list)}, indent=2, default=float))


if __name__ == "__main__":
    main()
