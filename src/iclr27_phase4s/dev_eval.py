"""TrackOCD dev evaluation of the trained semantic core over the frozen Q1
physical stream (Q2-alpha0.1 optional cross-frontend check).

Strictly online: per physical track the model accumulates belief frame by
frame; global novel memory evolves causally; GT is used only offline for
scoring (TrackOCD-v1.0 corrected evaluator) and for the frozen r_phys
calibration on dev validity labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.baselines import b0_episode_track
from src.iclr27_phase4s.episodes import load_episodic_universe
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4s.train import build_known_matrix
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def r_phys_calibration(rows):
    """Frozen dev calibration of physical reliability from (score, prior_hits).
    Target = valid-vs-FP only (no novel semantic labels). Logistic, closed form."""
    x, y = [], []
    for r in rows:
        s = max(0.0, min(1.0, float(r["score"])))
        p = min(float(r["prior_hits"]), 20.0)
        x.append([np.log(max(s, 1e-4)), np.log1p(p), s, p])
        y.append(1.0 if r["gt_role"] in ("known", "novel") else 0.0)
    X = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    # least squares on logit via iterative reweighting is overkill; simple
    # logistic with sklearn-free Newton steps (10 iterations)
    w = np.zeros(X.shape[1])
    for _ in range(20):
        z = X @ w
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (prob - yv) / len(yv)
        H = (X.T * prob * (1 - prob)) @ X / len(yv) + 1e-4 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, grad)
    return w


def compute_r_phys(rows, w):
    probs = []
    for r in rows:
        s = max(0.0, min(1.0, float(r["score"])))
        p = min(float(r["prior_hits"]), 20.0)
        z = w[0] * np.log(max(s, 1e-4)) + w[1] * np.log1p(p) + w[2] * s + w[3] * p
        probs.append(1.0 / (1.0 + np.exp(-z)))
    probs = np.asarray(probs)
    valid = np.asarray([r["gt_role"] in ("known", "novel") for r in rows])
    fp = ~valid
    p_hi = float(np.median(probs[valid])) if valid.any() else 0.5
    p_lo = float(np.median(probs[fp])) if fp.any() else 0.1
    # monotone affine map onto the episodic r_phys scale (valid median -> 0.75,
    # FP median -> 0.05), clipped; AUROC unchanged
    span = max(p_hi - p_lo, 1e-4)
    out = {}
    for r, pv in zip(rows, probs):
        out[id(r)] = float(np.clip(0.05 + 0.7 * (pv - p_lo) / span, 0.05, 0.95))
    return out


def run_frontend(
    model, tracks, feats_by_key, r_phys_map, known_list, known_cat_index,
    mode="b3", raw_protos=None, tau_known=0.55, tau_novel=0.45,
):
    """Online semantic decisions per physical track. Returns per-track results."""
    results = {}
    commits = []  # (track_key, kind, slot_id_or_cat, commit_age, r_phys)
    memory = NovelMemory(model.parameters().__next__().device)
    b0_mem = []
    ep_known_idx = list(range(len(known_list)))
    for key in sorted(tracks):
        rows = tracks[key]
        if mode == "b0":
            out = b0_episode_track(rows, feats_by_key, raw_protos, tau_known, tau_novel, b0_mem)
            results[key] = {"outcome": out, "n_rows": len(rows)}
            if out[0] in ("new", "existing"):
                commits.append((key, out[0], out[1], out[2], 1.0))
            continue
        h, m = model.belief_init(1, model.parameters().__next__().device)
        committed = None
        first_commit = None
        with torch.no_grad():
            for t, r in enumerate(rows):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]), int(r["image_id"])))
                if f is None:
                    continue
                z = model.encode(torch.from_numpy(f).unsqueeze(0).to(h.device))
                rp = float(r_phys_map[id(r)]) if mode == "b3" else 1.0
                rt = torch.tensor([[rp]], device=h.device)
                if mode in ("b2", "b3"):
                    h, m, _ = model.belief_step(z, rt, h, m, t)
                else:
                    h = z
                age = torch.tensor([[float(t + 1)]], device=h.device)
                logits, lsm = model.decision(h, ep_known_idx, memory, rt, age)
                if mode != "b3":
                    lsm = lsm.clone()
                    lsm[0, -1] = -float("inf")
                action = int(lsm[0].argmax())
                nk = len(known_list)
                if action < nk:
                    committed = ("known", known_list[action], t, rp)
                    if first_commit is None:
                        commits.append((key, "known", known_list[action], t, rp))
                        first_commit = committed
                elif action < nk + memory.size():
                    k = action - nk
                    committed = ("existing", k, t, rp)
                    memory.update(k, h, rp)
                    if first_commit is None:
                        commits.append((key, "existing", k, t, rp))
                        first_commit = committed
                elif action == nk + memory.size():
                    committed = ("new", memory.size(), t, rp)
                    memory.create(h, rp, {"track": key, "frame": r["frame_id"]})
                    if first_commit is None:
                        commits.append((key, "new", memory.size() - 1, t, rp))
                        first_commit = committed
                else:
                    committed = ("defer", None, t, rp) if committed is None else committed
        results[key] = {"outcome": first_commit, "n_rows": len(rows)}
    return results, memory, commits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/iclr27_phase4s/full_model/checkpoint.pth")
    ap.add_argument("--proposals", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--mode", default="b3", choices=["b0", "b1", "b2", "b3"])
    ap.add_argument("--out", default="outputs/iclr27_phase4s/dev_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau_known", type=float, default=0.55)
    ap.add_argument("--tau_novel", type=float, default=0.45)
    args = ap.parse_args()

    by_train, by_dev, features = load_episodic_universe()
    known_mat = build_known_matrix(features, {**by_train, **by_dev})
    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}
    dev = "cpu" if args.mode == "b0" else args.device
    model = SemanticCore(768, 256, known_prototypes=known_mat).to(dev)
    if args.mode != "b0":
        ck = torch.load(args.checkpoint, map_location=args.device)
        ck["model"].pop("known_raw", None)
        model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rows = load_proposals(Path(args.proposals))
    arr = np.load(args.feats)["feats"]
    assert len(arr) == len(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_phys_map = compute_r_phys(rows, w)

    raw_protos = None
    if args.mode == "b0":
        from src.iclr27_phase4s.episodes import category_prototypes
        all_cats = {c: s for c, s in by_train.items()}
        for c, s in by_dev.items():
            all_cats.setdefault(c, []).extend(s)
        raw_protos = category_prototypes(features, all_cats)

    results, memory, commits = run_frontend(
        model, tracks, feats_by_key, r_phys_map, known_list, known_cat_index,
        args.mode, raw_protos, args.tau_known, args.tau_novel,
    )

    # align predicted physical tracks to GT sample ids (IoU>=0.3 greedy)
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        out = results[key]["outcome"]
        if out is None or out[0] in ("defer",):
            preds.append({"sample_id": sid, "prediction_type": "unresolved",
                          "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            preds.append({"sample_id": sid, "prediction_type": "novel",
                          "virtual_category_id": out[1], "stream_order": order})
    gt_rows = [labels[sid] for sid in sorted(labels)]
    ev = TrackOCDEvaluator(gt_rows)
    metrics = ev.evaluate(preds, metadata={"memory_size": memory.size()})
    assignment = metrics.get("hungarian_assignment", {})
    slot_gt = {int(k): int(v) for k, v in assignment.items()}
    gt_of = labels
    slot_tracks = defaultdict(list)
    for key, kind, sid_or_cat, t, rp in commits:
        if kind == "new":
            slot_tracks[sid_or_cat].append(key)
    tax = Counter()
    for key, kind, slot_id, t, rp in commits:
        sid = mapping.get(key)
        role = gt_of[sid]["protocol_role"] if sid else "fp"
        cat = gt_of[sid]["ground_truth_category_id"] if sid else -1
        if role in ("supported_known", "zero_shot_known"):
            if kind == "known":
                if slot_id != cat:
                    tax["WRONG_KNOWN"] += 1
            else:
                tax["KNOWN_TO_NOVEL"] += 1
        elif role == "novel":
            if kind == "known":
                tax["NOVEL_TO_KNOWN"] += 1
            elif kind == "existing":
                tax["REUSE_COMMIT"] += 1
                if slot_id in slot_gt and slot_gt[slot_id] != cat:
                    tax["WRONG_REUSE"] += 1
            elif kind == "new":
                tax["BIRTH_COMMIT"] += 1
                if cat in slot_gt.values():
                    tax["OVERBIRTH"] += 1
        else:
            if kind == "new":
                tax["FP_BORN_MEMORY"] += 1
            elif kind == "existing":
                tax["WRONG_MEMORY_UPDATE"] += 1
            elif kind == "known":
                tax["FP_TO_KNOWN"] += 1
    for key, sid in mapping.items():
        role = gt_of[sid]["protocol_role"]
        out = results[key]["outcome"]
        if role == "novel" and out is not None:
            if out[0] == "new" and out[2] <= 1:
                tax["PREMATURE_BIRTH"] += 1
            if out[0] == "new" and out[2] >= 4:
                tax["DELAYED_COMMIT"] += 1
        if role == "novel" and (out is None or out[0] == "defer"):
            tax["UNDERBIRTH"] += 1
    slot_purity = {}
    for k, tks in slot_tracks.items():
        cats = [gt_of[mapping[t]]["ground_truth_category_id"]
                for t in tks if t in mapping]
        if cats:
            most = Counter(cats).most_common(1)[0]
            slot_purity[k] = {"dominant_gt_cat": most[0],
                              "purity": round(most[1] / len(cats), 4),
                              "support": len(cats)}
    fp_slots = [k for k, tks in slot_tracks.items()
                if not any(t in mapping for t in tks)]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "n_pred_tracks": len(results),
        "n_aligned_tracks": len(mapping),
        "r_phys_coefs": [float(x) for x in w],
        "memory_slots": memory.size() if args.mode != "b0"
        else len({c[2] for c in commits if c[1] == "new"}),
        "memory_support": list(memory.support),
        "error_taxonomy": dict(tax),
        "slot_purity": slot_purity,
        "fp_born_slots": len(fp_slots),
        "novel_slots_with_gt": len(slot_gt),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items()
                    if k != "hungarian_assignment"},
    }
    (out / f"dev_{args.mode}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
