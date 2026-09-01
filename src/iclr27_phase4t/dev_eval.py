"""Phase 4T hierarchical dev evaluation on the frozen Q1/Q2 dev streams.

Online hierarchical decisions per physical track; global dynamic memory;
TrackOCD-v1.0 corrected evaluator for scoring; routing-vs-K analysis.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.runtime import model_action_kind
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4s.train import build_known_matrix
from src.iclr27_phase4s.episodes import load_episodic_universe
from src.iclr27_phase4t.model import HierarchicalCore
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def qphys_from_rows(rows: list[dict]) -> dict[int, list[float]]:
    """Causal q_phys per proposal from the dev CSV (score, prior_hits, age,
    gap, running score mean, log area)."""
    by_track = defaultdict(list)
    for r in rows:
        by_track[(r["video_id"], r["track_id"])].append(r)
    out = {}
    for key, idxs in by_track.items():
        idxs.sort(key=lambda r: (r["frame_id"], r["track_id"]))
        last_frame, hits, ssum, n = None, 0, 0.0, 0
        for r in idxs:
            gap = 0 if last_frame is None else r["frame_id"] - last_frame - 1
            b = json.loads(r["bbox_xyxy"])
            area = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
            out[id(r)] = [r["score"], float(np.log1p(hits)), min(hits, 16) / 16.0,
                          float(np.log1p(max(gap, 0))),
                          ssum / n if n else r["score"],
                          float(np.log(area) / 12.0)]
            last_frame = r["frame_id"]
            hits += 1
            ssum += r["score"]
            n += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--proposals", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--use-hierarchy", action="store_true")
    ap.add_argument("--use-defer", action="store_true")
    ap.add_argument("--use-qphys", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    by_train, by_dev, features = load_episodic_universe()
    known_mat = build_known_matrix(features, {**by_train, **by_dev})
    known_list = sorted(known_ids())
    known_idx = list(range(len(known_list)))
    if args.use_hierarchy:
        model = HierarchicalCore(768, 256, known_prototypes=known_mat,
                                 use_defer=args.use_defer, use_qphys=args.use_qphys).to(args.device)
    else:
        model = SemanticCore(768, 256, known_prototypes=known_mat).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    ck["model"].pop("known_raw", None)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rows = load_proposals(Path(args.proposals))
    arr = np.load(args.feats)["feats"]
    assert len(arr) == len(rows)
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)

    memory = NovelMemory(args.device)
    results = {}
    commits = []
    by_k = []
    with torch.no_grad():
        for key in sorted(tracks):
            h, m = model.belief_init(1, args.device)
            first_commit = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]), int(r["image_id"])))
                if f is None:
                    continue
                z = model.encode(torch.from_numpy(f).unsqueeze(0).to(args.device))
                row_q = torch.tensor([qmap[id(r)]], device=args.device)
                row_r = torch.tensor([[float(r_scalar[id(r)])]], device=args.device)
                q = row_q if (args.use_hierarchy and args.use_qphys) else row_r
                h, m, g = model.belief_step(z, q, h, m, t)
                age = torch.tensor([[float(t + 1)]], device=args.device)
                if args.use_hierarchy:
                    out = model.decision(h, known_idx, memory, q, age)
                    a1 = int(out["l1_lsm"][0].argmax())
                    pred_l1 = a1
                else:
                    _, lsm = model.decision(h, known_idx, memory, q, age)
                    a = int(lsm[0].argmax())
                    kind = model_action_kind(a, len(known_idx), memory.size())
                    a1 = 2 if kind[0] == "defer" else (0 if kind[0] == "known" else 1)
                    pred_l1 = a1
                kb = 0 if memory.size() == 0 else (1 if memory.size() <= 2 else (2 if memory.size() <= 5 else (3 if memory.size() <= 10 else 4)))
                by_k.append((r["gt_role"], a1, kb, t))
                if a1 == 2:
                    continue
                if args.use_hierarchy:
                    if a1 == 0:
                        a2 = int(out["known"][0].argmax())
                        commit = ("known", known_list[a2], t)
                    else:
                        a2 = int(out["l2_lsm"][0].argmax())
                        if a2 == 0:  # EXISTING
                            slot = int(out["l2"]["novel"][0].argmax())
                            commit = ("existing", slot, t)
                            memory.update(slot, h, float(r_scalar[id(r)]))
                        elif a2 == 1:  # NEW
                            commit = ("new", memory.size(), t)
                            memory.create(h, float(r_scalar[id(r)]), {"track": key})
                        else:
                            commit = ("defer", None, t)
                else:
                    kind = model_action_kind(a, len(known_idx), memory.size())
                    if kind[0] == "known":
                        commit = ("known", known_list[kind[1]], t)
                    elif kind[0] == "existing":
                        commit = ("existing", kind[1], t)
                        memory.update(kind[1], h, 0.5)
                    elif kind[0] == "new":
                        commit = ("new", memory.size(), t)
                        memory.create(h, 0.5, {"track": key})
                    else:
                        commit = ("defer", None, t)
                if commit[0] != "defer" and first_commit is None:
                    first_commit = commit
                    commits.append((key, commit[0], commit[1], t))
            results[key] = {"outcome": first_commit}

    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        out = results[key]["outcome"]
        if out is None or out[0] == "defer":
            preds.append({"sample_id": sid, "prediction_type": "unresolved", "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            preds.append({"sample_id": sid, "prediction_type": "novel",
                          "virtual_category_id": out[1], "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = ev.evaluate(preds, metadata={"memory_size": memory.size()})

    # routing vs K
    by_k_stats = defaultdict(lambda: defaultdict(int))
    for role, a1, kb, t in by_k:
        if role == "fp":
            by_k_stats[kb]["fp_total"] += 1
            by_k_stats[kb]["fp_defer"] += int(a1 == 2)
        else:
            teach = 0 if role == "known" else 1
            by_k_stats[kb]["valid_total"] += 1
            by_k_stats[kb]["l1_correct"] += int(a1 == teach)
            by_k_stats[kb]["defer"] += int(a1 == 2)
            if teach == 0:
                by_k_stats[kb]["known_to_novel"] += int(a1 == 1)
            else:
                by_k_stats[kb]["novel_to_known"] += int(a1 == 0)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "n_aligned": len(mapping),
        "memory_slots": memory.size(),
        "routing_by_k": {str(k): dict(v) for k, v in sorted(by_k_stats.items())},
        "r_phys_coefs": [float(x) for x in w],
    }
    (out_dir / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("metrics", "n_aligned", "memory_slots", "routing_by_k")}, indent=2))


if __name__ == "__main__":
    main()
