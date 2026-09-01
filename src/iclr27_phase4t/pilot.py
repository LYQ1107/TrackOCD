"""Phase 4T episodic pilot: T1/T2/T3/T4 metrics on the chosen domain."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.episodes import (
    EpisodeConfig as SynConfig,
    category_prototypes,
    episode_to_batch,
    load_episodic_universe,
    make_episode,
)
from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4s.train import build_known_matrix
from src.iclr27_phase4t.episodes import (
    RealEpisodeConfig,
    RealStreamStore,
    make_real_episode,
    real_episode_batch,
)
from src.iclr27_phase4t.model import HierarchicalCore
from src.iclr27_phase4t.runtime import teacher_targets

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def eval_episode(model, batch, cfg, known_cat_index, known_list, memory, use_hierarchy):
    ep_known_cats = list(batch["pseudo_known"])
    ep_known_idx = [known_cat_index[c] for c in ep_known_cats]
    n_known = len(ep_known_cats)
    l1s, l2s, _, _ = teacher_targets(batch, cfg)
    n_per_occ = batch["mask"].sum(-1).cpu().numpy().tolist()
    outcomes = []
    by_k = []
    # episode-level slot provenance: slot index -> pseudo-novel category
    slot_cat = []
    for i in range(batch["feats"].shape[1]):
        n = int(n_per_occ[i])
        role, first = batch["role_first"][i]
        cat = int(batch["cats"][i])
        if n == 0:
            outcomes.append(None)
            continue
        h, m = model.belief_init(1, batch["feats"].device)
        first_commit = None
        for t in range(n):
            z = model.encode(batch["feats"][i, t : t + 1])
            if use_hierarchy and getattr(model, "use_qphys", False) and batch.get("qphys") is not None:
                r = torch.tensor([batch["qphys"][i, t].tolist()], device=h.device)
            else:
                r = batch["r_phys"][i, t : t + 1].unsqueeze(-1)
            h, m, g = model.belief_step(z, r, h, m, t)
            age = torch.tensor([[float(t + 1)]], device=h.device)
            if use_hierarchy:
                out = model.decision(h, ep_known_idx, memory, r, age)
                a1 = int(out["l1_lsm"][0].argmax())
                l1_teacher = l1s[i][t]
                k_now = memory.size()
                bucket = 0 if k_now == 0 else (1 if k_now <= 2 else (2 if k_now <= 5 else (3 if k_now <= 10 else 4)))
                by_k.append((role, a1, l1_teacher, bucket, t))
                if first_commit is None and a1 != 2:
                    if a1 == 0:
                        a2 = int(out["known"][0].argmax())
                        first_commit = ("known", ep_known_cats[a2], t, k_now)
                    else:
                        a2 = int(out["l2_lsm"][0].argmax())
                        if a2 == 0:  # EXISTING
                            slot = int(out["l2"]["novel"][0].argmax())
                            payload = slot_cat[slot] if slot < len(slot_cat) else -1
                            first_commit = ("existing", payload, t, k_now)
                            memory.update(slot, h, float(r[0, 0]))
                        elif a2 == 1:  # NEW
                            first_commit = ("new", cat, t, k_now)
                            memory.create(h, float(r[0, 0]), {"cat": cat})
                            slot_cat.append(cat)
                        else:
                            pass  # defer at level 2
            else:
                # flat Phase4S-style decisions (r_phys scalar)
                lsm = None
                logits = model.decision(h, ep_known_idx, memory, r, age)
                _, lsm = logits[0], logits[1]
                lsm = lsm.clone()
                a = int(lsm[0].argmax())
                from src.iclr27_phase4s.runtime import model_action_kind
                kind = model_action_kind(a, n_known, memory.size())
                bucket = 0 if memory.size() == 0 else (1 if memory.size() <= 2 else (2 if memory.size() <= 5 else (3 if memory.size() <= 10 else 4)))
                l1_teacher = 2 if role == "fp" else (0 if role == "known" else 1)
                pred_l1 = 2 if kind[0] == "defer" else (0 if kind[0] == "known" else 1)
                by_k.append((role, pred_l1, l1_teacher, bucket, t))
                if first_commit is None and kind[0] != "defer":
                    if kind[0] == "known":
                        first_commit = ("known", ep_known_cats[kind[1]], t, memory.size())
                    elif kind[0] == "existing":
                        first_commit = ("existing", slot_cat[kind[1]] if kind[1] < len(slot_cat) else -1, t, memory.size())
                        memory.update(kind[1], h, float(r[0, 0]))
                    else:
                        first_commit = ("new", cat, t, memory.size())
                        memory.create(h, float(r[0, 0]), {"cat": cat})
                        slot_cat.append(cat)
        outcomes.append(first_commit)
    return outcomes, by_k


def score_outcomes(outcomes, batch, cfg):
    l1s, l2s, _, _ = teacher_targets(batch, cfg)
    stats = defaultdict(int)
    for i, out in enumerate(outcomes):
        n = int(batch["mask"][i].sum())
        if n == 0:
            continue
        role, first = batch["role_first"][i]
        cat = int(batch["cats"][i])
        if role == "known":
            stats["known_total"] += 1
            if out is not None and out[0] == "known" and out[1] == cat:
                stats["known_correct"] += 1
            elif out is not None and out[0] in ("existing", "new"):
                stats["known_to_novel"] += 1
        elif role == "novel":
            if first:
                stats["novel_first_total"] += 1
                if out is not None and out[0] == "new":
                    stats["novel_first_correct"] += 1
                elif out is not None and out[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif out is not None and out[0] == "known":
                    stats["novel_to_known"] += 1
                else:
                    stats["underbirth"] += 1
            else:
                stats["novel_later_total"] += 1
                if out is not None and out[0] == "existing" and out[1] == cat:
                    stats["reuse_correct"] += 1
                elif out is not None and out[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif out is not None and out[0] == "new":
                    stats["overbirth"] += 1
                elif out is not None and out[0] == "known":
                    stats["novel_to_known"] += 1
                else:
                    stats["underbirth"] += 1
        else:
            stats["fp_total"] += 1
            if out is not None:
                stats["fp_commit"] += 1
                if out[0] == "new":
                    stats["fp_born"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--use-hierarchy", action="store_true")
    ap.add_argument("--use-defer", action="store_true")
    ap.add_argument("--use-qphys", action="store_true")
    ap.add_argument("--stream-csv", default="outputs/iclr27_phase4t/train_stream/proposals.csv")
    ap.add_argument("--stream-feats", default="outputs/iclr27_phase4t/train_stream/feats.npz")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    by_train, by_dev, syn_features = load_episodic_universe()
    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}
    known_mat = build_known_matrix(syn_features, {**by_train, **by_dev})
    if args.use_hierarchy:
        model = HierarchicalCore(768, 256, known_prototypes=known_mat,
                                 use_defer=args.use_defer, use_qphys=args.use_qphys).to(args.device)
    else:
        model = SemanticCore(768, 256, known_prototypes=known_mat).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device)
    ck["model"].pop("known_raw", None)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    if args.data == "real":
        import csv as _csv
        rows = list(_csv.DictReader(open(ROOT / args.stream_csv)))
        for r in rows:
            r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
            r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
            r["q_phys"] = json.loads(r["q_phys"])
            r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
            r["gt_role"] = r["gt_role"]
            r["gt_category_id"] = int(r["gt_category_id"])
            r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
            r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
            r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
        store = RealStreamStore(rows, np.load(ROOT / args.stream_feats)["feats"])
        cfg = RealEpisodeConfig()
    else:
        cfg = SynConfig()

    agg = defaultdict(int)
    by_k = defaultdict(lambda: defaultdict(int))
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    slot_counts = []
    for e in range(args.n_episodes):
        memory = NovelMemory(args.device)
        if args.data == "real":
            ep = make_real_episode(store, cfg, rng)
            batch = real_episode_batch(store, ep, cfg)
        else:
            ep = make_episode(by_dev, syn_features, cfg, rng, np_rng)
            batch = episode_to_batch(ep, cfg, {}, 8)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(args.device)
        with torch.no_grad():
            outcomes, bk = eval_episode(model, batch, cfg, known_cat_index,
                                        known_list, memory, args.use_hierarchy)
        s = score_outcomes(outcomes, batch, cfg)
        for k, v in s.items():
            agg[k] += v
        slot_counts.append(memory.size())
        for role, pred, teach, bucket, t in bk:
            if role == "fp":
                by_k[bucket]["fp_defer_rate"] += pred == 2
                by_k[bucket]["fp_total"] += 1
            else:
                by_k[bucket]["valid_total"] += 1
                by_k[bucket]["valid_l1_correct"] += int(pred == teach)
                if teach == 0:
                    by_k[bucket]["known_to_novel"] += int(pred == 1)
                if teach == 1:
                    by_k[bucket]["novel_to_known"] += int(pred == 0)
                by_k[bucket]["defer_rate"] += int(pred == 2)
    report = {
        "known_acc": round(agg["known_correct"] / max(agg["known_total"], 1), 4),
        "novel_first_new": round(agg["novel_first_correct"] / max(agg["novel_first_total"], 1), 4),
        "novel_later_reuse": round(agg["reuse_correct"] / max(agg["novel_later_total"], 1), 4),
        "wrong_reuse": agg["wrong_reuse"], "overbirth": agg["overbirth"],
        "underbirth": agg["underbirth"], "novel_to_known": agg["novel_to_known"],
        "known_to_novel": agg["known_to_novel"],
        "fp_commit_rate": round(agg["fp_commit"] / max(agg["fp_total"], 1), 4),
        "fp_born": agg["fp_born"],
        "mean_slots": round(float(np.mean(slot_counts)), 3),
    }
    for k in sorted(by_k):
        v = by_k[k]
        v = {kk: (round(vv / max(v["valid_total"], 1), 4) if kk.endswith("rate") and kk != "fp_defer_rate" else vv) for kk, vv in v.items()}
        report[f"k_bucket_{k}"] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
