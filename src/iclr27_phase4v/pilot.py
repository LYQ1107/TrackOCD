"""Episodic pilot for DS-TrackOCD (frozen branches + independent router)."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4t.episodes import (
    RealEpisodeConfig,
    RealStreamStore,
    make_real_episode,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    load_known_branch,
    load_novel_branch,
)
from src.iclr27_phase4v.train_router import LogisticRouter, MLPRouter


def load_store():
    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    return RealStreamStore(rows, feats)


def eval_episode(episode, store, cfg, known_list, router, ktsr, kcls,
                 ntsr, l2, device):
    memory = NovelMemory(device)
    slot_cat = []
    outcomes = []
    for occ in episode["occurrences"]:
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
        first_commit = None
        for t in range(n):
            f = torch.from_numpy(z[t:t + 1]).to(device)
            qt = torch.from_numpy(q[t:t + 1]).to(device)
            r_scalar = float(np.clip(q[t, 0], 0.05, 0.95))
            ev, s_k, s_n, nl, l2_new = ds.step(f, qt, r_scalar, t + 1, memory)
            r_logits = router(torch.from_numpy(ev).unsqueeze(0).to(device))[0]
            if first_commit is not None:
                break
            if r_logits[1] > r_logits[0]:  # KNOWN
                kl = kcls(s_k)[0]
                pred = known_list[int(kl.argmax())]
                first_commit = ("known", pred, t, memory.size())
            else:  # NOVEL
                cat = occ["category"]
                if nl.shape[1] >= 1 and nl.max() >= l2_new[0, 0]:
                    slot = int(nl.argmax())
                    payload = slot_cat[slot] if slot < len(slot_cat) else -1
                    first_commit = ("existing", payload, t, memory.size())
                    memory.update(slot, s_n, r_scalar)
                else:
                    first_commit = ("new", cat, t, memory.size())
                    memory.create(s_n, r_scalar, {"cat": cat})
                    slot_cat.append(cat)
        outcomes.append(first_commit)
    return outcomes, memory


def score(outcomes, episode):
    stats = defaultdict(int)
    seen = set()
    for occ, out in zip(episode["occurrences"], outcomes):
        role = occ["role"]
        cat = occ["category"]
        if role == "known":
            stats["known_total"] += 1
            if out is not None and out[0] == "known" and out[1] == cat:
                stats["known_correct"] += 1
            elif out is not None and out[0] in ("existing", "new"):
                stats["known_to_novel"] += 1
        elif role == "novel":
            first = occ.get("first", False)
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
    ap.add_argument("--router", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    store = load_store()
    cfg = RealEpisodeConfig()
    known_list = sorted(known_ids())
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    rc = torch.load(ROOT / args.router, map_location=args.device)
    dim = int(rc.get("dim", 15))
    router = (LogisticRouter(dim) if rc["arch"] == "logistic"
              else MLPRouter(dim)).to(args.device)
    router.load_state_dict(rc["model"])
    router.eval()

    rng = random.Random(args.seed)
    agg = defaultdict(int)
    slots = []
    for e in range(args.n_episodes):
        ep = make_real_episode(store, cfg, rng)
        outcomes, memory = eval_episode(ep, store, cfg, known_list, router,
                                        ktsr, kcls, ntsr, l2, args.device)
        s = score(outcomes, ep)
        for k, v in s.items():
            agg[k] += v
        slots.append(memory.size())
    report = {
        "known_acc": round(agg["known_correct"] / max(agg["known_total"], 1), 4),
        "novel_first_new": round(agg["novel_first_correct"] / max(agg["novel_first_total"], 1), 4),
        "novel_later_reuse": round(agg["reuse_correct"] / max(agg["novel_later_total"], 1), 4),
        "wrong_reuse": agg["wrong_reuse"], "overbirth": agg["overbirth"],
        "underbirth": agg["underbirth"], "novel_to_known": agg["novel_to_known"],
        "known_to_novel": agg["known_to_novel"],
        "fp_commit_rate": round(agg["fp_commit"] / max(agg["fp_total"], 1), 4),
        "fp_born": agg["fp_born"],
        "mean_slots": round(float(np.mean(slots)), 3),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
