"""Episodic pilot gate for B0/B1/B2/B3 on meta-dev episodes."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.baselines import b0_episode, neural_episode, score_outcomes
from src.iclr27_phase4s.episodes import (
    EpisodeConfig,
    category_prototypes,
    episode_to_batch,
    load_episodic_universe,
    make_episode,
)
from src.iclr27_phase4s.model import SemanticCore
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4s.train import build_known_matrix

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def retrieval_margins(model, by_cat, features, cfg, n_eps=40, seed=0):
    """Same-category cross-track vs cross-category cosine, frozen vs learned."""
    frozen_same, frozen_diff = [], []
    learned_same, learned_diff = [], []
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    dev = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(n_eps):
            ep = make_episode(by_cat, features, cfg, rng, np_rng)
            items = []  # (cat, frozen_mean, learned_hT)
            for occ in ep["occurrences"]:
                if occ["role"] != "novel":
                    continue
                z = occ["frames"].mean(axis=0)
                z = z / (np.linalg.norm(z) + 1e-12)
                h = torch.zeros(1, model.hidden, device=dev)
                m = torch.zeros(1, model.hidden, device=dev)
                for t in range(len(occ["frames"])):
                    zt = model.encode(torch.from_numpy(occ["frames"][t]).unsqueeze(0).to(dev))
                    r = torch.tensor([[float(occ["r_phys"][t])]], device=dev)
                    h, m, _ = model.belief_step(zt, r, h, m, t)
                h = torch.nn.functional.normalize(h, dim=-1)[0].cpu().numpy()
                items.append((occ["category"], z, h))
            for a in range(len(items)):
                for b in range(len(items)):
                    if a >= b:
                        continue
                    same = items[a][0] == items[b][0]
                    (frozen_same if same else frozen_diff).append(float(items[a][1] @ items[b][1]))
                    (learned_same if same else learned_diff).append(float(items[a][2] @ items[b][2]))
    return {
        "frozen_same": round(float(np.mean(frozen_same)), 4),
        "frozen_diff": round(float(np.mean(frozen_diff)), 4),
        "frozen_margin": round(float(np.mean(frozen_same) - np.mean(frozen_diff)), 4),
        "learned_same": round(float(np.mean(learned_same)), 4),
        "learned_diff": round(float(np.mean(learned_diff)), 4),
        "learned_margin": round(float(np.mean(learned_same) - np.mean(learned_diff)), 4),
    }


def run_methods(model, episodes, known_cat_index, known_list, raw_protos, tau_known, tau_novel):
    agg = {m: {} for m in ("b0", "b1", "b2", "b3")}
    steps = {m: {a: 0 for a in range(4)} for m in ("b0", "b1", "b2", "b3")}
    slots = {m: [] for m in ("b0", "b1", "b2", "b3")}
    for ep in episodes:
        batch = episode_to_batch(ep, EpisodeConfig(), {}, 8)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(next(model.parameters()).device)
        with torch.no_grad():
            b0, n0 = b0_episode(batch, EpisodeConfig(), raw_protos, tau_known, tau_novel)
            results = {"b0": b0}
            slots["b0"].append(n0)
            for mode in ("b1", "b2", "b3"):
                out, st, records, slot_cat = neural_episode(
                    model, batch, EpisodeConfig(), known_cat_index, known_list, mode=mode)
                results[mode] = out
                for a in range(4):
                    steps[mode][a] += st[a]
                slots[mode].append(len(slot_cat))
        for mode in results:
            stats, _ = score_outcomes(results[mode], batch, EpisodeConfig())
            for k, v in stats.items():
                agg[mode][k] = agg[mode].get(k, 0) + v
    return agg, steps, slots


def summarize(agg, steps, slots, n_eps):
    report = {}
    for mode in agg:
        a = agg[mode]
        report[mode] = {
            "known_acc": round(a["known_correct"] / max(a["known_total"], 1), 4),
            "novel_first_new_acc": round(a["novel_first_correct"] / max(a["novel_first_total"], 1), 4),
            "novel_later_reuse_acc": round(a["novel_later_correct"] / max(a["novel_later_total"], 1), 4),
            "existing_vs_new_acc": round(a["existing_vs_new_correct"] / max(a["existing_vs_new_total"], 1), 4),
            "wrong_reuse": a["wrong_reuse"],
            "overbirth": a["overbirth"],
            "novel_to_known": a["novel_to_known"],
            "known_to_novel": a["known_to_novel"],
            "fp_commit_rate": round(a["fp_commit"] / max(a["fp_total"], 1), 4),
            "fp_born_slots": a["fp_born_slots"],
            "unresolved_novel": a["unresolved"],
            "mean_slots_per_episode": round(float(np.mean(slots[mode])), 3),
        }
    report["steps_by_age"] = steps
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/iclr27_phase4s/full_model/checkpoint.pth")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--out", default="outputs/iclr27_phase4s/episodic_pilot")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    by_train, by_dev, features = load_episodic_universe()
    cfg = EpisodeConfig()
    known_list = sorted(known_ids())
    known_cat_index = {c: i for i, c in enumerate(known_list)}
    all_cats = {c: s for c, s in by_train.items()}
    for c, s in by_dev.items():
        all_cats.setdefault(c, []).extend(s)
    raw_protos = category_prototypes(features, all_cats)

    ck = torch.load(args.checkpoint, map_location=args.device)
    known_mat = build_known_matrix(features, {**by_train, **by_dev})
    model = SemanticCore(768, 256, known_prototypes=known_mat).to(args.device)
    ck["model"].pop("known_raw", None)  # rebuild matrix from the legal 48-class universe
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    episodes = [make_episode(by_dev, features, cfg, rng, np_rng) for _ in range(args.n_episodes)]

    # B0 threshold grid on the dev episodes (legal dev selection, frozen control)
    best = None
    for tk in (0.35, 0.45, 0.55):
        for tn in (0.30, 0.40, 0.50):
            totals = {"known_correct": 0, "novel_first_correct": 0, "novel_later_correct": 0,
                      "known_total": 0, "novel_first_total": 0, "novel_later_total": 0}
            for ep in episodes[:80]:
                batch = episode_to_batch(ep, EpisodeConfig(), {}, 8)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(next(model.parameters()).device)
                b0, _ = b0_episode(batch, EpisodeConfig(), raw_protos, tk, tn)
                stats, _ = score_outcomes(b0, batch, EpisodeConfig())
                for k in totals:
                    totals[k] += stats[k]
            acc = (totals["known_correct"] + totals["novel_first_correct"] +
                   totals["novel_later_correct"]) / max(
                totals["known_total"] + totals["novel_first_total"] +
                totals["novel_later_total"], 1)
            if best is None or acc > best[0]:
                best = (acc, tk, tn)
    _, tau_known, tau_novel = best

    agg, steps, slots = run_methods(model, episodes, known_cat_index, known_list,
                                    raw_protos, tau_known, tau_novel)
    report = summarize(agg, steps, slots, args.n_episodes)
    report["b0_thresholds"] = {"tau_known": tau_known, "tau_novel": tau_novel}
    report["retrieval"] = retrieval_margins(model, by_dev, features, cfg, seed=args.seed)
    report["n_episodes"] = args.n_episodes
    report["seed"] = args.seed
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
