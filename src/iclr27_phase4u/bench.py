"""Frozen-feature representation baselines and three-layer upper bound.

R0: single frame (first frame of each physical instance)
R1: causal mean pooling over the prefix
R2: quality-weighted causal mean (score as weight; uniform when q is None)
complete: full-track mean (upper bound)

Metrics: same-physical / cross-physical-same-semantic / different-semantic
cosine and margin, cross-track Recall@1/@5 (micro + macro, plus cross-video),
and category-prototype accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4u.data import class_sets, load_source

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def prefix_reprs(inst, p, mode: str) -> np.ndarray:
    f = inst["feats"][:p]
    if mode == "r0":
        return f[0:1]
    if mode == "r1":
        return f.mean(axis=0, keepdims=True)
    if mode == "r2":
        q = inst["q"]
        w = q[:p, 0].clip(0.05, 1.0) if q is not None else np.ones(p, dtype=np.float32)
        w = w / (w.sum() + 1e-12)
        return (f * w[:, None]).sum(axis=0, keepdims=True)
    raise ValueError(mode)


def complete_repr(inst) -> np.ndarray:
    return inst["feats"].mean(axis=0, keepdims=True)


def metrics_for_embeddings(
    embs: dict[str, np.ndarray],
    complete: dict[str, np.ndarray],
    src: dict,
    seed: int,
    prefix_label: str,
) -> dict:
    inst_by_id = {x["id"]: x for x in src["instances"]}
    ids = sorted(embs)
    cats = {i: inst_by_id[i]["cat"] for i in ids}
    videos = {i: inst_by_id[i]["video"] for i in ids}
    by_cat = defaultdict(list)
    for i in ids:
        by_cat[cats[i]].append(i)

    # leave-one-out category prototype and nearest-different-category prototype
    protos = {}
    for c, cids in by_cat.items():
        arr = np.stack([complete[i] for i in cids])
        protos[c] = arr.mean(axis=0)
        protos[c] /= np.linalg.norm(protos[c]) + 1e-12

    cos_sp, cos_cross, cos_diff, cos_diff_avg = [], [], [], []
    per_cat_cross = defaultdict(list)
    per_cat_diff = defaultdict(list)
    for i in ids:
        v = embs[i]
        c = cats[i]
        cids = by_cat[c]
        if len(cids) >= 2:
            others = [j for j in cids if j != i]
            loo = np.stack([complete[j] for j in others]).mean(axis=0)
            loo /= np.linalg.norm(loo) + 1e-12
            cc = float(v @ loo)
            cos_cross.append(cc)
            per_cat_cross[c].append(cc)
        if inst_by_id[i]["feats"].shape[0] >= 2:
            cos_sp.append(float(v @ complete[i]))
        other_cats = [cc for cc in protos if cc != c]
        scores = np.array([float(v @ protos[cc]) for cc in other_cats])
        cos_diff.append(float(scores.max()))
        cos_diff_avg.append(float(scores.mean()))
        per_cat_diff[c].append(float(scores.mean()))

    macro_cross = np.mean([float(np.mean(per_cat_cross[c])) for c in per_cat_cross])
    macro_diff = np.mean([float(np.mean(per_cat_diff[c])) for c in per_cat_diff])

    out = {
        "prefix": prefix_label,
        "n_instances": len(ids),
        "n_categories": len(by_cat),
        "cos_same_physical": round(float(np.mean(cos_sp)), 4) if cos_sp else None,
        "cos_cross_phys_same_sem": round(float(np.mean(cos_cross)), 4) if cos_cross else None,
        "cos_diff_sem_hard": round(float(np.mean(cos_diff)), 4),
        "cos_diff_sem_avg": round(float(np.mean(cos_diff_avg)), 4),
        "cross_margin": round(float(np.mean(cos_cross)) - float(np.mean(cos_diff)), 4)
        if cos_cross else None,
        "cross_margin_avg": round(float(np.mean(cos_cross)) - float(np.mean(cos_diff_avg)), 4)
        if cos_cross else None,
        "cross_margin_macro_avg": round(float(macro_cross - macro_diff), 4),
        "same_phys_margin": round(float(np.mean(cos_sp)) - float(np.mean(cos_diff)), 4)
        if cos_sp else None,
        "same_phys_margin_avg": round(float(np.mean(cos_sp)) - float(np.mean(cos_diff_avg)), 4)
        if cos_sp else None,
    }

    # sampled retrieval set (<=20 instances per category, categories with >=2)
    rng = random.Random(seed)
    subset = []
    for c in sorted(by_cat):
        if len(by_cat[c]) < 2:
            continue
        picked = by_cat[c][:]
        rng.shuffle(picked)
        subset.extend(picked[:20])
    if len(subset) >= 4:
        mat = np.stack([embs[i] for i in subset])
        sim = mat @ mat.T
        pos = np.zeros((len(subset), len(subset)), dtype=bool)
        for a, i in enumerate(subset):
            for b, j in enumerate(subset):
                if a != b and cats[i] == cats[j]:
                    pos[a, b] = True
        cv_pos = np.zeros_like(pos)
        for a, i in enumerate(subset):
            for b, j in enumerate(subset):
                if a != b and cats[i] == cats[j] and videos[i] != videos[j]:
                    cv_pos[a, b] = True
        r1 = r5 = 0
        cv_r1 = cv_r5 = cv_q = 0
        per_cat = defaultdict(lambda: [0, 0])
        for a, i in enumerate(subset):
            order = np.argsort(-sim[a])[1:]  # exclude self (rank 0)
            hit1 = hit5 = False
            for rank, b in enumerate(order):
                if not pos[a, b]:
                    continue
                if rank < 1:
                    hit1 = True
                if rank < 5:
                    hit5 = True
                break
            r1 += int(hit1)
            r5 += int(hit5)
            per_cat[cats[i]][0] += int(hit1)
            per_cat[cats[i]][1] += 1
            if cv_pos[a].any():
                cv_q += 1
                hit1 = hit5 = False
                for rank, b in enumerate(order):
                    if not cv_pos[a, b]:
                        continue
                    if rank < 1:
                        hit1 = True
                    if rank < 5:
                        hit5 = True
                    break
                cv_r1 += int(hit1)
                cv_r5 += int(hit5)
        macro_r1 = np.mean([v[0] / v[1] for v in per_cat.values()]) if per_cat else 0.0
        r5_by_cat = defaultdict(lambda: [0, 0])
        for a, i in enumerate(subset):
            order = np.argsort(-sim[a])[1:]  # exclude self (rank 0)
            hit5 = False
            for rank, b in enumerate(order):
                if not pos[a, b]:
                    continue
                if rank < 5:
                    hit5 = True
                break
            r5_by_cat[cats[i]][0] += int(hit5)
            r5_by_cat[cats[i]][1] += 1
        macro_r5 = np.mean([v[0] / v[1] for v in r5_by_cat.values()])
        out.update({
            "retrieval_micro_r1": round(r1 / len(subset), 4),
            "retrieval_micro_r5": round(r5 / len(subset), 4),
            "retrieval_macro_r1": round(float(macro_r1), 4),
            "retrieval_macro_r5": round(float(macro_r5), 4),
            "retrieval_queries": len(subset),
            "cross_video_micro_r1": round(cv_r1 / cv_q, 4) if cv_q else None,
            "cross_video_micro_r5": round(cv_r5 / cv_q, 4) if cv_q else None,
            "cross_video_queries": cv_q,
        })

    # category prototype accuracy (complete embeddings)
    proto_acc = 0
    for i in ids:
        scores = {c: float(complete[i] @ p) for c, p in protos.items()}
        pred = max(scores, key=scores.get)
        proto_acc += int(pred == cats[i])
    out["proto_acc_complete"] = round(proto_acc / max(len(ids), 1), 4)
    return out


def run_source(src: dict, prefix_lengths, seed: int, out_json: dict):
    instances = [x for x in src["instances"]]
    complete = {x["id"]: complete_repr(x)[0] for x in instances}
    complete_norm = {
        i: v / (np.linalg.norm(v) + 1e-12) for i, v in complete.items()}
    results = []
    for mode in ("r0", "r1", "r2"):
        for p in (prefix_lengths if mode != "r0" else (1,)):
            embs = {}
            for x in instances:
                if p > x["feats"].shape[0]:
                    continue
                v = prefix_reprs(x, p, mode)[0]
                v = v / (np.linalg.norm(v) + 1e-12)
                embs[x["id"]] = v.astype(np.float32)
            results.append(metrics_for_embeddings(
                embs, complete_norm, src, seed, f"{mode}_p{p}"))
    # complete-track upper bound
    results.append(metrics_for_embeddings(
        complete_norm, complete_norm, src, seed, "complete"))
    out_json[src["name"]] = results
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="real,episodic,dev")
    ap.add_argument("--prefix-lengths", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--class-set", default="all",
                    choices=["all", "meta_train", "meta_dev"])
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    meta_tr, meta_de = class_sets()
    allowed = None
    if args.class_set == "meta_train":
        allowed = meta_tr
    elif args.class_set == "meta_dev":
        allowed = meta_de
    prefix_lengths = [int(x) for x in args.prefix_lengths.split(",")]
    out_json = {"class_set": args.class_set, "seed": args.seed}
    for name in args.sources.split(","):
        src = load_source(name)
        if allowed is not None:
            src["instances"] = [x for x in src["instances"] if x["cat"] in allowed]
            src["by_cat"] = defaultdict(list)
            for x in src["instances"]:
                src["by_cat"][x["cat"]].append(x["id"])
        run_source(src, prefix_lengths, args.seed, out_json)
        print(name, "done", len(src["instances"]), "instances", flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_json, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
