"""Counterfactual real-class leave-out episodes for CHP training.

All pseudo-novel tracks come from REAL train-known classes (frozen
meta-train episode pool).  A class can play "episode-known" or
"episode-pseudo-novel" in a given episode; its prototype is excluded from
P_known in that episode.  Hardness is computed only from train-side
features.  Official validation classes are never used here.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def build_protos(z_cache, by_class, classes):
    """Adapted-space prototypes and radii for a set of classes."""
    protos = {}
    radii = {}
    for c in classes:
        zs = np.stack([z_cache[sid] for sid in by_class[c]])
        p = _norm(zs.mean(axis=0))
        protos[c] = p
        cos = zs @ p
        radii[c] = max(float(np.percentile(1.0 - cos, 50)), 0.02)
    return protos, radii


def hardness_of_classes(z_cache, by_class, candidates, known_classes):
    """h(c) = mean over z in c of max cos(z, p_k) over episode-known k.

    Higher = closer to the remaining known prototypes = harder to reject.
    Train-side only; uses real track features in adapted space.
    """
    protos, _ = build_protos(z_cache, by_class, known_classes)
    P = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    out = {}
    for c in candidates:
        zs = np.stack([z_cache[sid] for sid in by_class[c]])
        out[c] = float(np.mean(np.max(P @ zs.T, axis=0)))
    return out


def global_tiers(z_cache, by_class, pool_classes, n=3):
    """Leave-one-out hardness tiers over the frozen episode pool."""
    hardness = {}
    for c in pool_classes:
        others = [o for o in pool_classes if o != c]
        hardness[c] = hardness_of_classes(
            z_cache, by_class, [c], others)[c]
    order = sorted(pool_classes, key=lambda c: hardness[c])
    k = len(order)
    size = max(1, (k + n - 1) // n)
    tiers = {}
    for i in range(n):
        tiers[f"tier{i}"] = order[i * size:(i + 1) * size]
    tiers["easy"] = tiers["tier0"]
    tiers["medium"] = tiers["tier1"]
    tiers["hard"] = tiers["tier2"]
    tiers["hardness"] = hardness
    return tiers


def choose_novel_classes(mode, rng, np_rng, z_cache, by_class, pool_classes,
                         n_known=30, n_novel=8, hardness_map=None,
                         tiers=None):
    """Return (novel_classes, known_classes, hardness_meta)."""
    if mode == "random":  # E0
        pool = list(pool_classes)
        rng.shuffle(pool)
        novel = sorted(pool[:n_novel])
        known = sorted(pool[n_novel:])
        return novel, known, {}
    if mode == "hard":  # E1: sample candidates, keep the hardest n_novel
        pool = list(pool_classes)
        rng.shuffle(pool)
        candidates = pool[:n_novel * 2]
        known_candidates = [c for c in pool if c not in candidates]
        h = hardness_of_classes(z_cache, by_class, candidates,
                                known_candidates)
        ordered = sorted(candidates, key=lambda c: -h[c])
        novel = sorted(ordered[:n_novel])
        known = sorted([c for c in pool if c not in novel])
        return novel, known, {"hardness": {c: h[c] for c in novel}}
    if mode == "mixed":  # E2: easy/medium/hard mix
        if tiers is None:
            raise ValueError("mixed mode requires tiers")
        easy = list(tiers["easy"])
        medium = list(tiers["medium"])
        hard = list(tiers["hard"])
        rng.shuffle(easy)
        rng.shuffle(medium)
        rng.shuffle(hard)
        n_easy = max(1, n_novel // 4)
        n_hard = max(1, n_novel // 4)
        n_medium = n_novel - n_easy - n_hard
        novel = sorted(easy[:n_easy] + medium[:n_medium] + hard[:n_hard])
        known = sorted([c for c in pool_classes if c not in novel])
        return novel, known, {
            "n_easy": n_easy, "n_medium": n_medium, "n_hard": n_hard}
    raise ValueError(f"unknown episode mode {mode}")


def build_episode(mode, rng, np_rng, z_cache, by_class, pool_classes,
                  n_known=30, n_novel=8, known_per_class=2,
                  pad_options=(0, 50, 150, 300), n_confusers=12,
                  tiers=None):
    """Construct one counterfactual episode (train-side, causal)."""
    novel_classes, known_classes, meta = choose_novel_classes(
        mode, rng, np_rng, z_cache, by_class, pool_classes,
        n_known=n_known, n_novel=n_novel, tiers=tiers)
    protos, radii = build_protos(z_cache, by_class, known_classes)
    tier_of = {}
    if mode == "mixed":
        for tier_name in ("easy", "medium", "hard"):
            for c in tiers[tier_name]:
                tier_of[c] = tier_name
    elif mode == "hard":
        tier_of = {c: "hard" for c in novel_classes}
    else:
        tier_of = {c: "random" for c in novel_classes}

    known_queries = []
    for c in known_classes:
        ids = list(by_class[c])
        rng.shuffle(ids)
        for sid in ids[:known_per_class]:
            known_queries.append({"sample_id": sid, "label": c,
                                  "known": True, "first": False,
                                  "_frames": None})

    novel_queries = []
    seen = set()
    for c in novel_classes:
        for sid in by_class[c]:
            first = c not in seen
            seen.add(c)
            novel_queries.append({"sample_id": sid, "label": c,
                                  "known": False, "first": first,
                                  "_frames": None, "tier": tier_of[c]})
    rng.shuffle(novel_queries)

    n_pad = rng.choice(pad_options)
    pad_pool = [sid for c in known_classes for sid in by_class[c]]
    rng.shuffle(pad_pool)
    pad_sids = pad_pool[:n_pad]
    pad_counts = [int(rng.choice([1, 3, 10, 30])) for _ in pad_sids]

    conf_pool = [sid for c in known_classes for sid in by_class[c]]
    rng.shuffle(conf_pool)
    conf_z = []
    for sid in conf_pool[:n_confusers]:
        base = z_cache[sid]
        alpha = float(np_rng.uniform(0.62, 0.82))
        w = np_rng.randn(768).astype(np.float32)
        w = _norm(w)
        conf = _norm((alpha * base + (1.0 - alpha) * w).astype(np.float32))
        conf_z.append(conf)

    return {
        "mode": mode,
        "known_classes": known_classes,
        "novel_classes": novel_classes,
        "protos": protos,
        "radii": radii,
        "queries": known_queries + novel_queries,
        "pad_sids": pad_sids,
        "pad_counts": pad_counts,
        "conf_z": conf_z,
        "meta": meta,
    }
