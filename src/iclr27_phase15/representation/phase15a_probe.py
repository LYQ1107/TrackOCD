"""Phase-15A raw-feature correspondence ceiling and causal localization probe.

The script deliberately keeps the physical DSCT proposal rows unchanged.  It
trains only a small relation verifier on public, category/video-disjoint TAO
track pairs, evaluates category-disjoint retrieval, calibrates one fixed
threshold on the public calibration role, and replays a bounded causal linker
on the Phase14C DEV+ proposal sidecar.  Evaluator labels are loaded only after
all decisions have been produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase14c.evaluation.strict_mixed import evaluate as strict_evaluate
from src.iclr27_phase14c.evaluation.strict_mixed import oracle_controls

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DEFAULT_PREFIXES = [1, 2, 4, 8, 16]
SEEDS = [20260824, 20260825]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def l2_one(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def interval(values: Iterable[float]) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if len(a) == 0:
        return {"n": 0, "mean": None, "low": None, "high": None, "std": None}
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a))
    return {"n": int(len(a)), "mean": m, "low": m - 1.96 * se,
            "high": m + 1.96 * se, "std": sd}


def load_public() -> tuple[dict[str, np.ndarray], dict]:
    source = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz",
                     allow_pickle=False)
    manifest = json.loads((ROOT /
                           "outputs/iclr27_phase15/manifests/phase15_preregistration.json").read_text())
    return {k: source[k] for k in source.files}, manifest


def build_prefixes(frame_feats: np.ndarray, frame_mask: np.ndarray,
                   prefixes: list[int]) -> dict[int, np.ndarray]:
    """Causal normalized prefix means, one vector per physical track."""
    out = {}
    for p in prefixes:
        q = min(int(p), frame_feats.shape[1])
        x = frame_feats[:, :q].astype(np.float32)
        m = frame_mask[:, :q].astype(np.float32)[..., None]
        denom = np.maximum(m.sum(axis=1), 1.0)
        out[p] = l2_rows((x * m).sum(axis=1) / denom)
    return out


def _choose_pairs(indices: list[tuple[int, int]], n: int,
                  rng: np.random.Generator) -> list[tuple[int, int]]:
    if len(indices) <= n:
        arr = list(indices)
        rng.shuffle(arr)
        return arr
    chosen = rng.choice(len(indices), size=n, replace=False)
    return [indices[int(i)] for i in chosen]


def build_pair_indices(track_indices: list[int], labels: np.ndarray,
                       videos: np.ndarray, vectors: np.ndarray, seed: int,
                       max_positive: int = 3000,
                       max_negative: int = 3000) -> tuple[np.ndarray, np.ndarray, dict]:
    """Balanced positive/random-negative/hard-negative pair table."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in track_indices:
        groups[int(labels[i])].append(int(i))
    positives = []
    for cat, group in sorted(groups.items()):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                if int(videos[i]) != int(videos[j]):
                    positives.append((i, j))
    positives = _choose_pairs(positives, max_positive, rng)

    idx = np.asarray(track_indices, dtype=np.int64)
    vv = vectors[idx]
    random_neg_pool = []
    hard_neg_pool = []
    for a, i in enumerate(track_indices):
        sims = vv @ vv[a]
        order = np.argsort(-sims, kind="stable")
        hard = None
        for b in order:
            j = int(idx[int(b)])
            if int(videos[j]) != int(videos[i]) and int(labels[j]) != int(labels[i]):
                hard = j
                break
        if hard is not None:
            hard_neg_pool.append((int(i), int(hard)))
    # A finite candidate pool makes the sampling deterministic and cheap.
    seen = set()
    for _ in range(max_negative * 5 + 100):
        i, j = rng.choice(idx, size=2, replace=False)
        i, j = int(i), int(j)
        if int(videos[i]) == int(videos[j]) or int(labels[i]) == int(labels[j]):
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            random_neg_pool.append((i, j))
        if len(random_neg_pool) >= max_negative:
            break
    half = max_negative // 2
    hard = _choose_pairs(hard_neg_pool, half, rng)
    random_n = max_negative - len(hard)
    randoms = _choose_pairs(random_neg_pool, random_n, rng)
    negatives = hard + randoms
    if len(negatives) < max_negative:
        # This fallback is only relevant for unusually tiny category groups.
        all_candidates = []
        for a, i in enumerate(track_indices):
            for j in track_indices[a + 1:]:
                if int(videos[i]) != int(videos[j]) and int(labels[i]) != int(labels[j]):
                    all_candidates.append((int(i), int(j)))
        negatives = _choose_pairs(all_candidates, max_negative, rng)
    pairs = positives + negatives[:max_negative]
    y = np.asarray([1] * len(positives) + [0] * min(len(negatives), max_negative), dtype=np.float32)
    perm = rng.permutation(len(pairs))
    pairs_arr = np.asarray([pairs[int(k)] for k in perm], dtype=np.int64)
    y = y[perm]
    return pairs_arr, y, {
        "positive_pairs_available": int(sum(
            1 for cat, group in groups.items() for a in range(len(group))
            for b in range(a + 1, len(group))
            if int(videos[group[a]]) != int(videos[group[b]]))),
        "positive_pairs_sampled": int(len(positives)),
        "negative_pairs_sampled": int(len(negatives[:max_negative])),
        "hard_negative_pairs_sampled": int(len(hard)),
        "random_negative_pairs_sampled": int(len(randoms)),
        "categories": int(len(groups)),
    }


def build_temporal_pairs(track_indices: list[int], prefixes: dict[int, np.ndarray],
                         seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pairs with positives only within one physical trajectory.

    The category labels are not touched here.  Positives compare prefix-1 and
    prefix-8 vectors of the same track; negatives compare vectors from two
    different tracks.  This is a control for temporal consistency rather than
    cross-instance correspondence.
    """
    rng = np.random.default_rng(seed)
    p1, p8 = prefixes[min(prefixes)], prefixes[max(prefixes)]
    pairs = []
    for i in track_indices:
        pairs.append((int(i), int(i)))
    for i in track_indices:
        j = int(rng.choice(track_indices))
        if j == i and len(track_indices) > 1:
            j = int(track_indices[(track_indices.index(i) + 1) % len(track_indices)])
        pairs.append((int(i), int(j)))
    y = np.asarray([1] * len(track_indices) + [0] * len(track_indices), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), y, {
        "positive_pairs": len(track_indices), "negative_pairs": len(track_indices),
        "positive_supervision": "within_physical_track_prefix_1_to_8_only",
        "category_labels_used_for_pair_construction": False,
        "physical_id_used_as_feature": False,
    }


def relation_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.concatenate([a, b, np.abs(a - b), a * b], axis=-1).astype(np.float32)


class RelationMLP(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_relation(features: np.ndarray, labels: np.ndarray, seed: int,
                   device: torch.device, steps: int, batch_size: int = 256) -> tuple[RelationMLP, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = RelationMLP(features.shape[1] // 4).to(device)
    model.train()
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    losses = []
    for step in range(int(steps)):
        ix = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),),
                           generator=gen, device=device)
        logits = model(x[ix])
        loss = F.binary_cross_entropy_with_logits(logits, y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x)).detach().cpu().numpy()
    pred = (train_prob >= 0.5).astype(np.float32)
    return model, {"steps": int(steps), "seed": int(seed),
                   "loss_trace": losses,
                   "train_accuracy": float((pred == labels).mean()),
                   "train_positive_mean": float(train_prob[labels > 0.5].mean()),
                   "train_negative_mean": float(train_prob[labels < 0.5].mean())}


def score_pairs_model(model: RelationMLP, a: np.ndarray, b: np.ndarray,
                      device: torch.device, batch: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(a), batch):
            z = relation_features(a[start:start + batch], b[start:start + batch])
            logits = model(torch.from_numpy(z).to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


def causal_prefilter(base: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     max_candidates: int = 8) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Keep the causal linker cheap without changing its state semantics.

    The raw cosine prefilter is computed against every bank item, then the
    learned verifier is evaluated on only the top eight candidates.  The
    returned vector remains aligned with the complete bank (non-selected
    entries receive a value below any legal threshold).  This is a bounded
    candidate index, not a post-hoc label/metric filter.
    """
    def score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(b) == 0:
            return np.zeros((0,), dtype=np.float32)
        anchor = np.asarray(a[0], dtype=np.float32)
        raw = np.asarray(b, dtype=np.float32) @ anchor
        k = min(int(max_candidates), len(b))
        ids = np.argsort(-raw, kind="stable")[:k]
        out = np.full((len(b),), -1e6, dtype=np.float32)
        aa = np.repeat(anchor[None, :], len(ids), axis=0)
        out[ids] = np.asarray(base(aa, np.asarray(b)[ids]), dtype=np.float32)
        return out
    return score


def score_matrix(model: RelationMLP | None, vectors: np.ndarray,
                 device: torch.device | None, raw: bool = False) -> np.ndarray:
    n = len(vectors)
    if raw:
        return vectors @ vectors.T
    assert model is not None and device is not None
    out = np.zeros((n, n), dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    flat_i, flat_j = ii.reshape(-1), jj.reshape(-1)
    vals = score_pairs_model(model, vectors[flat_i], vectors[flat_j], device)
    out[flat_i, flat_j] = vals
    # A pair relation should be invariant to argument order for retrieval.
    return (out + out.T) / 2.0


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for k in range(bins):
        mask = (scores >= edges[k]) & ((scores < edges[k + 1]) if k + 1 < bins else (scores <= edges[k + 1]))
        if mask.any():
            total += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def pair_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return {"pairs": int(len(scores)), "roc_auc": None, "pr_auc": None,
                "ece": None, "positive_mean": None, "negative_mean": None,
                "gap": None}
    return {
        "pairs": int(len(scores)),
        "positives": int(labels.sum()), "negatives": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "ece": ece(scores, labels),
        "positive_mean": float(scores[labels == 1].mean()),
        "negative_mean": float(scores[labels == 0].mean()),
        "gap": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
    }


def retrieval_from_matrix(vectors: np.ndarray, labels: np.ndarray,
                          videos: np.ndarray, score_mat: np.ndarray) -> dict:
    r1, r5, aps = [], [], []
    by_cat: dict[int, list[float]] = defaultdict(list)
    by_vid: dict[int, list[float]] = defaultdict(list)
    n = len(vectors)
    for i in range(n):
        candidates = [j for j in range(n) if j != i and int(videos[j]) != int(videos[i])]
        pos = {j for j in candidates if int(labels[j]) == int(labels[i])}
        if not pos:
            continue
        ranked = sorted(candidates, key=lambda j: (-float(score_mat[i, j]), j))
        r1v = float(ranked[0] in pos)
        r5v = float(bool(set(ranked[:5]) & pos))
        hit = 0
        ap = 0.0
        for rank, j in enumerate(ranked, 1):
            if j in pos:
                hit += 1
                ap += hit / rank
        ap /= len(pos)
        r1.append(r1v); r5.append(r5v); aps.append(ap)
        by_cat[int(labels[i])].append(r1v)
        by_vid[int(videos[i])].append(r1v)
    return {
        "queries": int(len(r1)), "r1": float(np.mean(r1)) if r1 else None,
        "r5": float(np.mean(r5)) if r5 else None,
        "map": float(np.mean(aps)) if aps else None,
        "category_grouped_r1": interval([np.mean(v) for v in by_cat.values()]),
        "video_grouped_r1": interval([np.mean(v) for v in by_vid.values()]),
        "eligible_categories": int(len(by_cat)), "eligible_videos": int(len(by_vid)),
    }


def pair_table_from_matrix(score_mat: np.ndarray, labels: np.ndarray,
                           videos: np.ndarray, cross_video: bool | None) -> tuple[np.ndarray, np.ndarray]:
    scores, ys = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            is_cross = int(videos[i]) != int(videos[j])
            if cross_video is not None and is_cross != cross_video:
                continue
            scores.append(float(score_mat[i, j]))
            ys.append(int(labels[i] == labels[j]))
    return np.asarray(scores, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def prototype_matrix(vectors: np.ndarray, labels: np.ndarray,
                     videos: np.ndarray) -> np.ndarray:
    """Leave-one-track-out, cross-video category prototype score matrix."""
    n = len(vectors)
    out = np.full((n, n), -1e9, dtype=np.float32)
    categories = sorted(set(int(x) for x in labels))
    for i in range(n):
        proto = {}
        for c in categories:
            ids = [j for j in range(n) if j != i and int(videos[j]) != int(videos[i]) and int(labels[j]) == c]
            if ids:
                proto[c] = l2_one(vectors[ids].mean(axis=0))
        for j in range(n):
            if j != i and int(videos[j]) != int(videos[i]) and int(labels[j]) in proto:
                out[i, j] = float(vectors[i] @ proto[int(labels[j])])
    return out


def exemplar_matrix(vectors: np.ndarray, labels: np.ndarray,
                    videos: np.ndarray, max_exemplars: int = 4) -> np.ndarray:
    """A deterministic four-exemplar category bank for offline localization."""
    n = len(vectors)
    out = np.full((n, n), -1e9, dtype=np.float32)
    order = sorted(range(n), key=lambda i: (int(videos[i]), i))
    for i in range(n):
        bank: dict[int, list[int]] = defaultdict(list)
        for j in order:
            if j == i or int(videos[j]) == int(videos[i]):
                continue
            c = int(labels[j])
            if len(bank[c]) < max_exemplars:
                bank[c].append(j)
        for j in range(n):
            if j == i or int(videos[j]) == int(videos[i]):
                continue
            cands = bank.get(int(labels[j]), [])
            if cands:
                out[i, j] = float(max(vectors[i] @ vectors[k] for k in cands))
    return out


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) == 0:
        return 0.5, {"bacc": None, "grid_points": 0}
    lo, hi = float(scores.min()), float(scores.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.5, {"bacc": None, "grid_points": 0}
    if hi <= lo + 1e-12:
        grid = np.asarray([lo])
    else:
        grid = np.linspace(lo, hi, 100)
    best_t, best_b = float(grid[0]), -1.0
    for t in grid:
        pred = scores >= t
        pos = labels == 1
        neg = labels == 0
        b = 0.5 * (float((pred[pos]).mean()) if pos.any() else 0.0) + \
            0.5 * (float((~pred[neg]).mean()) if neg.any() else 0.0)
        if b > best_b + 1e-12 or (abs(b - best_b) <= 1e-12 and float(t) < best_t):
            best_b, best_t = b, float(t)
    return best_t, {"bacc": float(best_b), "grid_points": int(len(grid)),
                    "min": lo, "max": hi}


def make_pairs_for_role(role: str, manifest: dict, labels: np.ndarray,
                        videos: np.ndarray, vectors: np.ndarray, seed: int,
                        temporal: bool = False):
    idx = [int(x) for x in manifest["split"][role]["track_indices"]]
    if temporal:
        # This function is called with a dict through the caller for temporal.
        raise RuntimeError("temporal pairs require prefix vectors")
    return build_pair_indices(idx, labels, videos, vectors, seed)


def calibration_pairs(role: str, manifest: dict, labels: np.ndarray,
                      videos: np.ndarray, vectors: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    idx = [int(x) for x in manifest["split"][role]["track_indices"]]
    return build_pair_indices(idx, labels, videos, vectors, seed,
                              max_positive=1000, max_negative=1000)


def build_known_prototypes(data: dict, dev_videos: set[int], known_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, (cat, vid, known) in enumerate(zip(data["labels"], data["video_ids"], data["is_known"])):
        if int(known) and int(cat) in known_ids and int(vid) not in dev_videos:
            groups[int(cat)].append(i)
    cats = np.asarray(sorted(groups), dtype=np.int64)
    protos = np.asarray([l2_one(data["mean_feats"][groups[c]].astype(np.float32).mean(axis=0))
                         for c in cats], dtype=np.float32)
    return cats, protos


def write_semantic_csv(path: Path, rows: list[dict], decisions: dict[int, dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for f in ("sem_action", "sem_sid", "sem_kscore", "sem_slot"):
        if f not in fields:
            fields.append(f)
    output = []
    for i, r in enumerate(rows):
        q = dict(r)
        q.update(decisions[i])
        output.append(q)
    buf = []
    import io
    s = io.StringIO()
    writer = csv.DictWriter(s, fieldnames=fields)
    writer.writeheader(); writer.writerows(output)
    atomic_text(path, s.getvalue())


def causal_link(rows: list[dict], feats: np.ndarray, known_cats: np.ndarray,
                known_protos: np.ndarray, scorer: Callable[[np.ndarray, np.ndarray], np.ndarray],
                threshold: float, name: str) -> tuple[dict[int, dict], dict]:
    """Causal bounded four-exemplar state machine."""
    chrono = sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id", 0)), int(rows[i]["track_id"])))
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: Counter = Counter()
    states: dict[int, dict] = {}
    track_decisions: dict[tuple[int, int], dict] = {}
    next_state = 0
    decisions: dict[int, dict] = {}
    known_scores_seen = []
    state_scores_seen = []
    for i in chrono:
        r = rows[i]
        key = (int(r["video_id"]), int(r["track_id"]))
        # Once the physical tracker has assigned a semantic action at birth,
        # carry that immutable action through subsequent occurrences.  This
        # is the ordinary causal MOT carry-forward and avoids re-running a
        # relation network on identical physical state updates.
        if key in track_decisions:
            decisions[i] = dict(track_decisions[key])
            continue
        x = np.asarray(feats[i], dtype=np.float32)
        sums[key] = sums.get(key, np.zeros_like(x)) + x
        counts[key] += 1
        cur = l2_one(sums[key] / float(counts[key]))
        kscore_vec = scorer(np.repeat(cur[None, :], len(known_protos), axis=0), known_protos)
        if len(kscore_vec):
            kpos = int(np.argmax(kscore_vec))
            kscore = float(kscore_vec[kpos])
        else:
            kpos, kscore = -1, 0.0
        known_scores_seen.append(kscore)
        if kscore >= threshold and kpos >= 0:
            decisions[i] = {"sem_action": "known", "sem_sid": int(known_cats[kpos]),
                            "sem_kscore": kscore, "sem_slot": int(known_cats[kpos])}
            track_decisions[key] = dict(decisions[i])
            continue
        # Flatten all state exemplars once, use the same bounded raw prefilter
        # globally, and run the learned verifier only on the top candidates.
        # The previous per-state loop made a hard-negative state bank quadratic
        # when a low calibration score caused many births.
        flat_vectors, flat_owner, flat_local = [], [], []
        for sid in sorted(states):
            for j, ex in enumerate(states[sid]["vectors"]):
                flat_vectors.append(ex); flat_owner.append(sid); flat_local.append(j)
        best_sid, best_score, best_ex = None, -1.0, -1
        if flat_vectors:
            flat = np.asarray(flat_vectors, dtype=np.float32)
            q = scorer(np.repeat(cur[None, :], len(flat), axis=0), flat)
            qj = int(np.argmax(q)); best_score = float(q[qj])
            best_sid = int(flat_owner[qj]); best_ex = int(flat_local[qj])
        state_scores_seen.append(best_score if best_sid is not None else 0.0)
        if best_sid is not None and best_score >= threshold:
            decisions[i] = {"sem_action": "existing", "sem_sid": int(100000 + best_sid),
                            "sem_kscore": kscore, "sem_slot": int(best_sid)}
            track_decisions[key] = dict(decisions[i])
            st = states[best_sid]
            if len(st["vectors"]) < 4:
                st["vectors"].append(cur.copy()); st["quality"].append(best_score)
            elif best_score > min(st["quality"]):
                j = int(np.argmin(st["quality"]))
                st["vectors"][j] = cur.copy(); st["quality"][j] = best_score
        else:
            sid = next_state; next_state += 1
            states[sid] = {"vectors": [cur.copy()], "quality": [1.0], "birth": key}
            decisions[i] = {"sem_action": "new", "sem_sid": int(100000 + sid),
                            "sem_kscore": kscore, "sem_slot": int(sid)}
            track_decisions[key] = dict(decisions[i])
    return decisions, {
        "name": name, "rows": int(len(rows)), "states_born": int(len(states)),
        "max_exemplars": 4, "threshold": float(threshold),
        "physical_track_carry_forward": True,
        "known_score_min": float(min(known_scores_seen)) if known_scores_seen else None,
        "known_score_max": float(max(known_scores_seen)) if known_scores_seen else None,
        "future_frames_used": False, "q1_label_used": False,
        "private_gt_used_for_decision": False, "physical_id_used_as_feature": False,
    }


def proposal_rows_and_features() -> tuple[list[dict], np.ndarray, list[dict]]:
    p = ROOT / "outputs/iclr27_phase14c/proposals/proposals_mixed.csv"
    rows = [dict(r) for r in csv.DictReader(p.open())]
    feats = np.load(ROOT / "outputs/iclr27_phase14c/features/proposal_dinov2.npz",
                    allow_pickle=False)["feats"].astype(np.float32)
    aligned = [dict(r) for r in csv.DictReader(
        (ROOT / "outputs/iclr27_phase14c/proposals/proposals_aligned.csv").open())]
    if len(rows) != len(feats) or len(rows) != len(aligned):
        raise RuntimeError("proposal/feature/alignment length mismatch")
    return rows, feats, aligned


def run(args) -> None:
    started = time.time()
    data, manifest = load_public()
    labels = data["labels"].astype(np.int64)
    videos = data["video_ids"].astype(np.int64)
    track_ids = data["track_ids"].astype(np.int64)
    prefixes = build_prefixes(data["frame_feats"], data["frame_mask"], DEFAULT_PREFIXES)
    train_idx = [int(x) for x in manifest["split"]["representation_train"]["track_indices"]]
    cal_idx = [int(x) for x in manifest["split"]["calibration"]["track_indices"]]
    meta_idx = [int(x) for x in manifest["split"]["meta_validation"]["track_indices"]]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Pair fitting uses the preregistered maximum public prefix only.
    train_pairs, train_y, train_pair_info = build_pair_indices(
        train_idx, labels, videos, prefixes[8], SEEDS[0])
    train_a = prefixes[8][train_pairs[:, 0]]
    train_b = prefixes[8][train_pairs[:, 1]]
    train_x = relation_features(train_a, train_b)
    models: dict[str, RelationMLP] = {}
    train_logs = {}
    for seed in SEEDS:
        # Reorder pair sampling by seed only through deterministic orientation;
        # the registered pair table remains the same category-disjoint pool.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_pairs))
        model, log = train_relation(train_x[order], train_y[order], seed,
                                    device, args.steps)
        models[f"relation_seed{seed}"] = model
        train_logs[f"relation_seed{seed}"] = log
        atomic_torch(ROOT / f"outputs/iclr27_phase15/checkpoints/relation_seed{seed}.pth",
                     {"state_dict": model.state_dict(), "seed": seed,
                      "dim": 768, "training_prefix": 8,
                      "pair_info": train_pair_info})

    temp_pairs, temp_y, temp_info = build_temporal_pairs(train_idx, prefixes, SEEDS[0])
    temp_a = prefixes[1][temp_pairs[:, 0]]
    # A same-track positive is prefix-1 versus prefix-8; negatives use
    # prefix-1 versus a different track's prefix-8.
    temp_b = np.where(np.arange(len(temp_pairs))[:, None] < len(train_idx),
                      prefixes[8][temp_pairs[:, 0]], prefixes[8][temp_pairs[:, 1]])
    temp_x = relation_features(temp_a, temp_b)
    temporal_model, temporal_log = train_relation(temp_x, temp_y, SEEDS[0],
                                                  device, args.steps)
    models["temporal_only"] = temporal_model
    atomic_torch(ROOT / "outputs/iclr27_phase15/checkpoints/temporal_only.pth",
                 {"state_dict": temporal_model.state_dict(), "seed": SEEDS[0],
                  "dim": 768, "training_prefix": "within_track_1_to_8"})

    # Calibration is a fixed public role and never reads proposal labels.
    cal_pairs, cal_y, cal_info = calibration_pairs(
        "calibration", manifest, labels, videos, prefixes[8], SEEDS[0])
    cal_a, cal_b = prefixes[8][cal_pairs[:, 0]], prefixes[8][cal_pairs[:, 1]]
    cal_raw = np.clip((np.sum(cal_a * cal_b, axis=1) + 1.0) / 2.0, 0.0, 1.0)
    cal_rel = {name: score_pairs_model(model, cal_a, cal_b, device)
               for name, model in models.items() if name != "temporal_only"}
    cal_temp = score_pairs_model(temporal_model, cal_a, cal_b, device)
    thresholds = {}
    threshold_details = {}
    thresholds["raw_cosine"] , threshold_details["raw_cosine"] = select_threshold(cal_raw, cal_y)
    thresholds["temporal_only"], threshold_details["temporal_only"] = select_threshold(cal_temp, cal_y)
    for name, score in cal_rel.items():
        thresholds[name], threshold_details[name] = select_threshold(score, cal_y)

    # Category-disjoint meta-validation metrics.
    meta_labels = labels[meta_idx]
    meta_videos = videos[meta_idx]
    offline = {
        "protocol": "phase15a", "train_tracks": len(train_idx),
        "calibration_tracks": len(cal_idx), "meta_tracks": len(meta_idx),
        "train_pair_info": train_pair_info, "calibration_pair_info": cal_info,
        "training_prefix": 8, "prefixes": DEFAULT_PREFIXES,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "threshold_details": threshold_details, "representations": {},
        "train_logs": train_logs, "temporal_log": temporal_log,
        "leakage_flags": {"q1_label_used": False, "devplus_used_for_fit": False,
                           "devplus_used_for_calibration": False,
                           "future_frames_used": False, "physical_id_used_as_feature": False},
    }
    for p in DEFAULT_PREFIXES:
        if p > 8:
            continue
        vec = prefixes[p][meta_idx]
        raw_mat = score_matrix(None, vec, None, raw=True)
        proto_mat = prototype_matrix(vec, meta_labels, meta_videos)
        ex_mat = exemplar_matrix(vec, meta_labels, meta_videos, max_exemplars=4)
        raw_cv, raw_y = pair_table_from_matrix(raw_mat, meta_labels, meta_videos, True)
        proto_cv, _ = pair_table_from_matrix(proto_mat, meta_labels, meta_videos, True)
        ex_cv, _ = pair_table_from_matrix(ex_mat, meta_labels, meta_videos, True)
        item = {
            "raw_cosine": {
                "pair_cross_video": pair_metrics(np.clip((raw_cv + 1) / 2, 0, 1), raw_y),
                "retrieval": retrieval_from_matrix(vec, meta_labels, meta_videos, raw_mat),
            },
            "category_prototype": {
                "pair_cross_video": pair_metrics(np.clip((proto_cv + 1) / 2, 0, 1), raw_y),
                "retrieval": retrieval_from_matrix(vec, meta_labels, meta_videos, proto_mat),
            },
            "exemplar_bank_4": {
                "pair_cross_video": pair_metrics(np.clip((ex_cv + 1) / 2, 0, 1), raw_y),
                "retrieval": retrieval_from_matrix(vec, meta_labels, meta_videos, ex_mat),
            },
        }
        # Same-video and cross-video relation tables for both fixed seeds.
        for name, model in models.items():
            if name == "temporal_only":
                continue
            rel_mat = score_matrix(model, vec, device, raw=False)
            cv, cy = pair_table_from_matrix(rel_mat, meta_labels, meta_videos, True)
            sv, sy = pair_table_from_matrix(rel_mat, meta_labels, meta_videos, False)
            item[name] = {
                "pair_cross_video": pair_metrics(cv, cy),
                "pair_same_video": pair_metrics(sv, sy),
                "retrieval": retrieval_from_matrix(vec, meta_labels, meta_videos, rel_mat),
            }
        tmat = score_matrix(temporal_model, vec, device, raw=False)
        tcv, tcy = pair_table_from_matrix(tmat, meta_labels, meta_videos, True)
        tsv, tsy = pair_table_from_matrix(tmat, meta_labels, meta_videos, False)
        item["temporal_only"] = {
            "pair_cross_video": pair_metrics(tcv, tcy),
            "pair_same_video": pair_metrics(tsv, tsy),
            "retrieval": retrieval_from_matrix(vec, meta_labels, meta_videos, tmat),
        }
        offline["representations"][f"prefix{p}"] = item

    if args.skip_online:
        atomic_json(ROOT / "outputs/iclr27_phase15/eval/phase15a_offline_summary.json", offline)
        print(json.dumps({"offline_only": True, "duration_seconds": time.time() - started}, indent=2))
        return

    # Public proposal replay.  Feature/label sidecars are loaded separately;
    # only the physical stream and raw features enter this decision loop.
    rows, proposal_feats, aligned = proposal_rows_and_features()
    split14 = json.loads((ROOT / "outputs/iclr27_phase14b/manifests/devplus_split.json").read_text())
    dev_videos = set(int(x) for x in split14["devplus_videos"])
    known_ids = set(int(x) for x in json.loads(
        (ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json").read_text()))
    known_cats, known_protos = build_known_prototypes(data, dev_videos, known_ids)
    online = {"protocol": "phase15a", "known_prototype_categories": int(len(known_cats)),
              "candidates": {}, "leakage_flags": offline["leakage_flags"]}

    def raw_scorer(a, b):
        return np.sum(a * b, axis=1).astype(np.float32) * 0.5 + 0.5

    scorers = {"raw_cosine": raw_scorer,
               "temporal_only": causal_prefilter(
                   lambda a, b: score_pairs_model(temporal_model, a, b, device))}
    for name, model in models.items():
        if name != "temporal_only":
            scorers[name] = causal_prefilter(
                lambda a, b, m=model: score_pairs_model(m, a, b, device))
    for name, scorer in scorers.items():
        threshold = float(thresholds[name])
        decisions, causal_info = causal_link(rows, proposal_feats, known_cats,
                                             known_protos, scorer, threshold, name)
        csv_path = ROOT / f"outputs/iclr27_phase15/eval/phase15a_{name}.csv"
        write_semantic_csv(csv_path, rows, decisions)
        rel_csv = str(csv_path.relative_to(ROOT))
        sm, eval_rows, eval_aligned, eval_labels, mapping = strict_evaluate(
            rel_csv, "outputs/iclr27_phase14c/proposals/proposals_aligned.csv",
            "outputs/iclr27_phase14c/manifests/mixed_gt_tracks.jsonl")
        controls = oracle_controls(eval_rows, eval_aligned, eval_labels, mapping)
        payload = {"protocol": "phase15a", "name": name, "threshold": threshold,
                   "strict": sm, "evaluator_controls": controls,
                   "causal_info": causal_info,
                   "legacy_gate": {"known_ge_0_60": sm["known_occurrence_acc"] >= 0.60,
                                   "ct_reuse_gt_0": sm["ct_reuse"] > 0,
                                   "pass": sm["known_occurrence_acc"] >= 0.60 and sm["ct_reuse"] > 0}}
        atomic_json(ROOT / f"outputs/iclr27_phase15/eval/phase15a_{name}_strict.json", payload)
        online["candidates"][name] = payload

    atomic_json(ROOT / "outputs/iclr27_phase15/eval/phase15a_offline_summary.json", offline)
    atomic_json(ROOT / "outputs/iclr27_phase15/eval/phase15a_online_summary.json", online)

    # Deterministic branch selection, fixed before reading DEV+ values.
    p8 = offline["representations"]["prefix8"]
    raw_r = p8["raw_cosine"]["retrieval"]["r1"] or 0.0
    raw_m = p8["raw_cosine"]["retrieval"]["map"] or 0.0
    temp_r = p8["temporal_only"]["retrieval"]["r1"] or 0.0
    rel_entries = [p8[f"relation_seed{s}"] for s in SEEDS]
    rel_r = float(np.mean([x["retrieval"]["r1"] or 0.0 for x in rel_entries]))
    rel_m = float(np.mean([x["retrieval"]["map"] or 0.0 for x in rel_entries]))
    rel_auc = float(np.mean([x["pair_cross_video"]["roc_auc"] or 0.0 for x in rel_entries]))
    raw_auc = p8["raw_cosine"]["pair_cross_video"]["roc_auc"] or 0.0
    temp_auc = p8["temporal_only"]["pair_cross_video"]["roc_auc"] or 0.0
    offline_strong = bool(rel_r >= raw_r + 0.02 and rel_m >= raw_m + 0.02 and
                          rel_auc >= max(raw_auc, temp_auc) + 0.01)
    offline_weak = bool(rel_r <= raw_r - 0.02 or rel_m <= raw_m - 0.02 or
                        rel_auc <= raw_auc - 0.01)
    def get_strict(name):
        return online["candidates"][name]["strict"]
    raw_sm = get_strict("raw_cosine")
    rel_sms = [get_strict(f"relation_seed{s}") for s in SEEDS]
    rel_known = float(np.mean([x["known_occurrence_acc"] for x in rel_sms]))
    rel_ct = float(np.mean([x["ct_reuse"] for x in rel_sms]))
    raw_known = float(raw_sm["known_occurrence_acc"])
    raw_ct = float(raw_sm["ct_reuse"])
    online_improves = bool(rel_known >= raw_known - 0.05 and rel_ct > raw_ct)
    known_only = bool(rel_known > raw_known + 0.01 and rel_ct <= raw_ct)
    if offline_strong and online_improves:
        branch, status = "A", "authorize_phase15b_full_episodic_linker"
    elif offline_strong and not online_improves:
        branch, status = "B", "authorize_phase15b_explicit_three_way_state_probe"
    elif offline_weak:
        branch, status = "D", "run_one_crop_tube_diagnostic_then_stop_architecture_tuning"
    elif known_only:
        branch, status = "C", "authorize_phase15b_novelty_focused_probe"
    else:
        branch, status = "D", "run_one_crop_tube_diagnostic_then_stop_architecture_tuning"
    decision = {
        "protocol": "phase15a", "branch": branch, "status": status,
        "offline_strong": offline_strong, "offline_weak": offline_weak,
        "online_improves": online_improves,
        "known_only": known_only,
        "criteria": {"offline_r1_margin": 0.02, "offline_map_margin": 0.02,
                     "offline_auc_margin": 0.01, "online_known_floor": -0.05,
                     "online_ct_strictly_above_raw": True},
        "evidence": {"raw_r1": raw_r, "relation_mean_r1": rel_r,
                     "raw_map": raw_m, "relation_mean_map": rel_m,
                     "raw_auc": raw_auc, "relation_mean_auc": rel_auc,
                     "temporal_r1": temp_r, "temporal_auc": temp_auc,
                     "raw_known": raw_known, "relation_mean_known": rel_known,
                     "raw_ct_reuse": raw_ct, "relation_mean_ct_reuse": rel_ct},
        "q1_opened": False, "final_gate_passed": False,
    }
    atomic_json(ROOT / "outputs/iclr27_phase15/eval/phase15a_decision.json", decision)
    # The selected strict summary is a diagnostic handoff; Q1 remains closed.
    selected = online["candidates"]["relation_seed20260824"]
    atomic_json(ROOT / "outputs/iclr27_phase15/eval/strict_trackocd_summary.json", selected)
    atomic_json(ROOT / "outputs/iclr27_phase15/eval/causal_contract.json", {
        "protocol": "phase15a", "selected_candidate": "relation_seed20260824",
        "contract": selected["strict"]["causal_contract"],
        "q1_label_used": False, "future_frames_used": False,
        "physical_id_used_as_feature": False, "private_gt_used_for_decision": False,
    })
    atomic_json(ROOT / "outputs/iclr27_phase15/eval/resource_summary.json", {
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "one_gpu_max": True, "duration_seconds": time.time() - started,
        "phase15b_run": False, "q1_run": False,
    })
    print(json.dumps({"branch": branch, "status": status,
                      "offline_strong": offline_strong, "offline_weak": offline_weak,
                      "online_improves": online_improves,
                      "raw_known": raw_known, "rel_known": rel_known,
                      "raw_ct": raw_ct, "rel_ct": rel_ct}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--skip-online", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
