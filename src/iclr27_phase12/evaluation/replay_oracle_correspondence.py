"""Oracle semantic-correspondence ceiling for Phase 12.

This is intentionally an upper bound, not a legal TrackOCD method.  The
frozen Phase-8A B decision/state process is replayed on the unchanged Q1
physical stream.  A hidden Q1 category label is used *offline* to select a
category prototype in the already-trained Phase-8A trajectory space; the
online state machine itself receives only that resulting vector.  False
positive rows retain their ordinary causal B representation.

The experiment answers the narrow feasibility question: if a representation
already knew which novel category a row belongs to, can the unchanged
assign/create evaluator emit a correct cross-physical reuse?  A learned
encoder is deliberately not conflated with this ceiling; the companion legal
synthetic-episode experiment tests whether category supervision can be learned
without Q1 labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase8a.model.adapter import CausalTrajectoryAdapter, TorchSemanticStateSet  # noqa: E402
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project  # noqa: E402
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes  # noqa: E402
from src.iclr27_phase4s.protocol import group_tracks  # noqa: E402
from src.iclr27_phase7a.evaluation.strict_eval_any import load_gt_videos  # noqa: E402


Q1_VIDEO_IDS = [88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276,
                1572, 1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888]
KNOWN_ROLES = {"known", "supported_known"}


def chrono(rows):
    return sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))


def load_rows(path: Path):
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd], list(rd.fieldnames or [])


def atomic_csv(path: Path, fields, rows, actions, sids, scores, slots):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, row in enumerate(rows):
            d = dict(row)
            d["sem_action"] = actions[i]
            d["sem_sid"] = sids[i]
            d["sem_kscore"] = scores[i]
            d["sem_slot"] = slots[i]
            wr.writerow(d)
    os.replace(tmp, path)


def model_and_known(device):
    ck = torch.load(ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth",
                    map_location=device, weights_only=False)
    args = ck.get("args", {})
    dim = int(args.get("dim", 128))
    temp = float(ck.get("temp", 20.0))
    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=bool(args.get("frame_level", False))).to(device)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    create = CreateHead(dim=dim).to(device)
    create.load_state_dict(ck["create_head"])
    create.eval()
    tse, _, _ = load_tse(device)
    # Use the same public train-known track asset used by the Phase-10 replay;
    # this avoids a large 38k-row episode projection and keeps prototype
    # provenance independent of Q1 labels.
    known = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    raw = known["frame_feats"].astype(np.float32)
    mask = known["frame_mask"].astype(np.uint8)
    z = project(device, tse, raw.reshape(-1, raw.shape[-1])).reshape(raw.shape[0], raw.shape[1], -1)
    labels = known["labels"].astype(np.int64)
    known_ids = np.unique(labels).astype(np.int64)
    sums = torch.zeros((len(known_ids), dim), device=device)
    counts = torch.zeros(len(known_ids), device=device)
    cls_idx = {int(c): i for i, c in enumerate(known_ids)}
    with torch.no_grad():
        for i in range(len(z)):
            state = adapter.new_state()
            outputs = []
            for t in range(z.shape[1]):
                if not mask[i, t]:
                    continue
                out, state = adapter(torch.from_numpy(z[i, t]).to(device).unsqueeze(0), state)
                outputs.append(out[0])
            if not outputs:
                continue
            final = outputs[-1]
            j = cls_idx[int(labels[i])]
            sums[j] += final
            counts[j] += 1.0
    mu = torch.nn.functional.normalize(sums / torch.clamp(counts, min=1.0)[:, None], dim=-1)
    count = counts
    return adapter, create, tse, mu, count, known_ids, temp, dim


def causal_vectors(adapter, tse, rows, raw, device):
    z = project(device, tse, raw.astype(np.float32))
    h = np.zeros((len(rows), adapter.dim), dtype=np.float32)
    state_by_track = {}
    with torch.no_grad():
        for i in chrono(rows):
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            state = state_by_track.get(key)
            if state is None:
                state = adapter.new_state()
            out, state = adapter(torch.from_numpy(z[i]).to(device).unsqueeze(0), state)
            state_by_track[key] = state.detach()
            h[i] = out[0].cpu().numpy().astype(np.float32)
    return h


def category_prototypes(rows, h, oracle_labels):
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        role, category = oracle_labels.get(
            (int(row["video_id"]), int(row["track_id"])), ("fp", -1))
        if (role in KNOWN_ROLES or role == "novel") and category >= 0:
            buckets[category].append(h[i])
    out = {}
    for category, values in buckets.items():
        v = np.asarray(values, dtype=np.float32).mean(axis=0)
        out[int(category)] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out, {int(k): len(v) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv")
    ap.add_argument("--feats", default="outputs/iclr27_phase6b/q1/final_dsct/feats.npz")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--style", choices=["prototype", "ideal_novel"], default="prototype")
    args = ap.parse_args()
    device = torch.device(args.device)
    adapter, create, tse, mu, count, known_ids, temp, dim = model_and_known(device)
    rows, fields = load_rows(ROOT / args.proposals)
    for row in rows:
        row["video_id"] = int(row["video_id"])
        row["frame_id"] = int(row["frame_id"])
        row["track_id"] = int(row["track_id"])
        row["proposal_local_id"] = int(row.get("proposal_local_id") or 0)
    raw = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"proposal/feature mismatch: {len(rows)} vs {len(raw)}")
    # Use exactly the same proposal->private-GT alignment as strict_eval_any.
    # This is intentionally permitted only for the oracle ceiling; legal
    # replay and the synthetic experiment never read this mapping.
    gt_stream, gt_labels = load_gt_videos(Q1_VIDEO_IDS)
    mapping = align_pred_to_gt(group_tracks(rows), gt_track_boxes(gt_stream))
    oracle_labels = {}
    for key, sample_id in mapping.items():
        label = gt_labels[sample_id]
        oracle_labels[key] = (
            str(label["protocol_role"]), int(label["ground_truth_category_id"]))
    ordinary = causal_vectors(adapter, tse, rows, raw, device)
    prototypes, category_counts = category_prototypes(rows, ordinary, oracle_labels)
    ideal_novel = {}
    if args.style == "ideal_novel":
        # Construct a deterministic set of novel directions in the nullspace
        # of the known prototype span.  This is a stronger ceiling than the
        # learned Phase-8A geometry: it asks only whether an unmistakable,
        # category-consistent novel vector can pass the frozen create head.
        known_np = mu.detach().cpu().numpy().astype(np.float32)
        _, _, vh = np.linalg.svd(known_np, full_matrices=True)
        rank = int(np.linalg.matrix_rank(known_np))
        null = vh[rank:].T
        rng = np.random.RandomState(12012)
        novel_cats = sorted({
                int(category) for role, category in oracle_labels.values()
            if role == "novel"
        })
        raw_basis = rng.normal(size=(len(novel_cats), null.shape[1])).astype(np.float32)
        q, _ = np.linalg.qr(raw_basis.T)
        for j, category in enumerate(novel_cats):
            v = null @ q[:, j]
            ideal_novel[category] = v / max(float(np.linalg.norm(v)), 1e-12)

    states = TorchSemanticStateSet(
        dim=dim, max_slots=4096, sigma2=1.0,
        score_mode="cosine", cosine_temp=temp).to(device)
    states.init_known(mu.to(device), count.to(device))
    actions = [""] * len(rows)
    sids = [""] * len(rows)
    scores = [""] * len(rows)
    slots = [""] * len(rows)
    ages = defaultdict(int)
    with torch.no_grad():
        for i in chrono(rows):
            row = rows[i]
            role, category = oracle_labels.get(
                (int(row["video_id"]), int(row["track_id"])), ("fp", -1))
            # Oracle ceiling: hidden category is used only to select this
            # offline-fitted category prototype. FP rows remain ordinary B.
            v = None
            if role in KNOWN_ROLES or role == "novel":
                if args.style == "ideal_novel" and role == "novel":
                    v = ideal_novel.get(category)
                elif args.style == "ideal_novel" and category in set(int(x) for x in known_ids):
                    v = mu.detach().cpu().numpy()[list(map(int, known_ids)).index(category)]
                else:
                    v = prototypes.get(category)
            h = torch.from_numpy(v if v is not None else ordinary[i]).to(device)
            key = (int(row["video_id"]), int(row["track_id"]))
            ages[key] += 1
            age = ages[key]
            w = float(age)
            existing = states.log_scores(h, w)
            best = existing.max() if states.n else torch.zeros((), device=device)
            phys = phys_vec(row.get("score", 0.0), row.get("prior_hits", 0.0), age, device)
            create_logit = temp * create(h, phys, best)
            logits = states.logits(h, w, create_logit.reshape(1))
            pred = int(torch.argmax(logits))
            if states.n:
                p_assign = 1.0 / (1.0 + torch.exp(
                    create_logit - torch.logsumexp(existing, dim=0)))
            else:
                p_assign = torch.zeros((), device=device)
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is None:
                    slot = int(torch.argmax(existing))
                    states.assign(slot, h, w)
                    action = "known" if int(states.provenance[slot]) == 0 else "existing"
                else:
                    action = "new"
            else:
                slot = pred
                states.assign(slot, h, w)
                action = "known" if int(states.provenance[slot]) == 0 else "existing"
            actions[i] = action
            sids[i] = str(int(known_ids[slot])) if action == "known" else str(100000 + int(slot))
            scores[i] = f"{float(p_assign):.6f}"
            slots[i] = str(int(slot))

    out = ROOT / args.out_csv
    for name in ("sem_kscore", "sem_slot"):
        if name not in fields:
            fields.append(name)
    atomic_csv(out, fields, rows, actions, sids, scores, slots)
    meta = {
        "mode": f"hidden_category_oracle_{args.style}",
        "oracle_label_used": True,
        "oracle_label_scope": "strict proposal-to-private-GT mapping used only for the ceiling representation",
        "q1_labels_used_for_legal_training": False,
        "future_used": False,
        "physical_id_used_as_feature": False,
        "rows": len(rows),
        "category_counts": category_counts,
        "proposal_to_gt_mapping_tracks": len(mapping),
        "action_counts": dict(Counter(actions)),
        "known_ids": [int(x) for x in known_ids],
        "video_ids": Q1_VIDEO_IDS,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
