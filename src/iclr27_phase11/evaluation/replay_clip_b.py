"""Causal Q1 replay for the Phase-11 CLIP trajectory probe.

Only the representation is replaced.  The Phase-8A B decision process is
kept byte-for-byte in spirit: frozen known prototypes, the frozen
``CreateHead``, cosine prototype state updates, and immediate immutable
assign-or-create decisions.  The Q1 evaluator is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase11.model.clip_trajectory import ClipTrajectoryEncoder  # noqa: E402
from src.iclr27_phase8a.model.adapter import TorchSemanticStateSet  # noqa: E402
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402


def load_rows(path: Path):
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd], list(rd.fieldnames or [])


def load_pool(path: Path, max_t: int = 8):
    xs, masks, ids = [], [], []
    for p in sorted(path.glob("*.json")):
        r = json.loads(p.read_text())
        f = np.asarray(r["frame_embeddings"], dtype=np.float32)[:max_t]
        x = np.zeros((max_t, f.shape[1]), dtype=np.float32)
        m = np.zeros(max_t, dtype=np.uint8)
        x[:len(f)] = f
        m[:len(f)] = 1
        xs.append(x)
        masks.append(m)
        ids.append(r["sample_id"])
    if not xs:
        raise RuntimeError(f"empty feature pool: {path}")
    return np.asarray(xs), np.asarray(masks), ids


def known_embeddings(model, device: torch.device):
    labels_by_id = {}
    with (ROOT / "data/trackocd_v1/pure/public/train_known_tracks.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            labels_by_id[r["sample_id"]] = int(r["category_id"])
    raw, mask, ids = load_pool(ROOT / "data/caches/features/clip/train_known_mean")
    if set(ids) != set(labels_by_id):
        raise RuntimeError("known CLIP cache and public labels do not match")
    known_ids = np.asarray(sorted(set(labels_by_id.values())), dtype=np.int64)
    cat2idx = {int(c): i for i, c in enumerate(known_ids)}
    ys = np.asarray([cat2idx[labels_by_id[s]] for s in ids], dtype=np.int64)
    with torch.no_grad():
        final, _ = model(
            torch.from_numpy(raw).to(device),
            torch.from_numpy(mask).to(device),
        )
    h = final.detach().cpu().numpy().astype(np.float32)
    mu, count = [], []
    for i in range(len(known_ids)):
        v = h[ys == i].mean(axis=0)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        mu.append(v)
        count.append(float(np.sum(ys == i)))
    return np.asarray(mu, dtype=np.float32), np.asarray(count, dtype=np.float32), known_ids


def atomic_csv(path: Path, fields, rows, actions, sids, kscores, slots):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, row in enumerate(rows):
            d = dict(row)
            d["sem_action"] = actions[i]
            d["sem_sid"] = sids[i]
            d["sem_kscore"] = kscores[i]
            d["sem_slot"] = slots[i]
            wr.writerow(d)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--clip-feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-t", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = ClipTrajectoryEncoder(
        in_dim=int(ck.get("input_dim", 512)),
        out_dim=int(ck.get("out_dim", 128)),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dim = model.out_dim

    create = CreateHead(dim=dim).to(device)
    bck = torch.load(
        ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth",
        map_location=device,
        weights_only=False,
    )
    create.load_state_dict(bck["create_head"])
    create.eval()
    temp = float(bck.get("temp", 20.0))

    mu, count, known_ids = known_embeddings(model, device)
    rows, fields = load_rows(ROOT / args.proposals)
    raw = np.load(ROOT / args.clip_feats)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"proposal/CLIP row mismatch: {len(rows)} vs {len(raw)}")

    # This is the unchanged Phase-8A B state machine.  ``states`` is the
    # shared semantic memory; ``track_state`` is private only for causal
    # physical-stream recurrence and is never exposed as a semantic ID.
    states = TorchSemanticStateSet(
        dim=dim,
        max_slots=4096,
        sigma2=1.0,
        score_mode="cosine",
        cosine_temp=temp,
    ).to(device)
    states.init_known(torch.from_numpy(mu).to(device), torch.from_numpy(count).to(device))
    track_state, track_age = {}, {}
    actions = [""] * len(rows)
    sids = [""] * len(rows)
    kscores = [""] * len(rows)
    slots = [""] * len(rows)
    chrono = sorted(
        range(len(rows)),
        key=lambda i: (
            int(rows[i]["video_id"]),
            int(rows[i]["frame_id"]),
            int(rows[i].get("proposal_local_id") or 0),
            int(rows[i]["track_id"]),
        ),
    )
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            st = track_state.get(key)
            if st is None:
                st = model.new_state(device=device)
            h, st = model.step(torch.from_numpy(raw[i]).to(device).unsqueeze(0), st)
            h = h[0]
            track_state[key] = st
            age = int(track_age.get(key, 0) + 1)
            track_age[key] = age
            w = float(age)
            scores = states.log_scores(h, w)
            best_sim = scores.max() if states.n else torch.zeros((), device=device)
            pv = phys_vec(rows[i].get("score", 0.0), rows[i].get("prior_hits", 0.0), age, device)
            create_logit = temp * create(h, pv, best_sim)
            logits = states.logits(h, w, create_logit.reshape(1))
            pred = int(torch.argmax(logits))
            logsum = torch.logsumexp(scores, dim=0) if states.n else create_logit
            p_assign = 1.0 / (1.0 + torch.exp(create_logit - logsum)) if states.n else torch.zeros((), device=device)
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is None:
                    slot = int(torch.argmax(scores))
                    states.assign(slot, h, w)
                    prov = int(states.provenance[slot])
                    action = "known" if prov == 0 else "existing"
                else:
                    action = "new"
            else:
                slot = pred
                states.assign(slot, h, w)
                prov = int(states.provenance[slot])
                action = "known" if prov == 0 else "existing"
            actions[i] = action
            sids[i] = str(int(known_ids[slot])) if action == "known" else str(100000 + int(slot))
            kscores[i] = f"{float(p_assign):.6f}"
            slots[i] = str(int(slot))

    out = ROOT / args.out_csv
    for name in ("sem_kscore", "sem_slot"):
        if name not in fields:
            fields.append(name)
    atomic_csv(out, fields, rows, actions, sids, kscores, slots)
    meta = {
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "representation": "frozen OpenAI CLIP ViT-B/32 + causal GRU trajectory adapter",
        "semantic_decision": "Phase8A B frozen CreateHead + TorchSemanticStateSet",
        "q1_labels_used": False,
        "future_used": False,
        "physical_id_used_for_semantic_id": False,
        "rows": len(rows),
        "action_counts": dict(Counter(actions)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(Counter(actions))
    print("wrote", out)


if __name__ == "__main__":
    main()
