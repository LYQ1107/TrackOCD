"""Frozen Phase-8A-B replay after replacing only the crop representation.

The TSE and B adapter/CreateHead weights are frozen.  Public TRAIN-known
DINOv3 crops are used only to recompute representation-space known centroids;
the Q1 stream is processed chronologically and no Q1/private labels enter the
actions.  This is a feature-only compatibility test, not a new architecture.
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
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase8a.model.adapter import CausalTrajectoryAdapter, TorchSemanticStateSet
from src.iclr27_phase8a.model.create_head import CreateHead
from src.iclr27_phase8a.training.train_amortized import phys_vec


def load_rows(path: Path):
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        return [dict(x) for x in rd], list(rd.fieldnames or [])


def load_dino3_train():
    labels = {}
    with (ROOT / "data/tao_ow_ocd_v1/public/train_known_tracks.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[r["sample_id"]] = int(r["category_id"])
    cache = ROOT / "data/caches/features/dinov3_vitb16_lvd1689m/train_known"
    feats, ys = [], []
    for p in sorted(cache.glob("*.json")):
        d = json.loads(p.read_text())
        sid = d["sample_id"]
        if sid not in labels:
            continue
        feats.append(np.asarray(d["frame_embeddings"], dtype=np.float32))
        ys.append(labels[sid])
    if not feats:
        raise RuntimeError("empty DINOv3 train-known cache")
    return feats, np.asarray(ys, dtype=np.int64)


def train_known_centroids(tse, adapter, device):
    frame_feats, labels = load_dino3_train()
    known_ids = np.asarray(sorted(set(int(x) for x in labels)), dtype=np.int64)
    index = {int(c): i for i, c in enumerate(known_ids)}
    sums = torch.zeros(len(known_ids), adapter.dim, device=device)
    counts = torch.zeros(len(known_ids), device=device)
    with torch.no_grad():
        for raw, cat in zip(frame_feats, labels):
            z = project(device, tse, raw)
            state = adapter.new_state()
            for row in z:
                h, state = adapter(torch.from_numpy(row).to(device).unsqueeze(0), state)
                sums[index[int(cat)]] += h[0]
                counts[index[int(cat)]] += 1.0
    mu = torch.nn.functional.normalize(
        sums / torch.clamp(counts[:, None], min=1.0), dim=-1)
    return mu, counts, known_ids


def chrono(rows):
    return sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--checkpoint", default="outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    a = ck.get("args", {})
    dim = int(a.get("dim", 128))
    temp = float(ck.get("temp", a.get("temp", 20.0)))
    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=bool(a.get("frame_level", False))).to(device)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    create = CreateHead(dim=dim).to(device)
    create.load_state_dict(ck["create_head"])
    create.eval()
    tse, _, _ = load_tse(device)
    mu, counts, known_ids = train_known_centroids(tse, adapter, device)

    rows, fields = load_rows(ROOT / args.proposals)
    raw = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"row/feature mismatch: {len(rows)} vs {len(raw)}")
    z_all = project(device, tse, raw)
    states = TorchSemanticStateSet(
        dim=dim, max_slots=4096, sigma2=1.0,
        score_mode="cosine", cosine_temp=temp).to(device)
    states.init_known(mu, counts)
    track_state, track_age = {}, defaultdict(int)
    actions = [""] * len(rows)
    sids = [""] * len(rows)
    kscores = [""] * len(rows)
    slots = [""] * len(rows)
    with torch.no_grad():
        for i in chrono(rows):
            row = rows[i]
            key = (int(row["video_id"]), int(row["track_id"]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            h, st = adapter(torch.from_numpy(z_all[i]).to(device).unsqueeze(0), prev)
            h = h[0]
            track_state[key] = st.detach()
            track_age[key] += 1
            age = int(track_age[key])
            w = 1.0 if adapter.frame_level else float(age)
            scores = states.log_scores(h, w)
            best = scores.max() if states.n else torch.zeros((), device=device)
            pv = phys_vec(row.get("score", 0.0), row.get("prior_hits", 0.0), age, device)
            create_logit = temp * create(h, pv, best)
            logits = states.logits(h, w, create_logit.reshape(1))
            pred = int(torch.argmax(logits))
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is None:
                    slot = int(torch.argmax(scores))
                    states.assign(slot, h, w)
                    action = "known" if int(states.provenance[slot]) == 0 else "existing"
                else:
                    action = "new"
            else:
                slot = pred
                prov = int(states.provenance[slot])
                states.assign(slot, h, w)
                action = "known" if prov == 0 else "existing"
            if states.n and scores.numel():
                p_assign = 1.0 / (1.0 + torch.exp(create_logit - torch.logsumexp(scores, dim=0)))
            else:
                p_assign = torch.zeros((), device=device)
            actions[i] = action
            sids[i] = str(int(known_ids[slot])) if action == "known" else str(100000 + int(slot))
            kscores[i] = f"{float(p_assign):.6f}"
            slots[i] = str(int(slot))

    for name in ("sem_kscore", "sem_slot"):
        if name not in fields:
            fields.append(name)
    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, row in enumerate(rows):
            d = dict(row)
            d.update({"sem_action": actions[i], "sem_sid": sids[i],
                      "sem_kscore": kscores[i], "sem_slot": slots[i]})
            wr.writerow(d)
    os.replace(tmp, out)
    meta = {
        "representation": "frozen DINOv3 CLS -> frozen TSE -> frozen Phase8A adapter",
        "semantic_decision": "Phase8A B frozen CreateHead + TorchSemanticStateSet",
        "known_centroid_source": "public TAO TRAIN DINOv3 cache",
        "q1_labels_used": False, "private_gt_used": False,
        "future_used": False, "physical_id_used_for_semantic_id": False,
        "rows": len(rows), "action_counts": dict(Counter(actions)),
        "known_categories": len(known_ids), "checkpoint": str((ROOT / args.checkpoint).resolve()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
