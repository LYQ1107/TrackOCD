"""Strict Q1 replay for a Phase-10 representation checkpoint.

The Phase-8A B create head and online state process are frozen.  Only the
trajectory representation is replaced, allowing a representation-level Q1
comparison without adding a new semantic-memory mechanism.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase10.model.hybrid import HybridTrajectoryEncoder  # noqa: E402
from src.iclr27_phase8a.model.adapter import TorchSemanticStateSet  # noqa: E402
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import (  # noqa: E402
    load_tse,
    project,
)
from src.iclr27_phase6c.model.tse import TSE  # noqa: E402


def load_rows(path):
    with open(ROOT / path, newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd], list(rd.fieldnames or [])


def known_embeddings(device, model, tse, raw, mask, labels):
    z = project(device, tse, raw.reshape(-1, raw.shape[-1]).astype(np.float32))
    z = z.reshape(raw.shape[0], raw.shape[1], -1)
    with torch.no_grad():
        final, _ = model(torch.from_numpy(z).to(device),
                         torch.from_numpy(mask).to(device))
    h = final.cpu().numpy().astype(np.float32)
    ids = np.unique(labels).astype(np.int64)
    mu, cnt = [], []
    for c in ids:
        v = h[labels == c].mean(axis=0)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        mu.append(v)
        cnt.append(float((labels == c).sum()))
    return z, np.asarray(mu, dtype=np.float32), np.asarray(cnt, dtype=np.float32), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.checkpoint, map_location=device,
                    weights_only=False)
    a = ck.get("args", {})
    dim = 128
    model = HybridTrajectoryEncoder(dim=dim, hidden=int(a.get("hidden", 128)),
                                    out_dim=dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    create = CreateHead(dim=dim).to(device)
    bck = torch.load(ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth",
                     map_location=device, weights_only=False)
    create.load_state_dict(bck["create_head"])
    create.eval()
    temp = float(bck.get("temp", 20.0))

    tse, _, _ = load_tse(device)
    known = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    _, mu, cnt, known_ids = known_embeddings(
        device, model, tse, known["frame_feats"].astype(np.float32),
        known["frame_mask"], known["labels"])
    rows, fields = load_rows(args.proposals)
    raw = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"row/feature mismatch: {len(rows)} vs {len(raw)}")
    z = project(device, tse, raw)
    states = TorchSemanticStateSet(dim=dim, max_slots=4096, sigma2=1.0,
                                   score_mode="cosine", cosine_temp=temp).to(device)
    states.init_known(torch.from_numpy(mu).to(device), torch.from_numpy(cnt).to(device))
    track_state = {}
    track_age = {}
    out_action = [""] * len(rows)
    out_sid = [""] * len(rows)
    out_score = [""] * len(rows)
    out_slot = [""] * len(rows)
    chrono = sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            st = track_state.get(key)
            if st is None:
                st = model.new_state(device=device)
            x = torch.from_numpy(z[i]).to(device).unsqueeze(0)
            h, st = model.step(x, st)
            h = h[0]
            track_state[key] = st
            age = int(track_age.get(key, 0) + 1)
            track_age[key] = age
            w = float(age)
            scores = states.log_scores(h, w)
            best = scores.max() if states.n else torch.zeros((), device=device)
            pv = phys_vec(rows[i].get("score", 0.0), rows[i].get("prior_hits", 0.0), age, device)
            create_logit = temp * create(h, pv, best)
            logits = states.logits(h, w, create_logit.reshape(1))
            pred = int(torch.argmax(logits))
            p_assign = (1.0 / (1.0 + torch.exp(
                create_logit - torch.logsumexp(scores, dim=0)))
                        if states.n else torch.zeros((), device=device))
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
            out_action[i] = action
            out_sid[i] = str(int(known_ids[slot]) if action == "known"
                            else 100000 + int(slot))
            out_score[i] = f"{float(p_assign):.6f}"
            out_slot[i] = str(int(slot))

    for n in ("sem_kscore", "sem_slot"):
        if n not in fields:
            fields.append(n)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i, r in enumerate(rows):
            d = dict(r)
            d["sem_action"] = out_action[i]
            d["sem_sid"] = out_sid[i]
            d["sem_kscore"] = out_score[i]
            d["sem_slot"] = out_slot[i]
            wr.writerow(d)
    print(Counter(out_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
