"""Strict-causal replay for Architecture B's amortized create head.

The online state memory is cosine attention with an amortized create score;
all decisions are emitted immediately in the frozen physical stream order.
Known-state centroids come only from the legal Phase 7C meta-val TRAIN pool,
matching the Architecture A replay provenance.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase8a.model.adapter import (  # noqa: E402
    CausalTrajectoryAdapter,
    TorchSemanticStateSet,
)
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402
from src.iclr27_phase8a.training.train_bsp import (  # noqa: E402
    compute_centroids,
    load_assets,
)
from src.iclr27_phase7a.training.train_reliability_head import (  # noqa: E402
    load_tse,
    project,
)


def load_rows(path):
    with open(ROOT / path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--adapter-ckpt", required=True,
                    help="Architecture B checkpoint containing adapter/head")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--projection-checkpoint", default=None,
                    help="Optional Phase14C residual projection applied to raw DINOv2 before B.")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(ROOT / args.adapter_ckpt, map_location=dev,
                    weights_only=False)
    ck_args = ck.get("args", {})
    dim = int(ck_args.get("dim", 128))
    temp = float(ck.get("temp", ck_args.get("temp", 20.0)))
    frame_level = bool(ck_args.get("frame_level", False))
    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=frame_level).to(dev)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    create_head = CreateHead(dim=dim).to(dev)
    create_head.load_state_dict(ck["create_head"])
    create_head.eval()

    tse, _, _ = load_tse(dev)
    projection = None
    if args.projection_checkpoint:
        from src.iclr27_phase14c.training.train_projection import ResidualProjection
        pck = torch.load(ROOT / args.projection_checkpoint, map_location=dev,
                         weights_only=False)
        projection = ResidualProjection(rank=int(pck.get("rank", 32))).to(dev)
        projection.load_state_dict(pck["model"])
        projection.eval()
    _, ep, _ = load_assets("hard")
    z_anchor = project(dev, tse, ep["feats"].astype(np.float32))
    if projection is not None:
        with torch.no_grad():
            z_anchor = projection(
                torch.from_numpy(ep["feats"].astype(np.float32)).to(dev),
                torch.from_numpy(z_anchor).to(dev)).cpu().numpy()
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_anchor, dev)

    rows, fieldnames = load_rows(args.proposals)
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    assert len(rows) == len(feats)
    z_all = project(dev, tse, feats)
    if projection is not None:
        with torch.no_grad():
            z_all = projection(torch.from_numpy(feats).to(dev),
                               torch.from_numpy(z_all).to(dev)).cpu().numpy()

    states = TorchSemanticStateSet(
        dim=dim, max_slots=4096, sigma2=1.0,
        score_mode="cosine", cosine_temp=temp).to(dev)
    states.init_known(mu.detach(), cnt.detach())
    track_state = {}
    track_count = {}
    chrono = sorted(
        range(len(rows)),
        key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
                       int(rows[i].get("proposal_local_id") or 0),
                       int(rows[i]["track_id"])))
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    sem_slot = [""] * len(rows)
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            z = torch.from_numpy(z_all[i]).to(dev).unsqueeze(0)
            h, state = adapter(z, prev)
            h = h[0]
            age = int(track_count.get(key, 0) + 1)
            track_count[key] = age
            track_state[key] = state.detach()
            w = 1.0 if frame_level else float(age)
            scores = states.log_scores(h, w)
            best_sim = scores.max() if states.n else torch.zeros(
                (), device=dev)
            phys = phys_vec(rows[i].get("score", 0.0),
                            rows[i].get("prior_hits", 0.0), age, dev)
            # Match the training parameterization: existing scores and the
            # create score are both temperature-scaled logits.
            create_logit = temp * create_head(h, phys, best_sim)
            logits = states.logits(h, w, create_logit.reshape(1))
            if states.n:
                logsum = torch.logsumexp(scores, dim=0)
                p_assign = 1.0 / (1.0 + torch.exp(create_logit - logsum))
            else:
                p_assign = torch.zeros((), device=dev)
            pred = int(torch.argmax(logits))
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is not None:
                    sem_action[i] = "new"
                    sem_sid[i] = str(100000 + slot)
                    sem_slot[i] = str(slot)
                else:
                    slot = int(torch.argmax(scores))
                    prov = int(states.provenance[slot])
                    states.assign(slot, h, w)
                    sem_action[i] = "known" if prov == 0 else "existing"
                    sem_sid[i] = (str(known_ids[slot]) if prov == 0
                                  else str(100000 + slot))
                    sem_slot[i] = str(slot)
            else:
                slot = pred
                prov = int(states.provenance[slot])
                states.assign(slot, h, w)
                sem_action[i] = "known" if prov == 0 else "existing"
                sem_sid[i] = (str(known_ids[slot]) if prov == 0
                              else str(100000 + slot))
                sem_slot[i] = str(slot)
            sem_kscore[i] = f"{float(p_assign):.6f}"

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        # Phase14C intentionally supplies a physical-only proposal stream;
        # older Q1 CSVs already carried these columns.  Add every semantic
        # output field exactly once so the frozen replay remains compatible
        # with both input schemas.
        semantic_fields = ["sem_action", "sem_sid", "sem_kscore", "sem_slot"]
        out_fields = fieldnames + [x for x in semantic_fields if x not in fieldnames]
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out["sem_action"] = sem_action[i]
            out["sem_sid"] = sem_sid[i]
            out["sem_kscore"] = sem_kscore[i]
            out["sem_slot"] = sem_slot[i]
            writer.writerow(out)
    print(Counter(sem_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
