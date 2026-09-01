"""Q1 replay with a Phase-13 representation checkpoint only.

The Phase-8A B semantic decision/state process is copied in structure from the
earlier replays.  No threshold, memory, reliability, or assign/create change
is introduced here.
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
from src.iclr27_phase13.model.track_encoder import TrackSemanticEncoder  # noqa: E402
from src.iclr27_phase8a.model.adapter import TorchSemanticStateSet  # noqa: E402
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402


def parse_box(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(json.loads(value), dtype=np.float32)


def motion_step(box, prev):
    if prev is None:
        return np.zeros(4, dtype=np.float32)
    b = np.asarray(box, dtype=np.float32)
    p = np.asarray(prev, dtype=np.float32)
    cx, cy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    pcx, pcy = (p[0] + p[2]) * 0.5, (p[1] + p[3]) * 0.5
    w, h = max(float(b[2] - b[0]), 1.0), max(float(b[3] - b[1]), 1.0)
    pw, ph = max(float(p[2] - p[0]), 1.0), max(float(p[3] - p[1]), 1.0)
    scale = max(float(np.sqrt(pw * ph)), 1.0)
    out = np.asarray([(cx - pcx) / scale, (cy - pcy) / scale,
                      np.log(w / pw), np.log(h / ph)], dtype=np.float32)
    return np.clip(out, -8.0, 8.0)


def load_rows(path):
    with (ROOT / path).open(newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd], list(rd.fieldnames or [])


def chrono(rows):
    return sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))


def embed_known(model, dataset, device):
    z = np.load(ROOT / dataset)
    appearance = z["appearance"].astype(np.float32)
    motion = z["motion"].astype(np.float32)
    mask = z["mask"].astype(np.uint8)
    labels = z["labels"].astype(np.int64)
    out = []
    with torch.no_grad():
        for start in range(0, len(appearance), 128):
            h, _ = model(torch.from_numpy(appearance[start:start + 128]).to(device),
                         torch.from_numpy(motion[start:start + 128]).to(device),
                         torch.from_numpy(mask[start:start + 128]).to(device))
            out.append(h.cpu().numpy().astype(np.float32))
    h = np.concatenate(out, axis=0)
    ids = np.asarray(sorted(set(int(c) for c in labels)), dtype=np.int64)
    mu, count = [], []
    for c in ids:
        v = h[labels == c].mean(0)
        mu.append(v / max(float(np.linalg.norm(v)), 1e-12))
        count.append(float(np.sum(labels == c)))
    return np.asarray(mu, dtype=np.float32), np.asarray(count, dtype=np.float32), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--proposals", default="outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv")
    ap.add_argument("--feats", default="outputs/iclr27_phase6b/q1/final_dsct/feats.npz")
    ap.add_argument("--dataset", default="outputs/iclr27_phase13/dataset/real_tao_tracks.npz")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = TrackSemanticEncoder(appearance_dim=768, motion_dim=4,
                                hidden=int(a.get("hidden", 128)), out_dim=128).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    create = CreateHead(dim=128).to(device)
    bck = torch.load(ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth",
                     map_location=device, weights_only=False)
    create.load_state_dict(bck["create_head"]); create.eval()
    temp = float(bck.get("temp", 20.0))
    mu, counts, known_ids = embed_known(model, args.dataset, device)

    rows, fields = load_rows(args.proposals)
    raw = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"row/feature mismatch: {len(rows)} vs {len(raw)}")
    order = chrono(rows)
    states = TorchSemanticStateSet(dim=128, max_slots=4096, sigma2=1.0,
                                   score_mode="cosine", cosine_temp=temp).to(device)
    states.init_known(torch.from_numpy(mu).to(device), torch.from_numpy(counts).to(device))
    track_state, track_age, previous_box = {}, defaultdict(int), {}
    actions, sids, kscores, slots = [""] * len(rows), [""] * len(rows), [""] * len(rows), [""] * len(rows)
    with torch.no_grad():
        for i in order:
            row = rows[i]
            key = (int(row["video_id"]), int(row["track_id"]))
            box = parse_box(row["bbox_xyxy"])
            mot = motion_step(box, previous_box.get(key))
            previous_box[key] = box
            st = track_state.get(key)
            if st is None:
                st = model.new_state(device=device)
            h, st = model.step(torch.from_numpy(raw[i]).to(device).unsqueeze(0),
                               torch.from_numpy(mot).to(device).unsqueeze(0), st)
            h = h[0]; track_state[key] = st
            track_age[key] += 1; age = track_age[key]; w = float(age)
            existing = states.log_scores(h, w)
            best = existing.max() if states.n else torch.zeros((), device=device)
            pv = phys_vec(row.get("score", 0.0), row.get("prior_hits", 0.0), age, device)
            create_logit = temp * create(h, pv, best)
            logits = states.logits(h, w, create_logit.reshape(1))
            pred = int(torch.argmax(logits))
            p_assign = (1.0 / (1.0 + torch.exp(create_logit - torch.logsumexp(existing, dim=0)))
                        if states.n else torch.zeros((), device=device))
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is None:
                    slot = int(torch.argmax(existing)); states.assign(slot, h, w)
                    action = "known" if int(states.provenance[slot]) == 0 else "existing"
                else:
                    action = "new"
            else:
                slot = pred; states.assign(slot, h, w)
                action = "known" if int(states.provenance[slot]) == 0 else "existing"
            actions[i] = action
            sids[i] = str(int(known_ids[slot])) if action == "known" else str(100000 + int(slot))
            kscores[i] = f"{float(p_assign):.6f}"; slots[i] = str(int(slot))

    for name in ("sem_kscore", "sem_slot"):
        if name not in fields: fields.append(name)
    out = ROOT / args.out_csv; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for i, row in enumerate(rows):
            d = dict(row); d.update({"sem_action": actions[i], "sem_sid": sids[i],
                                     "sem_kscore": kscores[i], "sem_slot": slots[i]})
            wr.writerow(d)
    os.replace(tmp, out)
    meta = {
        "representation": "TrackSemanticEncoder(DINOv2 crop trajectory + causal box motion)",
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "semantic_decision": "Phase8A B frozen CreateHead + TorchSemanticStateSet",
        "q1_labels_used": False, "future_used": False,
        "physical_id_used_for_semantic_id": False, "rows": len(rows),
        "action_counts": dict(Counter(actions)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
