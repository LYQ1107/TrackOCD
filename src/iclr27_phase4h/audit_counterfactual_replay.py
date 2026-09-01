"""Phase 4H counterfactual memory replay.

Run P0 once with the frozen M2 model, snapshot the causal novel memory at
the first arrival where memory size reaches >= 32 / 128 / 256 / 400, then
replay every official novel query against each snapshot WITHOUT mutating
memory.  This isolates the effect of memory state on the frozen gate for a
fixed query and fixed visual evidence.

Offline audit only; never used to train or select a method.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import build_known
from src.orbit_msr.evaluate import embed_many
from src.orbit_msr.protocol import known_stats
from src.iclr27_phase4h.audit_permutations import (
    load_mdc_4h,
    prepare_official,
    replay_order,
)


class Snapshotter:
    def __init__(self, thresholds=(32, 128, 256, 400)):
        self.thresholds = sorted(thresholds)
        self.taken = set()
        self.snapshots = []

    def __call__(self, i, mem_size, mem):
        for t in self.thresholds:
            if t not in self.taken and mem_size >= t:
                self.taken.add(t)
                self.snapshots.append({
                    "label": f"mem{t}", "threshold": t,
                    "arrival": i, "memory_size": mem_size,
                    "mem": copy.deepcopy(mem),
                })


def gate_at_snapshot(model, ck, z, rel, snap, protos, radii, known_ids,
                     device, track_len):
    P_known = np.stack([protos[c] for c in known_ids]).astype(np.float32)
    mem = snap["mem"]
    P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
               .astype(np.float32)) if mem.novel else np.empty(
        (0, 768), dtype=np.float32)
    best_n = second_n = -1.0
    margin_n = 0.0
    dist_n = 1.0
    if P_novel.shape[0]:
        ns = P_novel @ z
        best_n = float(ns.max())
        order = np.argsort(ns)[::-1]
        second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
        margin_n = best_n - second_n
        nid = int(sorted(mem.novel)[int(order[0])])
        r_n = mem.novel_radii.get(nid, 0.3)
        dist_n = (1.0 - best_n) / max(r_n, 1e-6)
    gs = known_stats(z, P_known, radii, known_ids=known_ids,
                     best_n=best_n, second_n=second_n, margin_n=margin_n,
                     dist_n=dist_n, rel=rel, track_len=track_len,
                     n_novel=len(mem.novel), include_anchor=False)
    with torch.no_grad():
        gate_logit = float(model.gate_forward(
            torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
    gate_prob = float(torch.sigmoid(torch.as_tensor(gate_logit)))
    supports = [mem.support(v) for v in sorted(mem.novel)]
    return {
        "gate_prob": gate_prob,
        "action": "KNOWN" if gate_prob >= 0.5 else "NON_KNOWN",
        "best_known_sim": float(gs[0]),
        "known_margin": float(gs[2]),
        "best_novel_sim": best_n,
        "novel_margin": margin_n,
        "memory_size": len(mem.novel),
        "mean_support": float(np.mean(supports)) if supports else 0.0,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    device = "cuda"
    rows, gt, feats, train_feats, labels = prepare_official()
    model, ck = load_mdc_4h("runs/orbit_mdc/mdc_m2/model.pth", device)
    known_classes = sorted(set(labels.values()))
    protos, radii = build_known(model, train_feats, labels,
                                set(known_classes), device)
    known_ids = sorted(protos)
    sids = [r["sample_id"] for r in rows]
    zs, rels = embed_many(model, feats, sids, device)
    track_lens = {sid: len(f) for sid, f in feats.items()}

    snap = Snapshotter()
    logs, mem = replay_order(model, ck, rows, zs, rels, protos, radii,
                             known_ids, device, snapshot_cb=snap,
                             track_lens=track_lens)
    print("snapshots:", [(s["label"], s["arrival"], s["memory_size"])
                         for s in snap.snapshots], flush=True)
    by_sid = {l["sample_id"]: l for l in logs}
    hardness = {}
    novel_rows = [r for r in rows if r["role"] == "novel"]
    out = []
    for r in novel_rows:
        sid = r["sample_id"]
        actual = by_sid[sid]
        for s in snap.snapshots:
            g = gate_at_snapshot(model, ck, zs[sid], rels[sid], s, protos,
                                 radii, known_ids, device, track_lens[sid])
            out.append({
                "sample_id": sid, "true_class": r["class"],
                "arrival_index": actual["arrival_index"],
                "snapshot": s["label"], "snapshot_arrival": s["arrival"],
                "snapshot_memory_size": g["memory_size"],
                "snapshot_mean_support": round(g["mean_support"], 4),
                "gate_prob": round(g["gate_prob"], 6),
                "predicted_action": g["action"],
                "best_known_sim": round(g["best_known_sim"], 6),
                "known_margin": round(g["known_margin"], 6),
                "best_novel_sim": round(g["best_novel_sim"], 6),
                "novel_margin": round(g["novel_margin"], 6),
                "actual_gate_prob": round(actual["gate_prob"], 6),
                "actual_action": actual["predicted_action"],
                "delta_gate_vs_actual": round(g["gate_prob"] - actual["gate_prob"], 6),
            })
    out_dir = ROOT / "outputs/iclr27_phase4h/audit"
    write_csv(out_dir / "counterfactual_memory_replay.csv", out)
    print("saved", out_dir / "counterfactual_memory_replay.csv", "rows", len(out))


if __name__ == "__main__":
    main()
