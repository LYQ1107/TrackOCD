"""Train-side single-frame gate calibration for Phase 4J (J1).

The frozen M2 gate was trained on track-mean features plus synthetic novel
paths.  The frame-online deployment evaluates the gate on single frames
(track_len=1).  This script measures the gate operating point on the
train-side feature cache (known frames) and on synthetic novel frames from
the exact Phase 4F/MDC generator, then selects:

  C0: original threshold 0.5 (reference);
  C1: one global calibrated threshold;
  C2: two-band (early age <= k, stable age > k) threshold.

Only train-side data is used for selection.  The 20-video subset is never
used to choose a threshold (official-tuning ban).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit.evaluate import embed_track
from src.orbit_msr.protocol import known_stats
from src.orbit_msr.train import make_synthetic_tracks


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def embed_single(model, feat, device):
    x = torch.as_tensor(feat, dtype=torch.float32, device=device).view(1, 1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model.aggregate(x, mask)
    z = out["z"][0].cpu().numpy().astype(np.float32)
    rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
    return _norm(z), rel


def build_known(model, device):
    labels = load_train_labels()
    feats = load_frame_features("train_known_mean")
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in feats:
            by_class[int(c)].append(sid)
    protos, radii = {}, {}
    for c, ids in by_class.items():
        zs = []
        for sid in ids:
            z, _ = embed_track(model, feats[sid], device)
            zs.append(z)
        Z = np.stack(zs)
        p = _norm(Z.mean(axis=0).astype(np.float32))
        protos[int(c)] = p
        cos = Z @ p
        radii[int(c)] = float(np.percentile(1.0 - cos, 50).clip(min=0.02))
    return protos, radii


def gate_p(model, gs, device):
    with torch.no_grad():
        logit = float(model.gate_forward(
            torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
    return float(torch.sigmoid(torch.as_tensor(logit)))


def novel_stats_from_memory(z, pad_zs):
    """best_n/second_n/margin_n/dist_n against a pad novel memory."""
    if not len(pad_zs):
        return -1.0, -1.0, 0.0, 1.0
    ns = np.asarray(pad_zs, dtype=np.float32) @ z
    order = np.argsort(ns)[::-1]
    best_n = float(ns[order[0]])
    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
    margin_n = best_n - second_n
    dist_n = (1.0 - best_n) / 0.3
    return best_n, second_n, margin_n, dist_n


def collect(model, protos, radii, device, seed=1027, n_pad=24):
    """Return per-frame rows: (age, role, p_known) for two scenarios."""
    labels = load_train_labels()
    feats = load_frame_features("train_known_mean")
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    rng = np.random.RandomState(seed)

    # --- known single-frame observations (train-side only) ---
    known_rows = []
    known_zs = []
    for sid, c in labels.items():
        if sid not in feats:
            continue
        frames = feats[sid]
        for m, f in enumerate(frames, start=1):
            z, rel = embed_single(model, f, device)
            known_zs.append(z)
            known_rows.append({"age": m, "role": "known", "z": z,
                               "rel": rel, "class": int(c)})

    # --- synthetic novel single-frame observations (same generator) ---
    novel_rows = []
    syn_pool = sorted(feats)
    rng.shuffle(syn_pool)
    for k, sid in enumerate(syn_pool[:12]):
        base_z, _ = embed_track(model, feats[sid], device)
        tracks = make_synthetic_tracks(rng, base_z, n_tracks=4,
                                       alpha_range=(0.35, 0.65), sigma=0.12)
        for j, frames in enumerate(tracks):
            for m, f in enumerate(frames, start=1):
                z, rel = embed_single(model, f, device)
                novel_rows.append({"age": m, "role": "novel", "z": z,
                                   "rel": rel, "class": 1000000 + k})

    # pad novel memories for the memory-scale scenario (train-side only)
    pad_zs_all = np.stack(known_zs) if known_zs else \
        np.empty((0, 768), dtype=np.float32)
    pads = {}
    rng2 = np.random.RandomState(seed + 7)
    for n_novel in (0, 50, 150, 300):
        idx = rng2.choice(len(pad_zs_all), size=n_pad, replace=False) \
            if len(pad_zs_all) else []
        pads[n_novel] = [pad_zs_all[i] for i in idx]

    def gate_features(row, n_novel):
        if n_novel == 0:
            best_n = second_n = -1.0
            margin_n, dist_n = 0.0, 1.0
        else:
            best_n, second_n, margin_n, dist_n = novel_stats_from_memory(
                row["z"], pads[n_novel])
        return known_stats(row["z"], P_known, radii, known_ids=known_ids,
                           best_n=best_n, second_n=second_n,
                           margin_n=margin_n, dist_n=dist_n,
                           rel=row["rel"], track_len=1, n_novel=n_novel,
                           include_anchor=False)

    out = {"clean": [], "memory": []}
    for row in known_rows + novel_rows:
        out["clean"].append({
            "age": row["age"], "role": row["role"],
            "p_known": gate_p(model, gate_features(row, 0), device)})
        for n_novel in (0, 50, 150, 300):
            out["memory"].append({
                "age": row["age"], "role": row["role"],
                "n_novel": n_novel,
                "p_known": gate_p(model, gate_features(row, n_novel),
                                  device)})
    return out


def metrics(rows, thr_early, thr_stable=None, split_age=None):
    k2n = n2k = route = matched = known_n = novel_n = 0
    for r in rows:
        thr = thr_early
        if thr_stable is not None and split_age is not None and \
                r["age"] > split_age:
            thr = thr_stable
        pred = "known" if r["p_known"] >= thr else "novel"
        if pred == r["role"]:
            route += 1
        if r["role"] == "known":
            known_n += 1
            if pred == "novel":
                k2n += 1
        else:
            novel_n += 1
            if pred == "known":
                n2k += 1
        matched += 1
    return {
        "n": matched,
        "routing_accuracy": route / max(matched, 1),
        "k2n": k2n / max(known_n, 1),
        "n2k": n2k / max(novel_n, 1),
    }


def pick_operating_point(scan_rows, label):
    """Feasible: routing >= 0.50 and N2K <= 0.35; minimize K2N then N2K."""
    feasible = [r for r in scan_rows
                if r["routing_accuracy"] >= 0.50 and r["n2k"] <= 0.35]
    if not feasible:
        return None
    return min(feasible, key=lambda r: (r["k2n"], r["n2k"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "iclr27_phase4j" /
                    "calibration")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    protos, radii = build_known(model, device)
    rows_by_scen = collect(model, protos, radii, device)

    thrs = [round(0.05 * i, 2) for i in range(1, 20)] + [0.5]
    thrs = sorted(set(thrs))
    scan_clean, scan_memory = [], []
    for thr in thrs:
        m = metrics(rows_by_scen["clean"], thr)
        scan_clean.append({"threshold": thr, "config": "C1",
                           "split_age": "", **m})
        m = metrics(rows_by_scen["memory"], thr)
        scan_memory.append({"threshold": thr, "config": "C1",
                            "split_age": "", **m})

    # C2: two-band (early age <= k, stable age > k), k in {2,3,4}
    for k in (2, 3, 4):
        for t_e in thrs:
            for t_s in thrs:
                m = metrics(rows_by_scen["clean"], t_e, t_s, k)
                scan_clean.append({"threshold": f"{t_e}/{t_s}",
                                   "config": f"C2_k{k}",
                                   "split_age": k, **m})
                m = metrics(rows_by_scen["memory"], t_e, t_s, k)
                scan_memory.append({"threshold": f"{t_e}/{t_s}",
                                    "config": f"C2_k{k}",
                                    "split_age": k, **m})

    write_csv(args.out_dir / "threshold_scan_clean.csv", scan_clean)
    write_csv(args.out_dir / "threshold_scan_memory.csv", scan_memory)

    # distributions by age
    dist_rows = []
    for scen, rows in rows_by_scen.items():
        for role in ("known", "novel"):
            for age in sorted({r["age"] for r in rows}):
                sel = [r["p_known"] for r in rows
                       if r["role"] == role and r["age"] == age]
                if not sel:
                    continue
                dist_rows.append({
                    "scenario": scen, "role": role, "age": age,
                    "n": len(sel), "mean": float(np.mean(sel)),
                    "p25": float(np.percentile(sel, 25)),
                    "p50": float(np.percentile(sel, 50)),
                    "p75": float(np.percentile(sel, 75)),
                })
    write_csv(args.out_dir / "train_side_pknown_by_age.csv", dist_rows)

    c0_clean = next(r for r in scan_clean
                    if r["config"] == "C1" and r["threshold"] == 0.5)
    c0_memory = next(r for r in scan_memory
                     if r["config"] == "C1" and r["threshold"] == 0.5)
    c1_clean = pick_operating_point(
        [r for r in scan_clean if r["config"] == "C1"], "clean")
    c1_memory = pick_operating_point(
        [r for r in scan_memory if r["config"] == "C1"], "memory")
    c2_clean = None
    c2_memory = None
    for k in (2, 3, 4):
        cand_c = pick_operating_point(
            [r for r in scan_clean if r["config"] == f"C2_k{k}"], "clean")
        cand_m = pick_operating_point(
            [r for r in scan_memory if r["config"] == f"C2_k{k}"], "memory")
        if cand_c and (c2_clean is None or
                       cand_c["k2n"] + cand_c["n2k"] <
                       c2_clean["k2n"] + c2_clean["n2k"]):
            c2_clean = dict(cand_c)
            c2_clean["split_age"] = k
        if cand_m and (c2_memory is None or
                       cand_m["k2n"] + cand_m["n2k"] <
                       c2_memory["k2n"] + c2_memory["n2k"]):
            c2_memory = dict(cand_m)
            c2_memory["split_age"] = k

    def summary(c0, c1, c2, scen):
        use_c1 = c1 is not None and (
            c1["k2n"] + c1["n2k"] + 0.005 < c0["k2n"] + c0["n2k"])
        use_c2 = (c2 is not None and
                  (c2["k2n"] + c2["n2k"] + 0.01 <
                   (c1["k2n"] + c1["n2k"] if c1 is not None else 9.9)))
        chosen = "C2" if use_c2 else ("C1" if use_c1 else "C0")
        return {
            "scenario": scen,
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "chosen": chosen,
        }

    out = {
        "selection_rule": ("feasible routing>=0.50 and N2K<=0.35; "
                           "minimize (K2N, N2K); C2 used only if "
                           "K2N+N2K improves C1 by >=0.01"),
        "clean": summary(c0_clean, c1_clean, c2_clean, "clean"),
        "memory": summary(c0_memory, c1_memory, c2_memory, "memory"),
    }
    (args.out_dir / "calibration_config.json").write_text(
        json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
