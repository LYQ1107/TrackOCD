"""Read-only diagnosis of dual-space evidence on the Q1 dev stream.

For every proposal of every aligned/GT track we record the 15-dim evidence
and the router known-probability, then summarize by GT role and step.
No metric tuning; dev GT is used only for analysis.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import Q1_DEV, group_tracks, load_proposals
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    build_known_protos,
    load_known_branch,
    load_novel_branch,
)
from src.iclr27_phase4v.train_router import MLPRouter


def main():
    device = "cuda:0"
    ktsr, kcls = load_known_branch(device)
    ntsr, l2 = load_novel_branch(device)
    protos = build_known_protos(ktsr, device).to(device)
    rc = torch.load(ROOT / "outputs/iclr27_phase4v/router_pilot/router_mlp_masked/router.pth",
                    map_location=device)
    router = MLPRouter(int(rc.get("dim", 19))).to(device)
    router.load_state_dict(rc["model"])
    router.eval()

    rows = load_proposals(Path(Q1_DEV))
    arr = np.load(ROOT / "outputs/iclr27_phase4s/q1_features/feats.npz")["feats"]
    assert len(arr) == len(rows)
    # causal qphys per row (same as dev_eval)
    by_track = defaultdict(list)
    for r in rows:
        by_track[(r["video_id"], r["track_id"])].append(r)
    qmap = {}
    for key, idxs in by_track.items():
        idxs.sort(key=lambda r: (r["frame_id"], int(r.get("proposal_local_id") or 0)))
        last_frame, hits, ssum, n = None, 0, 0.0, 0
        for r in idxs:
            gap = 0 if last_frame is None else r["frame_id"] - last_frame - 1
            b = json.loads(r["bbox_xyxy"])
            area = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
            qmap[id(r)] = [r["score"], float(np.log1p(hits)), min(hits, 16) / 16.0,
                           float(np.log1p(max(gap, 0))),
                           ssum / n if n else r["score"],
                           float(np.log(area) / 12.0)]
            last_frame = r["frame_id"]
            hits += 1
            ssum += r["score"]
            n += 1
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)

    memory = NovelMemory(device)
    rows_out = []
    with torch.no_grad():
        for key in sorted(tracks):
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
            t_commit = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                ft = torch.from_numpy(f).unsqueeze(0).to(device)
                qt = torch.tensor([qmap[id(r)]], device=device)
                rs = float(r_scalar[id(r)])
                ev, s_k, s_n, nl, l2_new = ds.step(ft, qt, rs, t + 1, memory)
                rp = torch.softmax(router(torch.from_numpy(ev).unsqueeze(0).to(device))[0],
                                   dim=-1)[1].item()
                sims = torch.nn.functional.normalize(s_k, dim=-1) @ protos.t()
                sims = sims[0]
                ps = torch.softmax(sims / 0.1, dim=-1)
                top2s = torch.topk(ps, k=2, dim=-1).values
                ent_s = -(ps * torch.log(ps + 1e-9)).sum(-1)
                energy_s = torch.logsumexp(sims / 0.1, dim=-1)
                rows_out.append({
                    "role": r["gt_role"], "cat": int(r["gt_category_id"]),
                    "t": t, "rp": rp,
                    "known_top1": float(ev[4]), "known_margin": float(ev[5]),
                    "known_entropy": float(ev[6]), "known_energy": float(ev[7]),
                    "novel_max": float(ev[8]), "novel_margin": float(ev[9]),
                    "new_logit": float(ev[10]), "log1pK": float(ev[11]),
                    "disagree": float(ev[12]), "q_score": float(ev[13]),
                    "q_hits": float(ev[14]), "q_age": float(ev[15]),
                    "q_gap": float(ev[16]), "q_run": float(ev[17]),
                    "q_area": float(ev[18]),
                    "proto_top1": float(ps.max().item()),
                    "proto_margin": float((top2s[0] - top2s[1]).item()),
                    "proto_entropy": float(ent_s.item()),
                    "proto_energy": float(energy_s.item()),
                })
                if t_commit is None:
                    t_commit = 1 if rp > 0.5 else 0

    arr_rows = np.array([[x[k] for k in ("rp", "known_top1", "known_margin",
                                         "known_entropy", "known_energy",
                                         "novel_max", "novel_margin", "new_logit",
                                         "log1pK", "disagree", "q_score",
                                         "q_hits", "q_age", "q_gap", "q_run",
                                         "q_area", "proto_top1",
                                         "proto_margin", "proto_entropy",
                                         "proto_energy")] for x in rows_out])
    roles = np.array([x["role"] for x in rows_out])
    ts = np.array([x["t"] for x in rows_out])
    names = ["rp", "known_top1", "known_margin", "known_entropy",
             "known_energy", "novel_max", "novel_margin", "new_logit",
             "log1pK", "disagree", "q_score", "q_hits", "q_age",
             "q_gap", "q_run", "q_area"]
    from sklearn.metrics import roc_auc_score
    # AUROC per step for the strongest features (known full-48 evidence)
    auc_by_step = {}
    for step in range(5):
        m = ts == step
        mkn = m & (roles == "known")
        mno = m & (roles == "novel")
        if mkn.sum() < 5 or mno.sum() < 5:
            continue
        y = np.concatenate([np.zeros(int(mkn.sum())), np.ones(int(mno.sum()))])
        feats = {
            "rp": arr_rows[m, 0],
            "known_top1": arr_rows[m, 1],
            "known_entropy": -arr_rows[m, 3],
            "known_energy": -arr_rows[m, 4],
            "new_logit": -arr_rows[m, 7],
            "q_score": -arr_rows[m, 10],
            "q_run": -arr_rows[m, 14],
            "proto_top1": arr_rows[m, 16],
            "proto_margin": arr_rows[m, 17],
            "proto_entropy": -arr_rows[m, 18],
            "proto_energy": -arr_rows[m, 19],
        }
        auc_by_step[str(step)] = {
            k: round(float(roc_auc_score(y, np.concatenate([v[mkn[m]], v[mno[m]]]))), 4)
            for k, v in feats.items()
        }
        auc_by_step[str(step)]["n_known"] = int(mkn.sum())
        auc_by_step[str(step)]["n_novel"] = int(mno.sum())
    # means at steps 0 and 2
    summary = {}
    for role in ("known", "novel"):
        summary[role] = {}
        for step in (0, 1, 2):
            m = (roles == role) & (ts == step)
            if m.sum() == 0:
                continue
            summary[role][f"t{step}"] = {
                "rp": round(float(arr_rows[m, 0].mean()), 4),
                "known_top1": round(float(arr_rows[m, 1].mean()), 4),
                "known_entropy": round(float(arr_rows[m, 3].mean()), 4),
                "known_energy": round(float(arr_rows[m, 4].mean()), 4),
                "new_logit": round(float(arr_rows[m, 7].mean()), 4),
            }
    summary["auroc_by_step"] = auc_by_step
    out = ROOT / "outputs/iclr27_phase4v/evidence_audit/dev_evidence_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
