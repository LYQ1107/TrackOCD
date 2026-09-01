"""Strict-causal per-frame evaluation of unified assign-or-create on Q1/Q2.

Every proposal row (including FP) is processed in official chronological
order. Every frame emits an immediate action from {KNOWN(c),
EXISTING_NOVEL(k), NEW_NOVEL}; actions are immutable. Aligned GT rows are
used only for evaluation, never for decisions or training.

Legacy TrackOCD metrics are produced from both the first-frame and the
last-frame action per aligned physical track (first-frame is the strictest
and the direct O1c analogue).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.downstream.dev_eval import qphys_from_rows
from src.iclr27_phase4u.downstream.model import build_tsr_known_protos
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase5a.assign_create.creation_head import (
    CreationHead,
    head_action,
)
from src.iclr27_phase5a.assign_create.memory import CategoryMemory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def load_model(device, proto_dir: Path, embed: str):
    p = np.load(proto_dir / "protos.npz")
    known_list = [int(c) for c in p["known_list"]]
    if embed == "h":
        rep = TSR(arch="gru").to(device)
        ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                        map_location=device)
        tsr_sd = {k[len("rep."):]: v for k, v in ck["model"].items()
                  if k.startswith("rep.")}
        rep.load_state_dict(tsr_sd)
        rep.eval()
        protos = build_tsr_known_protos(rep, device).cpu().numpy().astype(np.float32)
    else:
        rep = None
        fp = np.load(proto_dir / "frame_protos.npz")
        protos = np.asarray(fp["protos"], dtype=np.float32)
    return rep, protos, known_list


def precompute_states(rows, feats, qmap, rep, device, embed):
    by_key = {}
    for i, r in enumerate(rows):
        by_key.setdefault((int(r["video_id"]), int(r["track_id"])), []).append(i)
    states = {}
    if embed == "h":
        with torch.no_grad():
            for key, idxs in by_key.items():
                tr = [rows[i] for i in idxs]
                ft = np.stack([feats[i] for i in idxs]).astype(np.float32)
                qt = np.stack([qmap[id(r)] for r in tr]).astype(np.float32)
                ss = rep.embed_sequence(
                    torch.from_numpy(ft).to(device),
                    torch.from_numpy(qt).to(device)).cpu().numpy()
                states[key] = ss
    else:
        for key, idxs in by_key.items():
            states[key] = np.stack([feats[i] for i in idxs]).astype(np.float32)
    return states


def replay(rows, feats, qmap, states, protos, known_list, device,
           mode="threshold", tau=0.75, ema_alpha=0.1, update_threshold=None,
           head=None, update_novel=True, min_birth_age=1, min_birth_score=0.0,
           min_birth_prior=0):
    proto_index = {int(c): i for i, c in enumerate(known_list)}
    mem = CategoryMemory(torch.from_numpy(protos), known_list,
                         ema_alpha=ema_alpha, device=device)
    ptr = defaultdict(int)
    records = []
    chrono = sorted(rows, key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                                         int(r.get("proposal_local_id") or 0),
                                         int(r["track_id"])))
    for r in chrono:
        key = (int(r["video_id"]), int(r["track_id"]))
        st = states.get(key)
        if st is None:
            continue
        i = ptr[key]
        ptr[key] += 1
        if i >= len(st):
            continue
        if mode == "jointcsv":
            a = r.get("sem_action") or "unresolved"
            sid = int(r["sem_sid"]) if r.get("sem_sid") not in ("", None) else None
            records.append({
                "key": list(key), "frame_id": int(r["frame_id"]),
                "row_id": id(r), "action": a, "sid": sid, "age": float(i + 1),
            })
            continue
        h = torch.from_numpy(st[i]).to(device)
        age = float(i + 1)
        if mode == "head":
            a, sid, _ = head_action(head, h, mem, age)
        else:
            a, sid, _ = mem.step(h, tau, key, update_novel=update_novel,
                                 update_threshold=update_threshold,
                                 allow_birth=(
                                     age >= min_birth_age
                                     and float(r["score"]) >= min_birth_score
                                     and int(r["prior_hits"]) >= min_birth_prior))
        if mode == "head":
            allow_birth = (age >= min_birth_age
                           and float(r["score"]) >= min_birth_score
                           and int(r["prior_hits"]) >= min_birth_prior)
            if a == "new" and not allow_birth:
                # force immediate assign to the best existing state
                hh = torch.nn.functional.normalize(h.reshape(1, -1), dim=-1)
                sims = mem.similarities(hh)[0]
                k0 = mem.known_protos.shape[0]
                idx = int(sims.argmax().item())
                if idx < k0:
                    a, sid = "known", mem.known_ids[idx]
                else:
                    a, sid = "existing", mem.novel_ids[idx - k0]
        records.append({
            "key": list(key), "frame_id": int(r["frame_id"]),
            "row_id": id(r), "action": a, "sid": sid, "age": age,
        })
        if mode == "head":
            if a == "new":
                slot = mem.size
                mem.novel_protos = torch.cat(
                    [mem.novel_protos, torch.nn.functional.normalize(
                        h.reshape(1, -1), dim=-1)], dim=0)
                mem.novel_ids.append(slot)
                mem.novel_birth_key[slot] = key
            elif a == "existing" and update_novel:
                slot = mem.novel_ids.index(sid)
                p0 = mem.novel_protos[slot]
                mem.novel_protos[slot] = torch.nn.functional.normalize(
                    (1 - ema_alpha) * p0 + ema_alpha * h, dim=-1)
        elif mode == "threshold" and not update_novel and a == "existing":
            # static control: no post-birth prototype update
            pass
    return records


def strict_metrics(records_by_row, aligned_rows, labels, mapping,
                   n_born_global=0):
    """records_by_row: dict row_id -> record for aligned evaluation."""
    aligned = []
    for r in aligned_rows:
        rec = records_by_row.get(id(r))
        if rec is None:
            continue
        key = tuple(rec["key"])
        sid = mapping[key]
        lab = labels[sid]
        aligned.append({
            "row": r, "rec": rec, "sid": sid,
            "role": lab["protocol_role"], "cat": int(lab["ground_truth_category_id"]),
            "is_novel": lab["protocol_role"] == "novel",
            "is_known": lab["protocol_role"] in ("supported_known", "zero_shot_known"),
        })
    aligned.sort(key=lambda a: (a["row"]["video_id"], a["row"]["frame_id"]))
    known = [a for a in aligned if a["is_known"]]
    novel = [a for a in aligned if a["is_novel"]]
    # first occurrence per GT novel category in the chronological stream
    first_seen = {}
    for a in novel:
        first_seen.setdefault(a["cat"], a)
    first_occ = list(first_seen.values())
    reuse = [a for a in novel if a is not first_seen[a["cat"]]]

    # slot -> gt cat from first NEW action on aligned novel occurrences
    slot_cat = {}
    slot_birth_key = {}
    for a in novel:
        if a["rec"]["action"] == "new":
            slot_cat.setdefault(a["rec"]["sid"], a["cat"])
            slot_birth_key.setdefault(a["rec"]["sid"], tuple(a["rec"]["key"]))

    def correct_reuse(a):
        return (a["rec"]["action"] == "existing"
                and slot_cat.get(a["rec"]["sid"]) == a["cat"])

    known_correct = sum(1 for a in known
                        if a["rec"]["action"] == "known"
                        and a["rec"]["sid"] == a["cat"])
    first_new_ok = sum(1 for a in first_occ if a["rec"]["action"] == "new")
    reuse_ok = sum(1 for a in reuse if correct_reuse(a))
    cross = [a for a in reuse
             if slot_birth_key.get(a["rec"]["sid"]) != tuple(a["rec"]["key"])]
    cross_ok = sum(1 for a in cross if correct_reuse(a))
    known_to_new = sum(1 for a in known if a["rec"]["action"] == "new")
    reuse_to_new = sum(1 for a in reuse if a["rec"]["action"] == "new")
    known_to_existing = sum(1 for a in known if a["rec"]["action"] == "existing")
    new_on_novel = sum(1 for a in novel if a["rec"]["action"] == "new")
    new_on_aligned = sum(1 for a in aligned if a["rec"]["action"] == "new")

    # semantic switch rate within physical tracks (aligned + unaligned handled
    # outside; here aligned only)
    by_key = defaultdict(list)
    for a in aligned:
        by_key[tuple(a["rec"]["key"])].append(a)
    switches = adj = 0
    for tr in by_key.values():
        tr.sort(key=lambda a: a["rec"]["frame_id"])
        for x, y in zip(tr, tr[1:]):
            adj += 1
            if (x["rec"]["action"], x["rec"]["sid"]) != (y["rec"]["action"], y["rec"]["sid"]):
                switches += 1

    # latency to first correct birth per novel category
    first_correct = {}
    for occ_idx, a in enumerate(novel):
        c = a["cat"]
        if c not in first_correct and a["rec"]["action"] == "new":
            first_correct[c] = occ_idx
    latency_occ = [first_correct[c] for c in sorted(first_seen)
                   if c in first_correct]

    # fragmentation / count
    frag = defaultdict(set)
    for a in novel:
        if a["rec"]["action"] in ("new", "existing"):
            frag[a["cat"]].add(a["rec"]["sid"])
    n_born = len({a["rec"]["sid"] for a in aligned if a["rec"]["action"] == "new"})
    n_true_cats = len(first_seen)
    novel_slots = sorted({a["rec"]["sid"] for a in novel
                          if a["rec"]["action"] in ("new", "existing")})
    y = np.array([a["cat"] for a in novel], dtype=np.int64)
    p = np.array([a["rec"]["sid"] for a in novel], dtype=np.int64)
    if len(novel) > 1:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        nmi = float(normalized_mutual_info_score(y, p))
        ari = float(adjusted_rand_score(y, p))
    else:
        nmi = ari = 0.0

    # known forgetting: split known occurrences by chronological frame order
    known_sorted = sorted(known, key=lambda a: (a["row"]["video_id"], a["row"]["frame_id"]))
    mid = len(known_sorted) // 2
    h1 = known_sorted[:mid]
    h2 = known_sorted[mid:]
    known_first_half = sum(1 for a in h1
                           if a["rec"]["action"] == "known" and a["rec"]["sid"] == a["cat"])
    known_second_half = sum(1 for a in h2
                            if a["rec"]["action"] == "known" and a["rec"]["sid"] == a["cat"])

    return {
        "n_aligned_occurrences": len(aligned),
        "n_known_occurrences": len(known),
        "n_novel_occurrences": len(novel),
        "n_first_novel_occurrences": len(first_occ),
        "n_novel_reuse_occurrences": len(reuse),
        "known_occurrence_acc": known_correct / max(len(known), 1),
        "first_novel_birth_acc": first_new_ok / max(len(first_occ), 1),
        "novel_reuse_acc": reuse_ok / max(len(reuse), 1),
        "cross_physical_reuse_acc": cross_ok / max(len(cross), 1),
        "cross_physical_reuse_share": len(cross) / max(len(reuse), 1),
        "known_to_new_rate": known_to_new / max(len(known), 1),
        "known_to_existing_rate": known_to_existing / max(len(known), 1),
        "reuse_to_new_rate": reuse_to_new / max(len(reuse), 1),
        "new_precision_on_aligned": new_on_novel / max(new_on_aligned, 1),
        "semantic_switch_rate": switches / max(adj, 1),
        "first_correct_birth_median_occurrence": (
            float(np.median(latency_occ)) if latency_occ else None),
        "n_true_novel_categories": n_true_cats,
        "n_born_novel_states": n_born,
        "novel_count_abs_error": abs(n_born - n_true_cats),
        "mean_fragmentation": (
            float(np.mean([len(v) for v in frag.values()])) if frag else 0.0),
        "duplicate_creation_rate": (
            float(np.mean([len(v) > 1 for v in frag.values()])) if frag else 0.0),
        "novel_nmi": nmi,
        "novel_ari": ari,
        "known_acc_first_half_stream": known_first_half / max(len(h1), 1),
        "known_acc_second_half_stream": known_second_half / max(len(h2), 1),
        "known_forgetting_delta": (
            known_second_half / max(len(h2), 1) - known_first_half / max(len(h1), 1)),
        "n_born_novel_states_global": n_born_global,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--proto-dir", default="outputs/iclr27_phase5a/pilot/episodes")
    ap.add_argument("--embed", choices=["h", "f"], default="h")
    ap.add_argument("--mode", choices=["threshold", "head", "jointcsv"],
                    default="threshold")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--ema-alpha", type=float, default=0.1)
    ap.add_argument("--update-threshold", type=float, default=None)
    ap.add_argument("--head-checkpoint", default=None)
    ap.add_argument("--filter", choices=["all", "aligned"], default="all")
    ap.add_argument("--min-birth-age", type=float, default=1.0)
    ap.add_argument("--min-birth-score", type=float, default=0.0)
    ap.add_argument("--min-birth-prior", type=int, default=0)
    ap.add_argument("--no-update-novel", dest="update_novel",
                    action="store_false", default=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = load_proposals(Path(args.proposals))
    arr = np.load(ROOT / args.feats)["feats"]
    assert len(arr) == len(all_rows)
    rows = all_rows
    qmap = qphys_from_rows(rows)
    tracks = group_tracks(rows)
    stream, labels_all = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    if args.filter == "aligned":
        aligned_keys = set(mapping)
        keep = [i for i, r in enumerate(all_rows)
                if (int(r["video_id"]), int(r["track_id"])) in aligned_keys]
        rows = [all_rows[i] for i in keep]
        arr = arr[keep]
        tracks = group_tracks(rows)
        qmap = qphys_from_rows(rows)
    rep, protos, known_list = load_model(args.device, ROOT / args.proto_dir,
                                         args.embed)
    states = precompute_states(rows, arr, qmap, rep, args.device, args.embed)

    head = None
    if args.mode == "head":
        head = CreationHead()
        ck = torch.load(ROOT / args.head_checkpoint, map_location=args.device)
        head.load_state_dict(ck["state_dict"])
        head.to(args.device)
        head.eval()

    records = replay(rows, arr, qmap, states, protos, known_list, args.device,
                     mode=args.mode, tau=args.tau, ema_alpha=args.ema_alpha,
                     update_threshold=args.update_threshold, head=head,
                     update_novel=args.update_novel,
                     min_birth_age=args.min_birth_age,
                     min_birth_score=args.min_birth_score,
                     min_birth_prior=args.min_birth_prior)
    records_by_row = {r["row_id"]: r for r in records}

    # aligned rows: only rows of aligned physical tracks
    aligned_keys = set(mapping)
    aligned_rows = [r for r in rows if (int(r["video_id"]), int(r["track_id"]))
                    in aligned_keys]
    n_born_global = sum(1 for r in records if r["action"] == "new")
    sm = strict_metrics(records_by_row, aligned_rows, labels, mapping,
                        n_born_global=n_born_global)

    # legacy per-track predictions: first-frame and last-frame actions
    first_preds, last_preds = [], []
    order = 0
    by_track_records = defaultdict(list)
    for r in records:
        by_track_records[tuple(r["key"])].append(r)
    for key, sid in sorted(mapping.items()):
        order += 1
        recs = sorted(by_track_records.get(key, []), key=lambda r: r["frame_id"])
        if not recs:
            first_preds.append({"sample_id": sid, "prediction_type": "unresolved",
                                "stream_order": order})
            last_preds.append({"sample_id": sid, "prediction_type": "unresolved",
                               "stream_order": order})
            continue
        for target, out in ((recs[0], first_preds), (recs[-1], last_preds)):
            if target["action"] == "known":
                out.append({"sample_id": sid, "prediction_type": "known",
                            "semantic_category_id": target["sid"],
                            "stream_order": order})
            elif target["action"] in ("existing", "new"):
                out.append({"sample_id": sid, "prediction_type": "novel",
                            "virtual_category_id": target["sid"],
                            "stream_order": order})
            else:
                out.append({"sample_id": sid, "prediction_type": "unresolved",
                            "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    legacy_first = ev.evaluate(first_preds)
    legacy_last = ev.evaluate(last_preds)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "occurrences.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary = {
        "config": vars(args),
        "strict": sm,
        "legacy_first_frame": {k: (float(v) if isinstance(v, (int, float))
                                   else v) for k, v in legacy_first.items()
                               if k != "hungarian_assignment"},
        "legacy_last_frame": {k: (float(v) if isinstance(v, (int, float))
                                  else v) for k, v in legacy_last.items()
                              if k != "hungarian_assignment"},
        "n_rows": len(rows),
        "n_records": len(records),
        "n_aligned_tracks": len(mapping),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["strict"], indent=2, default=float))
    print("legacy_first:", {k: round(v, 3) for k, v in
          summary["legacy_first_frame"].items()
          if isinstance(v, float)})
    print("legacy_last:", {k: round(v, 3) for k, v in
          summary["legacy_last_frame"].items()
          if isinstance(v, float)})


if __name__ == "__main__":
    main()
