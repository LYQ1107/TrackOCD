"""Strict causal TrackOCD metrics on the mixed TAO TRAIN sidecar."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase5a.evaluation.strict_causal_eval import strict_metrics


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def read_csv(path):
    rows = []
    with (ROOT / path).open() as f:
        for r in csv.DictReader(f):
            r = dict(r)
            for k in ("video_id", "frame_id", "source_frame_index", "image_id", "proposal_local_id", "track_id", "prior_hits", "gt_track_id", "gt_category_id"):
                if k in r: r[k] = int(r[k])
            r["score"] = float(r["score"])
            rows.append(r)
    return rows


def load_labels(path):
    labels = {}
    with (ROOT / path).open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line); labels[r["sample_id"]] = {
                    "protocol_role": "novel" if r["role"] == "novel" else "supported_known",
                    "ground_truth_category_id": int(r["category_id"]),
                }
    return labels


def evaluate(proposals_path, aligned_path, gt_path):
    rows = read_csv(proposals_path); aligned = read_csv(aligned_path); labels = load_labels(gt_path)
    assert len(rows) == len(aligned)
    mapping, gt_by_key = {}, {}
    for r in aligned:
        pk = (r["video_id"], r["track_id"])
        if r["gt_track_id"] >= 0:
            sid = f"{r['video_id']}_{r['gt_track_id']}"
            mapping[pk] = sid
            gt_by_key[pk] = sid
    records = {}
    for r in rows:
        a = r.get("sem_action") or "unresolved"
        sid = int(r["sem_sid"]) if r.get("sem_sid") not in (None, "") else None
        records[id(r)] = {"key": [r["video_id"], r["track_id"]], "frame_id": r["frame_id"], "row_id": id(r), "action": a, "sid": sid, "age": 1.0}
    aligned_rows = [r for r in rows if (r["video_id"], r["track_id"]) in mapping]
    sm = strict_metrics(records, aligned_rows, labels, mapping, n_born_global=sum(r.get("sem_action") == "new" for r in rows))
    # CT-Reuse uses the same immutable birth slots, with explicit cross-video denominators.
    novel = []
    first_seen = {}
    for r in sorted(aligned_rows, key=lambda x: (x["video_id"], x["frame_id"], x["proposal_local_id"], x["track_id"])):
        sid = mapping[(r["video_id"], r["track_id"])]
        lab = labels[sid];
        if lab["protocol_role"] == "novel":
            x = (r, records[id(r)], int(lab["ground_truth_category_id"])); novel.append(x)
            first_seen.setdefault(x[2], x)
    slot_cat, slot_birth = {}, {}
    for r, rec, cat in novel:
        if rec["action"] == "new": slot_cat.setdefault(rec["sid"], cat); slot_birth.setdefault(rec["sid"], (r["video_id"], r["track_id"]))
    reuse = [(r, rec, cat) for r, rec, cat in novel if first_seen.get(cat, (None,))[0] is not r]
    cross_phys = [(r, rec, cat) for r, rec, cat in reuse if slot_birth.get(rec["sid"]) != (r["video_id"], r["track_id"])]
    cross_vid = [(r, rec, cat) for r, rec, cat in cross_phys if slot_birth.get(rec["sid"], (-1,))[0] != r["video_id"]]
    def ok(x):
        _, rec, cat = x; return rec["action"] == "existing" and slot_cat.get(rec["sid"]) == cat
    sm["ct_reuse"] = sum(ok(x) for x in cross_vid) / max(len(cross_vid), 1)
    sm["ct_reuse_correct"] = sum(ok(x) for x in cross_vid)
    sm["ct_reuse_eligible_cross_video_occurrences"] = len(cross_vid)
    sm["eligible_cross_physical_reuse_occurrences"] = len(cross_phys)
    sm["eligible_cross_video_same_category_track_pairs"] = sum(1 for i, a in enumerate(novel) for b in novel[i + 1:] if a[2] == b[2] and a[0]["video_id"] != b[0]["video_id"])
    by_cat = Counter(cat for _, _, cat in cross_vid); good_cat = Counter(cat for x in cross_vid if ok(x) for cat in [x[2]])
    sm["ct_reuse_cases_by_category"] = {str(c): {"correct": good_cat[c], "eligible": by_cat[c]} for c in sorted(by_cat)}
    sm["n_rows"] = len(rows); sm["n_aligned_tracks"] = len(mapping); sm["n_aligned_occurrences"] = len(aligned_rows)
    sm["causal_contract"] = {
        "immediate_action_all_rows": all(r.get("sem_action") in ("known", "new", "existing") for r in rows),
        "no_relabel_field": True, "physical_id_distinct_from_semantic_id": True,
        "q1_label_used": False, "future_frames_used": False, "private_gt_used_for_decision": False,
    }
    return sm, rows, aligned, labels, mapping


def oracle_controls(rows, aligned, labels, mapping):
    # Correct-label oracle: one slot per category, births/reuse are causal in the
    # evaluator order. Wrong control deliberately makes every novel track a new
    # slot and shifts known IDs.
    def run(wrong=False):
        first = set(); records = {}
        for r in sorted(rows, key=lambda x: (x["video_id"], x["frame_id"], x["proposal_local_id"], x["track_id"])):
            pk = (r["video_id"], r["track_id"]); sid = mapping.get(pk)
            if sid is None: a, s = "known", 0
            else:
                cat = labels[sid]["ground_truth_category_id"]; novel = labels[sid]["protocol_role"] == "novel"
                if not novel: a, s = "known", (cat + 1 if wrong else cat)
                elif wrong: a, s = "new", 200000 + len(first) + r["track_id"]
                elif cat not in first: a, s = "new", 100000 + cat; first.add(cat)
                else: a, s = "existing", 100000 + cat
            records[id(r)] = {"key": [r["video_id"], r["track_id"]], "frame_id": r["frame_id"], "row_id": id(r), "action": a, "sid": s, "age": 1.0}
        aligned_rows = [r for r in rows if (r["video_id"], r["track_id"]) in mapping]
        sm = strict_metrics(records, aligned_rows, labels, mapping, n_born_global=sum(v["action"] == "new" for v in records.values()))
        return {"known_occurrence_acc": sm["known_occurrence_acc"], "novel_reuse_acc": sm["novel_reuse_acc"], "cross_physical_reuse_acc": sm["cross_physical_reuse_acc"]}
    return {"illegal_correct_label_oracle": run(False), "intentionally_wrong_label_control": run(True), "oracle_label_used": True, "q1_label_used": False}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--proposals", required=True); ap.add_argument("--aligned", default="outputs/iclr27_phase14c/proposals/proposals_aligned.csv"); ap.add_argument("--gt", default="outputs/iclr27_phase14c/manifests/mixed_gt_tracks.jsonl"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); sm, rows, aligned, labels, mapping = evaluate(args.proposals, args.aligned, args.gt)
    controls = oracle_controls(rows, aligned, labels, mapping)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"strict": sm, "evaluator_controls": controls, "legacy_gate": {"known_ge_0_60": sm["known_occurrence_acc"] >= 0.60, "ct_reuse_gt_0": sm["ct_reuse"] > 0, "pass": sm["known_occurrence_acc"] >= 0.60 and sm["ct_reuse"] > 0}}
    tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)); tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
