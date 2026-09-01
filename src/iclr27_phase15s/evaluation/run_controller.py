"""Run public calibration and frozen DEV+ CLS/ROI causal audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.iclr27_phase15s.evaluation.causal_controller import build_bank, calibration_grid, replay, strict_metrics
from src.iclr27_phase15s.validation.transition_validator import validate_transitions

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def read_csv(path): return list(csv.DictReader((ROOT / path).open()))


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, path)


def write_csv(path, rows, decisions):
    out = []
    for r, d in zip(rows, decisions):
        q = dict(r); q.update(d); out.append(q)
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--public-proposals", default="outputs/iclr27_phase15s/dsct_bank/public_roles/proposals.csv"); ap.add_argument("--public-features", default="outputs/iclr27_phase15s/features/public_cls_roi.npz"); ap.add_argument("--devplus-proposals", default="data/iclr27_phase15s/sources/proposals_aligned.csv"); ap.add_argument("--devplus-features", default="outputs/iclr27_phase15s/features/devplus_cls_roi.npz"); ap.add_argument("--roles", default="outputs/iclr27_phase15s/manifests/data_split_and_leakage_audit.json"); ap.add_argument("--out-dir", default="outputs/iclr27_phase15s"); args = ap.parse_args()
    public, dev = read_csv(args.public_proposals), read_csv(args.devplus_proposals); pz, dz = np.load(ROOT / args.public_features, allow_pickle=False), np.load(ROOT / args.devplus_features, allow_pickle=False)
    roles = json.load((ROOT / args.roles).open())["roles"]; known = {int(x) for x in json.load((ROOT / "data/iclr27_phase15s/sources/supported_known_ids.json").open())}; frozen = json.load((ROOT / "outputs/iclr27_phase15/manifests/phase15_preregistration.json").open()); dev_novel = set(map(int, frozen["devplus_novel_categories"]))
    all_summaries = {}; calibration_summary = {}; public_audit = {}; strict = {}
    for mode in ("cls", "roi"):
        bank = build_bank(public, pz[mode], roles["known_bank_train"], known, mode)
        cal_idx = [i for i, r in enumerate(public) if int(r["video_id"]) in set(roles["known_calibration"])]
        cal_rows, cal_feats = [public[i] for i in cal_idx], pz[mode][cal_idx]
        selected_cal = {int(r["gt_category_id"]) for r in cal_rows if r.get("gt_role") == "novel"}
        thresholds, cal = calibration_grid(cal_rows, cal_feats, bank, known, selected_cal)
        calibration_summary[mode] = {"bank": {"categories": bank["categories"], "rows": bank["rows"], "tracks": bank["tracks"]}, "thresholds": thresholds, "grid": cal, "label_access": "public calibration only; no DEV+/Q1 labels"}
        audit_idx = [i for i, r in enumerate(public) if int(r["video_id"]) in set(roles["known_audit"])]
        ar, af = [public[i] for i in audit_idx], pz[mode][audit_idx]
        ad, ai = replay(ar, af, bank, thresholds, known); am = strict_metrics(ar, ad, known, selected_cal); am["transition_contract"] = validate_transitions(ad, known, internal_state_count=ai.get("internal_state_count")); public_audit[mode] = am
        dd, di = replay(dev, dz[mode], bank, thresholds, known); dm = strict_metrics(dev, dd, known, dev_novel); dm["transition_contract"] = validate_transitions(dd, known, internal_state_count=di.get("internal_state_count")); dm["internal"] = di; strict[mode] = dm
        write_csv(ROOT / args.out_dir / "csv" / f"{mode}_devplus.csv", dev, dd)
        write_csv(ROOT / args.out_dir / "csv" / f"{mode}_public_audit.csv", ar, ad)
    atomic_json(ROOT / "outputs/iclr27_phase15s/eval/episodic_calibration_summary.json", {"protocol": "trackocd_iclr27_phase15s16", "candidates": calibration_summary})
    atomic_json(ROOT / "outputs/iclr27_phase15s/eval/public_known_audit.json", {"protocol": "trackocd_iclr27_phase15s16", "candidates": public_audit, "known_roles": {"bank": roles["known_bank_train"], "calibration": roles["known_calibration"], "audit": roles["known_audit"]}, "devplus_labels_used_for_fit": False, "q1_label_used": False})
    atomic_json(ROOT / "outputs/iclr27_phase15s/eval/strict_trackocd_summary.json", {"protocol": "trackocd_iclr27_phase15s16", "candidates": strict, "legacy_gate": {m: {"known_ge_0_60": bool(v["known_occurrence_acc"] >= 0.60), "fixed_ct_gt_0": bool(v["fixed_ct"]["recall"] > 0), "pass": bool(v["known_occurrence_acc"] >= 0.60 and v["fixed_ct"]["recall"] > 0)} for m, v in strict.items()}, "fixed_ct_denominator_same": len({v["fixed_ct"]["fixed_ct"]["eligible"] if False else v["fixed_ct"]["eligible"] for v in strict.values()}) == 1})
    atomic_json(ROOT / "outputs/iclr27_phase15s/eval/transition_contract.json", {"protocol": "trackocd_iclr27_phase15s16", "candidates": {m: v["transition_contract"] for m, v in strict.items()}, "self_state_excluded_from_known": True, "cross_physical_only": True, "overflow_invalid": True})
    print(json.dumps({"calibration": calibration_summary, "public_audit": public_audit, "devplus": {m: {"known": v["known_occurrence_acc"], "known_macro": v["known_macro_category_acc"], "fixed_ct": v["fixed_ct"]["recall"], "fixed_ct_correct": v["fixed_ct"]["correct"], "fixed_ct_eligible": v["fixed_ct"]["eligible"], "states": v["internal"].get("states")} for m, v in strict.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
