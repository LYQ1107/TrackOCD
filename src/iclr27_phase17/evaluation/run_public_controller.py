"""Phase 17 public complete-episode calibration and one frozen DEV+ audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase15s.evaluation.causal_controller import build_bank, replay, strict_metrics
from src.iclr27_phase15s.validation.transition_validator import validate_transitions

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ORDERS = (20260825, 20260826, 20260827)


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, path)


def _rows(path: Path, roles: dict[int, str] | None = None) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            q = dict(r)
            # The common audit names are intentionally translated once at this
            # boundary; the original common CSV remains immutable.
            q["gt_role"] = q.get("gt_role_common", q.get("gt_role", "fp"))
            if q["gt_role"] == "supported_known": q["gt_role"] = "supported_known"
            q["gt_category_id"] = q.get("gt_category_id_common", q.get("gt_category_id", -1))
            q["gt_track_id"] = q.get("gt_track_id", -1)
            q["video_id"] = q["video_id"]; q["frame_id"] = q["frame_id"]; q["proposal_local_id"] = q.get("proposal_local_id", 0); q["track_id"] = q.get("track_id", -1)
            if roles is not None: q["role17"] = roles.get(int(float(q["video_id"])), "excluded")
            out.append(q)
    return out


def _feature_lookup(path: Path) -> dict[str, int]:
    z = np.load(path, allow_pickle=False); return {str(k): i for i, k in enumerate(z["row_keys"])}


def _features(rows: list[dict[str, Any]], path: Path, mode: str = "roi") -> np.ndarray:
    z = np.load(path, allow_pickle=False); lookup = {str(k): i for i, k in enumerate(z["row_keys"])}
    missing = [r.get("row_key") for r in rows if r.get("row_key") not in lookup]
    if missing: raise RuntimeError(f"missing feature keys: {missing[:3]}")
    return np.asarray(z[mode][[lookup[str(r["row_key"])] for r in rows]], dtype=np.float32)


def _video_order(rows: list[dict[str, Any]], seed: int) -> list[int]:
    vids = sorted({int(r["video_id"]) for r in rows}); vids.sort(key=lambda v: ((v * 2654435761 + seed) % 4294967291, v)); rank = {v: i for i, v in enumerate(vids)}
    return sorted(range(len(rows)), key=lambda i: (rank[int(rows[i]["video_id"])], int(rows[i]["frame_id"]), int(rows[i].get("proposal_local_id", 0)), i))


def full_calibration(rows: list[dict[str, Any]], feats: np.ndarray, bank: dict[str, Any], known: set[int], selected: set[int]) -> tuple[dict[str, float], dict[str, Any]]:
    vals = []
    for seed in ORDERS:
        order = _video_order(rows, seed); rr = [rows[i] for i in order]; ff = feats[order]
        # A single preregistered historical operating point is used for this
        # feasibility probe.  It avoids an unregistered threshold lottery;
        # because no alternative is compared, the manifest always marks the
        # reuse threshold unidentified (even if this point happens to reuse).
        for tk in (0.15,):
            for tc in (0.15,):
                for margin in (0.0,):
                    ds, internal = replay(rr, ff, bank, {"tau_known": tk, "tau_cross_physical_reuse": tc, "margin_new": margin}, known)
                    if not internal.get("valid"): continue
                    met = strict_metrics(rr, ds, known, selected); ct = met["fixed_ct"]["recall"]
                    vals.append({"seed": seed, "tau_known": tk, "tau_cross_physical_reuse": tc, "margin_new": margin,
                                 "objective": met["known_occurrence_acc"] + ct + met["birth_precision"] - .1 * met["fragmentation_mean_states"],
                                 "metrics": {"known": met["known_occurrence_acc"], "known_macro": met["known_macro_category_acc"], "ct": ct, "ct_eligible": met["fixed_ct"]["eligible"], "birth_precision": met["birth_precision"], "fragmentation": met["fragmentation_mean_states"]}})
    vals.sort(key=lambda x: (-x["objective"], x["tau_known"], x["tau_cross_physical_reuse"], x["margin_new"], x["seed"]))
    max_ct = max((v["metrics"]["ct"] for v in vals), default=0.0); identified = False
    best = vals[0] if vals else {"tau_known": .45, "tau_cross_physical_reuse": .45, "margin_new": .05, "objective": None, "metrics": {}}
    thresholds = {k: best[k] for k in ("tau_known", "tau_cross_physical_reuse", "margin_new")}
    return thresholds, {"video_orders": list(ORDERS), "grid_size": 1, "evaluated": len(vals), "complete_episode_rows": len(rows), "max_calibration_rows": None, "min_ct_denominator": min((v["metrics"]["ct_eligible"] for v in vals), default=0), "max_ct_recall": max_ct, "reuse_threshold_unidentified": True, "threshold_identification_note": "single historical operating point; no tie-break or calibration claim", "best": best, "top10": vals[:10]}


def evaluate(rows: list[dict[str, Any]], feats: np.ndarray, bank: dict[str, Any], thresholds: dict[str, float], known: set[int], selected: set[int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ds, internal = replay(rows, feats, bank, thresholds, known); met = strict_metrics(rows, ds, known, selected); met["transition_contract"] = validate_transitions(ds, known, internal_state_count=internal.get("internal_state_count")); met["internal"] = internal; return met, ds, internal


def write_decisions(path: Path, rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); out = []
    for r, d in zip(rows, decisions): q = dict(r); q.update(d); out.append(q)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    roles_manifest = json.loads(args.roles.read_text()); role_map = {int(v): k for k, vs in roles_manifest["roles"].items() for v in vs}
    known = {int(x) for x in json.loads(args.known.read_text())}; cal_cats = set(map(int, roles_manifest["novel_calibration_categories"])); audit_cats = set(map(int, roles_manifest["novel_audit_categories"]))
    pub_all = _rows(args.public_rows, role_map); pfeat_all = _features(pub_all, args.public_features, args.mode)
    bank_idx = [i for i, r in enumerate(pub_all) if r.get("role17") == "known_bank"]
    bank = build_bank(pub_all, pfeat_all, roles_manifest["roles"]["known_bank"], known, args.mode)
    cal_idx = [i for i, r in enumerate(pub_all) if r.get("role17") in {"known_calibration", "novel_calibration"}]
    cal_rows = [pub_all[i] for i in cal_idx]; cal_feat = pfeat_all[cal_idx]; thresholds, cal = full_calibration(cal_rows, cal_feat, bank, known, cal_cats)
    audit_idx = [i for i, r in enumerate(pub_all) if r.get("role17") in {"known_audit", "novel_audit"}]
    audit_rows = [pub_all[i] for i in audit_idx]; audit_feat = pfeat_all[audit_idx]; audit, ad, ai = evaluate(audit_rows, audit_feat, bank, thresholds, known, audit_cats)
    pub_cal = {"protocol": "trackocd_iclr27_phase17_public_calibration", "representation": args.mode, "bank": {"categories": bank["categories"], "rows": bank["rows"], "tracks": bank["tracks"]}, "thresholds": thresholds, "grid": cal, "fixed_ct_selected_categories": sorted(cal_cats), "complete_episodes": True, "label_access": "public annotations only; no DEV+/Q1 labels"}
    pub_audit = {"protocol": "trackocd_iclr27_phase17_public_final_audit", "representation": args.mode, "roles": {k: roles_manifest["roles"][k] for k in ("known_audit", "novel_audit")}, "selected_fixed_ct_categories": sorted(audit_cats), "metrics": audit, "public_audit_gate": {"known_ge_0_60": audit["known_occurrence_acc"] >= .60, "fixed_ct_gt_0": audit["fixed_ct"]["recall"] > 0, "ct_categories": len(audit["fixed_ct"]["by_category"]), "ct_videos": len(audit["fixed_ct"]["by_video"]), "predicted_existing_precision_ge_0_20": audit["fixed_ct"]["predicted_existing_precision"] >= .20, "transition_valid": audit["transition_contract"]["valid"]}}
    atomic(args.out_calibration, pub_cal); atomic(args.out_audit, pub_audit); write_decisions(args.out_dir / "csv/public_final_audit_decisions.csv", audit_rows, ad)
    # DEV+ is a single locked evaluation only when the public audit passes.
    dev_rows = _rows(args.devplus_rows); dev_feat = _features(dev_rows, args.devplus_features, args.mode); dev_selected = {int(x) for x in json.loads(args.devplus_categories.read_text())}; dev_result = None
    if all(pub_audit["public_audit_gate"].values()):
        dm, dd, di = evaluate(dev_rows, dev_feat, bank, thresholds, known, dev_selected); dev_result = {"protocol": "trackocd_iclr27_phase17_strict_devplus_summary", "representation": args.mode, "metrics": dm, "one_time_frozen_public_lock": True}; atomic(args.out_devplus, dev_result); write_decisions(args.out_dir / "csv/strict_devplus_decisions.csv", dev_rows, dd)
    lock = {"protocol": "trackocd_iclr27_phase17_public_lock", "representation": args.mode, "thresholds": thresholds, "calibration_manifest": str(args.out_calibration.resolve()), "audit_manifest": str(args.out_audit.resolve()), "reuse_threshold_unidentified": bool(cal["reuse_threshold_unidentified"]), "devplus_run": dev_result is not None, "feature_source": str(args.public_features.resolve())}
    atomic(args.out_lock, lock)
    result = {"public_calibration": pub_cal, "public_audit": pub_audit, "devplus": dev_result, "public_lock": lock}
    print(json.dumps(result, indent=2, sort_keys=True)); return result


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--public-rows", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/public_role_rows.csv"); ap.add_argument("--public-features", type=Path, default=ROOT / "data/iclr27_phase17/sources/public_dinov2_features.npz"); ap.add_argument("--devplus-rows", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/common_devplus.csv"); ap.add_argument("--devplus-features", type=Path, default=ROOT / "data/iclr27_phase17/sources/devplus_dinov2_features.npz"); ap.add_argument("--roles", type=Path, default=ROOT / "outputs/iclr27_phase17/manifests/data_split_and_leakage_audit.json"); ap.add_argument("--known", type=Path, default=ROOT / "data/iclr27_phase17/sources/supported_known_ids.json"); ap.add_argument("--devplus-categories", type=Path, default=ROOT / "outputs/iclr27_phase15s/eval/fixed_ct_contract.json"); ap.add_argument("--mode", choices=["cls", "roi"], default="roi"); ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/iclr27_phase17"); ap.add_argument("--out-calibration", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/public_calibration_summary.json"); ap.add_argument("--out-audit", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/public_final_audit.json"); ap.add_argument("--out-devplus", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/strict_devplus_summary.json"); ap.add_argument("--out-lock", type=Path, default=ROOT / "outputs/iclr27_phase17/manifests/public_lock.json"); args = ap.parse_args()
    # The historical fixed-CT manifest stores an object; Phase15S's selected
    # DEV+ categories are used only for the frozen DEV+ denominator.
    if args.devplus_categories.exists():
        x = json.loads(args.devplus_categories.read_text()); cats = x.get("selected_novel_categories", x if isinstance(x, list) else [])
        tmp = args.out_dir / "manifests/devplus_selected_categories_phase17.json"; atomic(tmp, cats); args.devplus_categories = tmp
    run(args)


if __name__ == "__main__": main()
