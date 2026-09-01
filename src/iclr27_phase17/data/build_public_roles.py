"""Deterministic, opportunity-driven Phase 17 public role construction.

The public DSCT videos are reused as they are.  Only annotation-derived
opportunity statistics choose roles; no DEV+/Q1 frequency, representation, or
controller output is consulted.  Roles are video-disjoint and all chosen rows
remain in the common alignment CSV for an auditable denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
Q1_VIDEOS = {88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276, 1572,
             1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def _iv(r: dict[str, Any], k: str, default: int = -1) -> int:
    try:
        return int(float(r.get(k, default)))
    except (TypeError, ValueError):
        return default


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [dict(r) for r in csv.DictReader(f)]


def _fixed_rows(rows: list[dict[str, Any]], categories: set[int]) -> list[dict[str, Any]]:
    """GT-only chronological fixed-CT eligibility (row-level denominator)."""
    history: dict[int, set[tuple[int, int]]] = defaultdict(set)
    eligible: list[dict[str, Any]] = []
    order = sorted(rows, key=lambda r: (_iv(r, "video_id"), _iv(r, "frame_id"), _iv(r, "proposal_local_id")))
    for r in order:
        if _iv(r, "assigned") and str(r.get("gt_role_common")) == "novel":
            c = _iv(r, "gt_category_id_common")
            key = (_iv(r, "video_id"), _iv(r, "gt_track_id"))
            if c in categories and any(v != key[0] for v, _ in history[c]):
                eligible.append(r)
            history[c].add(key)
    return eligible


def _pairs(rows: list[dict[str, Any]], categories: set[int]) -> int:
    tracks: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in rows:
        if _iv(r, "assigned") and str(r.get("gt_role_common")) == "novel":
            c = _iv(r, "gt_category_id_common")
            if c in categories:
                tracks[c].add((_iv(r, "video_id"), _iv(r, "gt_track_id")))
    total = 0
    for vals in tracks.values():
        vals = sorted(vals)
        for i, (v, _) in enumerate(vals):
            total += sum(int(v != v2) for v2, _ in vals[i + 1:])
    return total


def _select_categories(rows: list[dict[str, Any]], legal: set[int], *, exclude: set[int],
                       min_categories: int, min_rows: int, min_pairs: int) -> tuple[list[int], dict[int, dict[str, Any]]]:
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if _iv(r, "video_id") not in legal or not _iv(r, "assigned") or str(r.get("gt_role_common")) != "novel":
            continue
        c = _iv(r, "gt_category_id_common")
        if c not in exclude:
            by[c].append(r)
    stats: dict[int, dict[str, Any]] = {}
    for c, rs in by.items():
        stats[c] = {"rows": len(rs), "videos": sorted({_iv(r, "video_id") for r in rs}),
                    "tracks": sorted({(_iv(r, "video_id"), _iv(r, "gt_track_id")) for r in rs}),
                    "fixed_rows": len(_fixed_rows(rs, {c})), "cross_video_pairs": _pairs(rs, {c})}
    # Categories with one video cannot create an identifiable cross-video
    # correspondence and are intentionally left out of role construction.
    ordered = sorted((c for c, s in stats.items() if len(s["videos"]) >= 2 and s["fixed_rows"] > 0),
                     key=lambda c: (-stats[c]["fixed_rows"], -stats[c]["cross_video_pairs"], -stats[c]["rows"], c))
    selected: list[int] = []
    while ordered and (len(selected) < min_categories or
                       len(_fixed_rows([r for r in rows if _iv(r, "video_id") in legal], set(selected))) < min_rows or
                       _pairs(rows, set(selected)) < min_pairs):
        selected.append(ordered.pop(0))
    return selected, stats


def _choose_videos_for_categories(rows: list[dict[str, Any]], categories: set[int], legal: set[int], used: set[int]) -> list[int]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        v = _iv(r, "video_id")
        if v in legal and v not in used and _iv(r, "assigned") and str(r.get("gt_role_common")) == "novel" and _iv(r, "gt_category_id_common") in categories:
            by_video[v].append(r)
    # Greedily cover every category with at least two videos, then include all
    # remaining occurrences of the selected categories.  This is deterministic
    # and preserves the fixed denominator rather than sampling rows.
    chosen: set[int] = set()
    for c in sorted(categories):
        cand = sorted(v for v, rs in by_video.items() if any(_iv(r, "gt_category_id_common") == c for r in rs))
        for v in cand[:2]:
            chosen.add(v)
    for v in sorted(by_video):
        chosen.add(v)
    return sorted(chosen)


def _rank_videos(rows: list[dict[str, Any]], available: set[int], n: int,
                 require: str | None = None) -> list[int]:
    stats: dict[int, Counter] = defaultdict(Counter)
    cats: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        v = _iv(r, "video_id")
        if v not in available:
            continue
        if _iv(r, "assigned") and str(r.get("gt_role_common")) == "supported_known":
            stats[v]["known"] += 1; cats[v].add(_iv(r, "gt_category_id_common"))
        elif _iv(r, "assigned") and str(r.get("gt_role_common")) == "novel":
            stats[v]["novel"] += 1
        else:
            stats[v]["fp"] += 1
    cand = [v for v in available if stats[v]["known"] + stats[v]["novel"] > 0]
    if require == "novel":
        cand = [v for v in cand if stats[v]["novel"] > 0]
    # Preserve competing states: prefer videos with both known and novel, then
    # known density/categories, false proposals, and stable small IDs.
    # Greedy diversity first: calibration/audit gates are category and video
    # gates, so a dense single-category video must not consume the whole role.
    chosen: list[int] = []
    seen: set[int] = set()
    while cand and len(chosen) < n:
        v = max(cand, key=lambda x: (len(cats[x] - seen),
                                     int(stats[x]["known"] > 0 and stats[x]["novel"] > 0),
                                     stats[x]["known"], stats[x]["novel"], stats[x]["fp"], -x))
        chosen.append(v); seen.update(cats[v]); cand.remove(v)
    return chosen


def build(common_csv: Path, known_path: Path, devplus_gt: Path, frozen_prereg: Path,
          out: Path, rows_csv: Path) -> dict[str, Any]:
    rows = _rows(common_csv)
    known_ids = {int(x) for x in json.loads(known_path.read_text())}
    frozen = json.loads(frozen_prereg.read_text())
    dev_videos = {int(x) for x in frozen["devplus_videos"]}
    legal = {v for v in {_iv(r, "video_id") for r in rows} if v not in dev_videos and v not in Q1_VIDEOS}

    # The category choice is annotation/opportunity driven and is recorded,
    # rather than copied from the old role manifest.
    cal_cats, cal_stats = _select_categories(rows, legal, exclude=set(), min_categories=5, min_rows=100, min_pairs=10)
    audit_cats, audit_stats = _select_categories(rows, legal, exclude=set(cal_cats), min_categories=3, min_rows=50, min_pairs=5)
    used: set[int] = set()
    novel_cal = _choose_videos_for_categories(rows, set(cal_cats), legal, used); used.update(novel_cal)
    novel_audit = _choose_videos_for_categories(rows, set(audit_cats), legal, used); used.update(novel_audit)
    # The meta role is category-disjoint from both chronological roles and is
    # selected from the densest remaining novel evidence.
    remaining_novel = [r for r in rows if _iv(r, "video_id") in legal - used and _iv(r, "assigned") and str(r.get("gt_role_common")) == "novel" and _iv(r, "gt_category_id_common") not in set(cal_cats) | set(audit_cats)]
    novel_train_candidates = sorted({_iv(r, "video_id") for r in remaining_novel})
    density = Counter(_iv(r, "video_id") for r in remaining_novel)
    novel_train = sorted(novel_train_candidates, key=lambda v: (-density[v], v))[:20]; used.update(novel_train)

    # Known calibration/audit are held out after novel roles.  Selecting 20 +
    # 10 videos keeps >=300/200 assigned rows while retaining a large bank.
    known_cal = _rank_videos(rows, legal - used, 20); used.update(known_cal)
    known_audit = _rank_videos(rows, legal - used, 10, require="novel"); used.update(known_audit)
    # If the novel competing-state requirement made the ten-video list short,
    # fill deterministically from the remaining known-rich videos.
    if len(known_audit) < 10:
        extra = _rank_videos(rows, legal - used, 10 - len(known_audit)); known_audit += extra; used.update(extra)
    bank = sorted(legal - used)

    role_videos = {"known_bank": bank, "known_calibration": sorted(known_cal),
                   "known_audit": sorted(known_audit), "novel_correspondence_train": sorted(novel_train),
                   "novel_calibration": sorted(novel_cal), "novel_audit": sorted(novel_audit),
                   "devplus_evaluation": sorted(dev_videos), "q1_quarantined": sorted(Q1_VIDEOS)}
    role_by_video: dict[int, str] = {}
    for role in ["known_bank", "known_calibration", "known_audit", "novel_correspondence_train", "novel_calibration", "novel_audit"]:
        for v in role_videos[role]: role_by_video[v] = role
    role_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        v = _iv(r, "video_id")
        if v in role_by_video:
            q = dict(r); q["role17"] = role_by_video[v]; role_rows[role_by_video[v]].append(q)
    with rows_csv.with_suffix(rows_csv.suffix + ".tmp").open("w", newline="") as f:
        fields = list(rows[0].keys()) + ["role17"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for role in sorted(role_rows):
            for r in role_rows[role]: w.writerow(r)
    os.replace(rows_csv.with_suffix(rows_csv.suffix + ".tmp"), rows_csv)

    dev_categories: set[int] = set()
    with devplus_gt.open() as f:
        for line in f:
            if line.strip(): dev_categories.add(int(json.loads(line)["category_id"]))
    def gate(role: str) -> dict[str, Any]:
        rs = role_rows[role]
        known = [r for r in rs if _iv(r, "assigned") and r.get("gt_role_common") == "supported_known"]
        novel = [r for r in rs if _iv(r, "assigned") and r.get("gt_role_common") == "novel"]
        fp = [r for r in rs if not _iv(r, "assigned")]
        return {"rows": len(rs), "videos": len(role_videos[role]), "assigned_known_rows": len(known),
                "assigned_novel_rows": len(novel), "false_proposal_rows": len(fp),
                "known_categories": len({_iv(r, "gt_category_id_common") for r in known}),
                "novel_categories": len({_iv(r, "gt_category_id_common") for r in novel}),
                "has_false_proposals": bool(fp), "has_competing_states": bool(known and novel)}
    cal_rs = [r for r in role_rows["novel_calibration"] if _iv(r, "gt_category_id_common") in set(cal_cats)]
    aud_rs = [r for r in role_rows["novel_audit"] if _iv(r, "gt_category_id_common") in set(audit_cats)]
    cal_eligible = _fixed_rows(cal_rs, set(cal_cats)); aud_eligible = _fixed_rows(aud_rs, set(audit_cats))
    bank_known_cats = {_iv(r, "gt_category_id_common") for r in role_rows["known_bank"] if _iv(r, "assigned") and r.get("gt_role_common") == "supported_known"}
    # The frozen ceiling is a row-coverage quantity, not a category-count
    # quantity: some DEV+ supported-known categories have no public DSCT
    # annotation at all.  Unobserved categories cannot be counted as bank
    # misses when measuring the legal public bank ceiling.
    dev_csv = common_csv.with_name("common_devplus.csv")
    dev_rows = _rows(dev_csv) if dev_csv.exists() else []
    dev_known_rows = [r for r in dev_rows if _iv(r, "assigned") and r.get("gt_role_common") == "supported_known"]
    covered_dev_rows = sum(_iv(r, "gt_category_id_common") in bank_known_cats for r in dev_known_rows)
    coverage = covered_dev_rows / max(len(dev_known_rows), 1)
    gates = {"known_bank": {"devplus_known_vocabulary": len(dev_categories), "bank_categories": len(bank_known_cats),
                             "devplus_known_rows": len(dev_known_rows), "covered_devplus_known_rows": covered_dev_rows,
                             "ceiling": coverage, "passed": coverage >= .8},
             "known_calibration": {**gate("known_calibration"), "passed": gate("known_calibration")["assigned_known_rows"] >= 300 and gate("known_calibration")["known_categories"] >= 10 and gate("known_calibration")["videos"] >= 20},
             "known_audit": {**gate("known_audit"), "passed": gate("known_audit")["assigned_known_rows"] >= 200 and gate("known_audit")["known_categories"] >= 8 and gate("known_audit")["videos"] >= 10},
             "novel_calibration": {**gate("novel_calibration"), "fixed_ct_rows": len(cal_eligible), "cross_video_pairs": _pairs(cal_rs, set(cal_cats)), "categories": cal_cats, "passed": len(cal_eligible) >= 100 and len(cal_cats) >= 5 and _pairs(cal_rs, set(cal_cats)) >= 10},
             "novel_audit": {**gate("novel_audit"), "fixed_ct_rows": len(aud_eligible), "cross_video_pairs": _pairs(aud_rs, set(audit_cats)), "categories": audit_cats, "unseen_to_calibration": not (set(audit_cats) & set(cal_cats)), "passed": len(aud_eligible) >= 50 and len(audit_cats) >= 3 and _pairs(aud_rs, set(audit_cats)) >= 5 and not (set(audit_cats) & set(cal_cats))}}
    all_pass = all(bool(v.get("passed")) for v in gates.values())
    result = {"protocol": "trackocd_iclr27_phase17_public_roles", "common_csv": str(common_csv.resolve()),
              "known_ids_for_audit_only": sorted(known_ids), "devplus_excluded": sorted(dev_videos), "q1_excluded": sorted(Q1_VIDEOS),
              "legal_public_train_videos": sorted(legal), "roles": role_videos,
              "role_overlap": {f"{a}__{b}": sorted(set(role_videos[a]) & set(role_videos[b])) for a in role_videos for b in role_videos if a < b},
              "novel_calibration_categories": cal_cats, "novel_audit_categories": audit_cats,
              "category_statistics_calibration_candidates": {str(k): v for k, v in cal_stats.items()},
              "category_statistics_audit_candidates": {str(k): v for k, v in audit_stats.items()},
              "gates": gates, "all_opportunity_gates_pass": all_pass,
              "selection": "common-assigned annotation opportunity only; deterministic sorted ties; no DEV+/Q1 frequency",
              "label_access": {"public_gt_for_role_diagnostic": True, "devplus_gt_for_selection": False, "q1_labels": False},
              "terminal_if_failed": "P17-F_CALIBRATION_OPPORTUNITY_BLOCKED" if not all_pass else None}
    atomic_json(out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/common_public.csv")
    ap.add_argument("--known", type=Path, default=ROOT / "data/iclr27_phase17/sources/supported_known_ids.json")
    ap.add_argument("--devplus-gt", type=Path, default=ROOT / "data/iclr27_phase17/sources/devplus_gt_tracks.jsonl")
    ap.add_argument("--frozen-prereg", type=Path, default=ROOT / "outputs/iclr27_phase15/manifests/phase15_preregistration.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/iclr27_phase17/manifests/data_split_and_leakage_audit.json")
    ap.add_argument("--rows-out", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/public_role_rows.csv")
    args = ap.parse_args(); build(args.common, args.known, args.devplus_gt, args.frozen_prereg, args.out, args.rows_out)


if __name__ == "__main__": main()
