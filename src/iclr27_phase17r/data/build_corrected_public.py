"""Build full Phase17R public rows with causal geometry and immutable orders."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SEEDS = (20260825, 20260826, 20260827)
TRAIN_ROLES = {"known_bank", "novel_correspondence_train"}
CAL_ROLES = {"known_calibration", "novel_calibration"}
AUDIT_ROLES = {"known_audit", "novel_audit"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def parse_box(v: Any) -> list[float]:
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def stable_video_order(videos: Iterable[int], seed: int) -> list[int]:
    def key(v: int) -> tuple[str, int]:
        return hashlib.sha256((str(seed) + ":" + str(v)).encode()).hexdigest(), v
    return sorted(set(videos), key=key)


def fixed_eligible(rows: list[dict[str, Any]], selected: set[int], rank_field: str) -> list[dict[str, Any]]:
    history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    out = []
    for r in sorted(rows, key=lambda x: int(x[rank_field])):
        if r["gt_role_common"] != "novel" or int(r["gt_category_id_common"]) not in selected:
            continue
        cat, vid, track = int(r["gt_category_id_common"]), int(r["video_id"]), int(r["gt_track_id"])
        if any(pv != vid and (pv, pt) != (vid, track) for pv, pt in history[cat]):
            out.append(r)
        history[cat].append((vid, track))
    return out


def main() -> None:
    src = ROOT / "data/iclr27_phase17r/sources"
    rows = [dict(r) for r in csv.DictReader((src / "public_role_rows_phase17.csv").open())]
    annotation = json.loads((src / "tao_train_annotations.json").read_text())
    images = {int(x["id"]): x for x in annotation["images"]}
    roles = json.loads((src / "phase17_roles.json").read_text())

    for r in rows:
        im = images[int(r["image_id"])]
        width, height = int(im["width"]), int(im["height"])
        box = parse_box(r["bbox_xyxy"])
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        r.update({
            "image_width": width, "image_height": height, "image_path": im["file_name"],
            "box_x1_norm": max(0.0, min(1.0, x1 / width)),
            "box_y1_norm": max(0.0, min(1.0, y1 / height)),
            "box_x2_norm": max(0.0, min(1.0, x2 / width)),
            "box_y2_norm": max(0.0, min(1.0, y2 / height)),
            "box_width_norm": min(1.0, bw / width), "box_height_norm": min(1.0, bh / height),
            "box_area_norm": min(1.0, bw * bh / (width * height)),
            "box_aspect_log": math.log(max(bw / bh, 1e-6)),
            "border_left_norm": max(0.0, min(1.0, x1 / width)),
            "border_top_norm": max(0.0, min(1.0, y1 / height)),
            "border_right_norm": max(0.0, min(1.0, (width - x2) / width)),
            "border_bottom_norm": max(0.0, min(1.0, (height - y2) / height))
        })

    # Causal quantities are built only from current and earlier observations.
    by_track: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_track[(int(r["video_id"]), int(r["track_id"]))].append(r)
    later_diff = 0
    for track_rows in by_track.values():
        track_rows.sort(key=lambda r: (int(r["frame_id"]), int(r["proposal_local_id"]), r["row_key"]))
        prev_box = None
        prev_smooth = None
        for age, r in enumerate(track_rows):
            cur = parse_box(r["bbox_xyxy"])
            if prev_smooth is None:
                smooth = cur
                stability = 0.0
            else:
                smooth = [(0.70 * a + 0.30 * b) for a, b in zip(cur, prev_smooth)]
                stability = box_iou(cur, prev_box)
                later_diff += int(any(abs(a - b) > 1e-6 for a, b in zip(cur, smooth)))
            r["causal_prefix_age"] = age
            r["causal_prefix_count"] = age + 1
            r["causal_prefix_age_norm"] = min(age, 99) / 99.0
            r["causal_box_stability_iou"] = stability
            r["causal_smoothed_bbox_xyxy"] = json.dumps(smooth, separators=(",", ":"))
            prev_box, prev_smooth = cur, smooth

    videos = {int(r["video_id"]) for r in rows}
    for oi, seed in enumerate(SEEDS):
        video_rank = {v: i for i, v in enumerate(stable_video_order(videos, seed))}
        order = sorted(range(len(rows)), key=lambda i: (
            video_rank[int(rows[i]["video_id"])], int(rows[i]["frame_id"]),
            int(rows[i]["proposal_local_id"]), int(rows[i]["track_id"]), rows[i]["row_key"]))
        for rank, idx in enumerate(order):
            rows[idx]["event_rank_order" + str(oi)] = rank
    for r in rows:
        r["event_rank"] = r["event_rank_order0"]

    rows.sort(key=lambda r: int(r["event_rank_order0"]))
    out_csv = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    fields = list(rows[0].keys())
    with tmp_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp_csv, out_csv)

    by_role = Counter(r["role17"] for r in rows)
    role_videos = defaultdict(set)
    role_categories = defaultdict(set)
    for r in rows:
        role_videos[r["role17"]].add(int(r["video_id"]))
        if int(r["assigned"]): role_categories[r["role17"]].add(int(r["gt_category_id_common"]))
    split_sets = {
        "train": {r["row_key"] for r in rows if r["role17"] in TRAIN_ROLES},
        "calibration": {r["row_key"] for r in rows if r["role17"] in CAL_ROLES},
        "audit": {r["row_key"] for r in rows if r["role17"] in AUDIT_ROLES}
    }
    train_v = {int(r["video_id"]) for r in rows if r["role17"] in TRAIN_ROLES}
    cal_v = {int(r["video_id"]) for r in rows if r["role17"] in CAL_ROLES}
    audit_v = {int(r["video_id"]) for r in rows if r["role17"] in AUDIT_ROLES}
    leakage = {
        "train_calibration_rows": len(split_sets["train"] & split_sets["calibration"]),
        "train_audit_rows": len(split_sets["train"] & split_sets["audit"]),
        "calibration_audit_rows": len(split_sets["calibration"] & split_sets["audit"]),
        "train_calibration_videos": sorted(train_v & cal_v),
        "train_audit_videos": sorted(train_v & audit_v),
        "calibration_audit_videos": sorted(cal_v & audit_v)
    }
    split_audit = {
        "protocol": "trackocd_iclr27_phase17r_split_and_leakage",
        "source_roles_immutable": str((src / "phase17_roles.json").resolve()),
        "population": {
            "all_rows": len(rows), "train_rows": len(split_sets["train"]),
            "calibration_rows": len(split_sets["calibration"]), "audit_rows": len(split_sets["audit"]),
            "role_rows": dict(sorted(by_role.items())),
            "role_videos": {k: len(v) for k, v in sorted(role_videos.items())},
            "role_assigned_categories": {k: len(v) for k, v in sorted(role_categories.items())}
        },
        "leakage": leakage,
        "all_disjoint": not any([leakage["train_calibration_rows"], leakage["train_audit_rows"], leakage["calibration_audit_rows"], leakage["train_calibration_videos"], leakage["train_audit_videos"], leakage["calibration_audit_videos"]]),
        "q1_labels_used": False,
        "devplus_labels_used_for_selection": False
    }
    atomic_json(ROOT / "outputs/iclr27_phase17r/manifests/data_split_and_leakage_audit.json", split_audit)

    denom = {"calibration": {}, "audit": {}}
    selected = {
        "calibration": set(map(int, roles["novel_calibration_categories"])),
        "audit": set(map(int, roles["novel_audit_categories"]))
    }
    populations = {
        "calibration": [r for r in rows if r["role17"] in CAL_ROLES],
        "audit": [r for r in rows if r["role17"] in AUDIT_ROLES]
    }
    for split in ("calibration", "audit"):
        for oi, seed in enumerate(SEEDS):
            eligible = fixed_eligible(populations[split], selected[split], "event_rank_order" + str(oi))
            keys = [r["row_key"] for r in eligible]
            observable = [r for r in eligible if int(r["assigned"]) and float(r["row_iou"]) >= 0.5]
            denom[split][str(seed)] = {
                "eligible": len(keys), "row_keys_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                "categories": sorted({int(r["gt_category_id_common"]) for r in eligible}),
                "videos": sorted({int(r["video_id"]) for r in eligible}),
                "oracle_observable_rows": len(observable),
                "oracle_observable_categories": sorted({int(r["gt_category_id_common"]) for r in observable}),
                "oracle_observable_videos": sorted({int(r["video_id"]) for r in observable}),
                "row_keys": keys
            }
    atomic_json(ROOT / "outputs/iclr27_phase17r/eval/fixed_ct_denominators.json", {
        "protocol": "trackocd_iclr27_phase17r_prediction_independent_fixed_ct",
        "episode_order_seeds": list(SEEDS), "selected_categories": {k: sorted(v) for k, v in selected.items()},
        "denominators": denom, "prediction_independent": True
    })

    size_counts = Counter((int(r["image_width"]), int(r["image_height"])) for r in rows)
    contract = {
        "protocol": "trackocd_iclr27_phase17r_geometry_chronology_contract",
        "rows": len(rows), "unique_row_keys": len({r["row_key"] for r in rows}),
        "actual_image_size_count": len(size_counts),
        "common_image_sizes": [{"width": w, "height": h, "rows": n} for (w, h), n in size_counts.most_common(12)],
        "rows_not_640x480": sum((int(r["image_width"]), int(r["image_height"])) != (640, 480) for r in rows),
        "later_temporal_boxes_different_from_raw": later_diff,
        "later_temporal_difference_nontrivial": later_diff >= 100,
        "event_rank_columns": ["event_rank_order0", "event_rank_order1", "event_rank_order2"],
        "event_rank_unique_each_order": all(len({int(r["event_rank_order" + str(i)]) for r in rows}) == len(rows) for i in range(3)),
        "downstream_required_to_consume_event_rank": True,
        "model_input_fields": ["score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou"],
        "forbidden_deployment_fields": ["row_iou", "track_temporal_iou", "gt_bbox_xyxy", "gt_category_id_common", "proposal_track_length", "source_family", "event_rank"],
        "future_or_gt_deployment_input": False,
        "full_track_length_deployment_input": False,
        "physical_id_semantic_feature": False,
        "passed": len({r["row_key"] for r in rows}) == len(rows) and later_diff >= 100 and split_audit["all_disjoint"]
    }
    atomic_json(ROOT / "outputs/iclr27_phase17r/eval/geometry_and_chronology_contract.json", contract)
    print(json.dumps({"split": split_audit, "contract": contract, "denominator_summary": {s: {k: v["eligible"] for k, v in d.items()} for s, d in denom.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
