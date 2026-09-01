"""Independently reproduce the Phase17R terminal observability diagnosis.

This is Phase18-only evaluator forensics.  It reads immutable Phase17R rows,
denominators, and source code, but never regenerates a Phase17R artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ROWS_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
DENOM_PATH = ROOT / "outputs/iclr27_phase17r/eval/fixed_ct_denominators.json"
EVALUATOR_PATH = ROOT / "src/iclr27_phase17r/evaluation/evaluate_candidate.py"
OUT_PATH = ROOT / "outputs/iclr27_phase18/eval/phase17r_terminal_diagnosis_reproduction.json"
SEEDS = (20260825, 20260826, 20260827)
ROLE_SETS = {
    "calibration": {"known_calibration", "novel_calibration"},
    "audit": {"known_audit", "novel_audit"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def reliable(row: dict[str, str]) -> bool:
    return int(row["assigned"]) == 1 and float(row["row_iou"]) >= 0.5


def row_census(rows: list[dict[str, str]], categories: set[int]) -> dict[str, Any]:
    by_category_video: dict[int, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["gt_role_common"] != "novel":
            continue
        category = int(row["gt_category_id_common"])
        if category in categories:
            by_category_video[category][int(row["video_id"])].append(row)

    category_values = {}
    for category, videos in sorted(by_category_video.items()):
        video_values = {}
        for video, current in sorted(videos.items()):
            video_values[str(video)] = {
                "rows": len(current),
                "assigned_rows": sum(int(row["assigned"]) == 1 for row in current),
                "reliable_rows_iou_ge_0_5": sum(reliable(row) for row in current),
                "exact_zero_iou_rows": sum(float(row["row_iou"]) == 0.0 for row in current),
                "proposal_tracklets": len({int(row["track_id"]) for row in current}),
                "reliable_proposal_tracklets": len({int(row["track_id"]) for row in current if reliable(row)}),
                "gt_tracklets": len({int(row["gt_track_id"]) for row in current}),
            }
        reliable_videos = [int(video) for video, value in video_values.items() if value["reliable_rows_iou_ge_0_5"] > 0]
        category_values[str(category)] = {
            "rows": sum(value["rows"] for value in video_values.values()),
            "reliable_rows_iou_ge_0_5": sum(value["reliable_rows_iou_ge_0_5"] for value in video_values.values()),
            "videos": len(video_values),
            "reliable_videos": reliable_videos,
            "reliable_video_count": len(reliable_videos),
            "two_reliable_videos": len(reliable_videos) >= 2,
            "by_video": video_values,
        }
    return category_values


def both_reliable_rows(
    population: list[dict[str, str]], rank_field: str, eligible_keys: set[str]
) -> dict[str, Any]:
    prior_reliable_videos: dict[int, set[int]] = defaultdict(set)
    eligible_both = []
    for row in sorted(population, key=lambda item: int(item[rank_field])):
        category, video = int(row["gt_category_id_common"]), int(row["video_id"])
        if (
            row["row_key"] in eligible_keys
            and reliable(row)
            and any(prior_video != video for prior_video in prior_reliable_videos[category])
        ):
            eligible_both.append(row)
        if row["gt_role_common"] == "novel" and reliable(row):
            prior_reliable_videos[category].add(video)
    return {
        "count": len(eligible_both),
        "row_keys": [row["row_key"] for row in eligible_both],
        "row_keys_sha256": hashlib.sha256(
            json.dumps([row["row_key"] for row in eligible_both]).encode()
        ).hexdigest(),
        "categories": sorted({int(row["gt_category_id_common"]) for row in eligible_both}),
        "videos": sorted({int(row["video_id"]) for row in eligible_both}),
    }


def perfect_pair_replay(
    population: list[dict[str, str]], rank_field: str, eligible_keys: set[str], merge_capable: bool
) -> dict[str, Any]:
    """Replay a perfect category-pair oracle with exact causal reliability.

    The old controller checks local continuity first.  The merge-capable
    control checks a legal earlier different-video promoted state whenever the
    current row is reliable, and may remap the local track from that action
    forward.  Earlier births/actions remain unchanged.
    """

    states: dict[int, dict[str, Any]] = {}
    local_state: dict[tuple[int, int], int] = {}
    decisions: dict[str, dict[str, Any]] = {}
    next_state = 100000
    ordered = sorted(population, key=lambda item: int(item[rank_field]))

    for position, row in enumerate(ordered):
        if row["gt_role_common"] != "novel":
            continue
        physical = (int(row["video_id"]), int(row["track_id"]))
        video, category = physical[0], int(row["gt_category_id_common"])
        is_reliable = reliable(row)
        candidates = []
        if is_reliable:
            for state_id, state in states.items():
                if state["category"] != category:
                    continue
                if any(anchor_video != video and anchor_physical != physical for anchor_video, anchor_physical in state["anchors"]):
                    candidates.append(state_id)
        perfect_match = min(candidates) if candidates else None

        if not merge_capable and physical in local_state:
            state_id, action, evidence = local_state[physical], "EXISTING_NOVEL", "old_local_first"
        elif merge_capable and perfect_match is not None and local_state.get(physical) != perfect_match:
            state_id, action, evidence = perfect_match, "EXISTING_NOVEL", "later_reliable_merge"
            local_state[physical] = state_id
        elif physical in local_state:
            state_id, action, evidence = local_state[physical], "EXISTING_NOVEL", "local_continuity"
        elif perfect_match is not None:
            state_id, action, evidence = perfect_match, "EXISTING_NOVEL", "perfect_cross_video_pair"
            local_state[physical] = state_id
        else:
            state_id = next_state
            next_state += 1
            action, evidence = "NEW_NOVEL", "new_local_state"
            local_state[physical] = state_id
            states[state_id] = {
                "category": category,
                "birth_video": video,
                "birth_position": position,
                "anchors": [],
            }

        if is_reliable:
            states[state_id]["anchors"].append((video, physical))
        decisions[row["row_key"]] = {
            "action": action,
            "state_id": state_id,
            "evidence": evidence,
        }

    correct_keys = []
    for row in ordered:
        if row["row_key"] not in eligible_keys:
            continue
        decision = decisions[row["row_key"]]
        state = states[decision["state_id"]]
        if (
            decision["action"] == "EXISTING_NOVEL"
            and state["category"] == int(row["gt_category_id_common"])
            and state["birth_video"] != int(row["video_id"])
        ):
            correct_keys.append(row["row_key"])

    actions = Counter(decision["action"] for decision in decisions.values())
    evidence = Counter(decision["evidence"] for decision in decisions.values())
    return {
        "eligible": len(eligible_keys),
        "correct": len(correct_keys),
        "recall": len(correct_keys) / max(len(eligible_keys), 1),
        "correct_row_keys": correct_keys,
        "correct_row_keys_sha256": hashlib.sha256(json.dumps(correct_keys).encode()).hexdigest(),
        "global_states": len(states),
        "actions": dict(actions),
        "evidence": dict(evidence),
        "exact_reliability": True,
        "perfect_semantic_pair": True,
        "past_actions_rewritten": False,
        "merge_capable": merge_capable,
    }


def source_line(source: str, needle: str) -> int:
    for line_number, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return line_number
    raise ValueError(f"source evidence not found: {needle}")


def main() -> None:
    with ROWS_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    denominators = json.loads(DENOM_PATH.read_text())
    source = EVALUATOR_PATH.read_text()
    assert len(rows) == 43423 and len({row["row_key"] for row in rows}) == 43423

    audit_categories = set(map(int, denominators["selected_categories"]["audit"]))
    audit_population = [row for row in rows if row["role17"] in ROLE_SETS["audit"]]
    audit_census = row_census(audit_population, audit_categories)
    order_diagnoses = {}
    calibration_diagnoses = {}
    oracle_controls = {}

    for order_index, seed in enumerate(SEEDS):
        rank_field = f"event_rank_order{order_index}"
        audit_entry = denominators["denominators"]["audit"][str(seed)]
        audit_key_set = set(audit_entry["row_keys"])
        audit_rows = [row for row in audit_population if row["row_key"] in audit_key_set]
        order_diagnoses[str(seed)] = {
            "eligible": len(audit_rows),
            "categories": sorted({int(row["gt_category_id_common"]) for row in audit_rows}),
            "videos": sorted({int(row["video_id"]) for row in audit_rows}),
            "current_reliable_rows_iou_ge_0_5": sum(reliable(row) for row in audit_rows),
            "exact_zero_iou_rows": sum(float(row["row_iou"]) == 0.0 for row in audit_rows),
            "positive_below_0_5_iou_rows": sum(0.0 < float(row["row_iou"]) < 0.5 for row in audit_rows),
            "row_keys_sha256_reproduced": hashlib.sha256(json.dumps(audit_entry["row_keys"]).encode()).hexdigest(),
            "row_keys_sha256_recorded": audit_entry["row_keys_sha256"],
        }

        calibration_population = [row for row in rows if row["role17"] in ROLE_SETS["calibration"]]
        calibration_entry = denominators["denominators"]["calibration"][str(seed)]
        calibration_keys = set(calibration_entry["row_keys"])
        calibration_diagnoses[str(seed)] = {
            "eligible": len(calibration_keys),
            "both_prior_different_video_reliable_source_and_current_reliable_target": both_reliable_rows(
                calibration_population, rank_field, calibration_keys
            ),
        }
        oracle_controls[str(seed)] = {
            "old_local_first_perfect_pair": perfect_pair_replay(
                calibration_population, rank_field, calibration_keys, merge_capable=False
            ),
            "merge_capable_perfect_pair": perfect_pair_replay(
                calibration_population, rank_field, calibration_keys, merge_capable=True
            ),
        }

    reproduced = {
        "audit_categories_are_267_831_1014": sorted(audit_categories) == [267, 831, 1014],
        "audit_each_order_30": all(value["eligible"] == 30 for value in order_diagnoses.values()),
        "audit_each_order_zero_current_reliable": all(
            value["current_reliable_rows_iou_ge_0_5"] == 0 for value in order_diagnoses.values()
        ),
        "audit_each_order_23_exact_zero_iou": all(
            value["exact_zero_iou_rows"] == 23 for value in order_diagnoses.values()
        ),
        "no_audit_category_has_two_reliable_videos": all(
            not value["two_reliable_videos"] for value in audit_census.values()
        ),
        "calibration_both_reliable_matches_prior_independent_counts": [
            calibration_diagnoses[str(seed)]["both_prior_different_video_reliable_source_and_current_reliable_target"]["count"]
            for seed in SEEDS
        ] == [67, 53, 45],
        "old_local_first_perfect_pair_matches_prior_independent_counts": [
            oracle_controls[str(seed)]["old_local_first_perfect_pair"]["correct"] for seed in SEEDS
        ] == [29, 45, 47],
        "merge_capable_perfect_pair_matches_prior_independent_counts": [
            oracle_controls[str(seed)]["merge_capable_perfect_pair"]["correct"] for seed in SEEDS
        ] == [82, 68, 60],
    }

    result = {
        "protocol": "trackocd_iclr27_phase18_phase17r_terminal_diagnosis_reproduction",
        "historical_artifacts_modified": False,
        "sources": {
            "corrected_rows": str(ROWS_PATH.resolve()),
            "corrected_rows_sha256": sha256(ROWS_PATH),
            "fixed_denominators": str(DENOM_PATH.resolve()),
            "fixed_denominators_sha256": sha256(DENOM_PATH),
            "phase17r_evaluator": str(EVALUATOR_PATH.resolve()),
            "phase17r_evaluator_sha256": sha256(EVALUATOR_PATH),
        },
        "population": {"rows": len(rows), "unique_row_keys": len({row["row_key"] for row in rows})},
        "audit_selected_categories": sorted(audit_categories),
        "audit_category_video_census": audit_census,
        "audit_registered_order_diagnosis": order_diagnoses,
        "calibration_both_reliable_diagnosis": calibration_diagnoses,
        "perfect_semantic_pair_exact_reliability_controls": oracle_controls,
        "phase17r_code_forensics": {
            "oracle_both_routing_oracleizes_only_known_and_observability": True,
            "semantic_pair_scorer_remains_learned": True,
            "oracle_control_call_line": source_line(source, '("oracle_both_routing", True, True)'),
            "learned_pair_scorer_line": source_line(source, "probs = scorer.score"),
            "local_continuity_checked_before_global_matching": True,
            "local_first_line": source_line(source, "elif physical in local_novel"),
            "global_candidate_loop_line": source_line(source, "for k, state in states.items()"),
            "local_first_precedes_global": source_line(source, "elif physical in local_novel")
            < source_line(source, "for k, state in states.items()"),
            "interpretation": "Phase17R oracle_both_routing uses exact known/observability routing but retains the trained pair scorer; it is not a perfect semantic correspondence oracle. The local-novel branch executes before global matching, so a provisional local state cannot later switch to an earlier correct global state.",
        },
        "reproduction_checks": reproduced,
        "all_required_reproductions_match": all(reproduced.values()),
    }
    atomic_json(OUT_PATH, result)
    print(json.dumps({
        "audit_categories": sorted(audit_categories),
        "audit_orders": {seed: {"eligible": value["eligible"], "reliable": value["current_reliable_rows_iou_ge_0_5"], "zero_iou": value["exact_zero_iou_rows"]} for seed, value in order_diagnoses.items()},
        "calibration_both_reliable": [calibration_diagnoses[str(seed)]["both_prior_different_video_reliable_source_and_current_reliable_target"]["count"] for seed in SEEDS],
        "old_local_first": [oracle_controls[str(seed)]["old_local_first_perfect_pair"]["correct"] for seed in SEEDS],
        "merge_capable": [oracle_controls[str(seed)]["merge_capable_perfect_pair"]["correct"] for seed in SEEDS],
        "all_required_reproductions_match": result["all_required_reproductions_match"],
    }, indent=2))


if __name__ == "__main__":
    main()
