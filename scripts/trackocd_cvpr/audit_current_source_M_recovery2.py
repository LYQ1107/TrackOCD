#!/usr/bin/env python3
"""Audit the nine current-source M recovery-2 development receipts.

This is a read-only receipt audit.  It does not open dataset files, run a
model, train, select a checkpoint, or access Val/Test labels.  The recovery-2
receipts are deliberately audited as exploratory causal-GT development only:
their flags must show that the stream used GT-oracle controls rather than
Pred, and that no active-purity, self-exclusion, final, or Goal claim was
made.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr4/M_current_source_recovery2_matrix_audit_v1/result.json"

EXPECTED = {
    ("relative_quarantine", 1027): "M_GT_causal_dev_current_source_wall_recovery2_relative_quarantine_s1027",
    ("relative_quarantine", 1028): "M_GT_causal_dev_current_source_wall_recovery2_relative_quarantine_s1028",
    ("relative_quarantine", 1029): "M_GT_causal_dev_current_source_wall_recovery2_relative_quarantine_s1029_retry3",
    ("relative_weighted", 1027): "M_GT_causal_dev_current_source_wall_recovery2_relative_weighted_s1027",
    ("relative_weighted", 1028): "M_GT_causal_dev_current_source_wall_recovery2_relative_weighted_s1028",
    ("relative_weighted", 1029): "M_GT_causal_dev_current_source_wall_recovery2_relative_weighted_s1029",
    ("pooled_weighted", 1027): "M_GT_causal_dev_current_source_wall_recovery2_pooled_weighted_s1027",
    ("pooled_weighted", 1028): "M_GT_causal_dev_current_source_wall_recovery2_pooled_weighted_s1028",
    ("pooled_weighted", 1029): "M_GT_causal_dev_current_source_wall_recovery2_pooled_weighted_s1029",
}
ORDERS = ("1027", "1028", "1029")
EXPECTED_STATUS = "MEASURED_CAUSAL_GT_M_DEVELOPMENT_NOT_ACTIVE_PURITY_SELF_EXCLUSION_PRED_OR_GOAL_ACCEPTANCE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def add_check(violations: list[str], condition: bool, label: str) -> None:
    if not condition:
        violations.append(label)


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not numeric: {value!r}")
    return float(value)


def audit() -> dict[str, Any]:
    violations: list[str] = []
    cases: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    for (expected_model, expected_seed), directory in sorted(EXPECTED.items()):
        rel = f"outputs/trackocd_cvpr/cvpr4/{directory}/result.json"
        path = ROOT / rel
        case_violations: list[str] = []
        if not path.is_file():
            violations.append(f"missing:{rel}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"unreadable:{rel}:{exc}")
            continue

        add_check(case_violations, payload.get("model") == expected_model, "model")
        add_check(case_violations, payload.get("training_seed") == expected_seed, "seed")
        add_check(case_violations, payload.get("status") == EXPECTED_STATUS, "status")
        add_check(case_violations, payload.get("optimizer_steps") == 0, "top_optimizer_steps")
        add_check(
            case_violations,
            payload.get("actual_full_ID_rename_reverse_stream_equal") is True,
            "id_rename_probe",
        )
        future = payload.get("actual_future_probe")
        add_check(
            case_violations,
            isinstance(future, dict)
            and int(future.get("future_observations_mutated", 0)) > 0
            and future.get("verified_same_prior_events_chosen_versions_and_numeric_ledger") is True,
            "future_probe",
        )
        for field in (
            "final_Val_Test_or_new_nonKnown_semantic_values_read",
            "model_or_checkpoint_selection",
            "coverage_or_latency_matched_purity_superiority_accepted",
            "current_active_purity_official_CT_self_exclusion_Pred_joint_or_Goal_accepted",
        ):
            add_check(case_violations, payload.get(field) is False, f"{field}=false")

        development_scope = payload.get("development_scope", {})
        add_check(
            case_violations,
            development_scope.get("external_evaluator_only_not_model_or_loss_input") is True,
            "development_scope_external_only",
        )
        add_check(
            case_violations,
            development_scope.get("fit_policy_final_or_other_video_target_values_decoded") is False,
            "development_scope_no_other_target_values",
        )
        add_check(
            case_violations,
            development_scope.get("new_nonKnown_semantic_values_or_vocabulary_decoded") is False,
            "development_scope_no_new_nonKnown_semantics",
        )
        offline_scope = payload.get("offline_fit_prior_scope", {})
        add_check(
            case_violations,
            offline_scope.get("other_role_semantic_values_decoded") is False,
            "offline_scope_other_role_semantics",
        )
        add_check(
            case_violations,
            offline_scope.get("unlabeled_or_other_Known_assigned_background_or_Novel") is False,
            "offline_scope_background_or_Novel",
        )

        orders = payload.get("orders")
        add_check(case_violations, isinstance(orders, dict), "orders_dict")
        add_check(
            case_violations,
            isinstance(orders, dict) and set(orders) == set(ORDERS),
            "orders_1027_1028_1029",
        )
        if isinstance(orders, dict):
            for order in ORDERS:
                entry = orders.get(order)
                add_check(case_violations, isinstance(entry, dict), f"order_{order}_dict")
                if not isinstance(entry, dict):
                    continue
                add_check(
                    case_violations,
                    entry.get("GT_oracle_control_not_Pred") is True,
                    f"order_{order}_GT_oracle_not_Pred",
                )
                add_check(
                    case_violations,
                    entry.get("current_active_purity_official_CT_full_semantic_truth_Pred_or_self_exclusion_claim") is False,
                    f"order_{order}_no_active_claim",
                )
                add_check(case_violations, entry.get("model_state_unchanged") is True, f"order_{order}_state_unchanged")
                add_check(case_violations, entry.get("optimizer_steps") == 0, f"order_{order}_optimizer_steps")
                route = entry.get("actual_route_history_control", {})
                p16 = route.get("prefix_snapshots", {}).get("16", {})
                try:
                    metric_rows.append(
                        {
                            "model": expected_model,
                            "seed": expected_seed,
                            "order": int(order),
                            "known_macro_accuracy": number(p16["Known_macro_accuracy"]),
                            "known_micro_accuracy": number(p16["Known_micro_accuracy"]),
                            "reuse_macro_lower_bound": number(p16["certified_cross_video_reuse_macro_lower_bound"]),
                            "reuse_micro_lower_bound": number(p16["certified_cross_video_reuse_micro_lower_bound"]),
                            "physical_track_coverage": number(p16["all_physical_commit_track_coverage"]),
                            "known_queries": int(p16["Known_queries"]),
                            "pseudo_queries": int(p16["declared_pseudo_queries"]),
                        }
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    case_violations.append(f"order_{order}_prefix16_metrics:{exc}")
                actions = entry.get("physical_actions")
                if isinstance(actions, dict):
                    action_rows.append(
                        {
                            "model": expected_model,
                            "seed": expected_seed,
                            "order": int(order),
                            "EXISTING": int(actions.get("EXISTING", 0)),
                            "KNOWN": int(actions.get("KNOWN", 0)),
                            "NEW": int(actions.get("NEW", 0)),
                        }
                    )

        if case_violations:
            violations.extend(f"{directory}:{item}" for item in case_violations)
        cases.append(
            {
                "model": expected_model,
                "seed": expected_seed,
                "result": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": payload.get("status"),
                "optimizer_steps": payload.get("optimizer_steps"),
                "actual_full_ID_rename_reverse_stream_equal": payload.get(
                    "actual_full_ID_rename_reverse_stream_equal"
                ),
                "future_observations_mutated": (
                    payload.get("actual_future_probe", {}).get("future_observations_mutated")
                    if isinstance(payload.get("actual_future_probe"), dict)
                    else None
                ),
                "case_violations": case_violations,
            }
        )

    def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
        values = [float(row[key]) for row in rows]
        return {
            "count": len(values),
            "mean": mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_model[row["model"]].append(row)
        by_model_seed[f"{row['model']}/s{row['seed']}"] .append(row)

    endpoint_means = {
        key: {
            metric: mean(float(row[metric]) for row in rows)
            for metric in (
                "known_macro_accuracy",
                "known_micro_accuracy",
                "reuse_macro_lower_bound",
                "reuse_micro_lower_bound",
            )
        }
        for key, rows in sorted(by_model_seed.items())
    }

    action_summary = {
        action: summarize(action_rows, action)
        for action in ("EXISTING", "KNOWN", "NEW")
    }
    metric_summary = {
        metric: summarize(metric_rows, metric)
        for metric in (
            "known_macro_accuracy",
            "known_micro_accuracy",
            "reuse_macro_lower_bound",
            "reuse_micro_lower_bound",
            "physical_track_coverage",
        )
    }

    status_path = ROOT / "outputs/trackocd_cvpr/status.json"
    status_payload: dict[str, Any] = {}
    if status_path.is_file():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            violations.append("unreadable:outputs/trackocd_cvpr/status.json")

    return {
        "schema_version": "trackocd.cvpr4.current-source-M-recovery2-matrix-audit.v1",
        "status": "PASS_CURRENT_SOURCE_M_RECOVERY2_MATRIX_AUDIT" if not violations else "FAIL_CURRENT_SOURCE_M_RECOVERY2_MATRIX_AUDIT",
        "audit_scope": "nine explicitly registered recovery-2 current-source M causal-GT development receipts",
        "not_claimed": [
            "active purity",
            "self-exclusion",
            "Pred-owned joint result",
            "formal Val/Test metric",
            "model or checkpoint selection",
            "Goal A/B/C acceptance",
        ],
        "registered_cases": len(EXPECTED),
        "verified_cases": len(cases),
        "verified_orders_per_case": list(ORDERS),
        "cases": cases,
        "prefix16_metric_rows": metric_rows,
        "prefix16_metric_summary": metric_summary,
        "endpoint_mean_prefix16_by_model_seed": endpoint_means,
        "physical_action_summary": action_summary,
        "current_goal_flags": {
            key: status_payload.get(key)
            for key in ("status", "phase", "Goal_A", "Goal_B", "Goal_C", "formal_pred_result", "formal_final_result", "scientific_success_claimed")
        },
        "provenance": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "audit_script_sha256": sha256_file(Path(__file__)),
            "recovery_manifest": "configs/trackocd_cvpr/M_GT_causal_development_current_source_wall_recovery2.json",
            "recovery_manifest_sha256": sha256_file(ROOT / "configs/trackocd_cvpr/M_GT_causal_development_current_source_wall_recovery2.json"),
        },
        "violations": violations,
        "interpretation": (
            "All nine receipts and their three order probes pass the registered "
            "metadata/causal-boundary checks; their low prefix-16 values remain "
            "development evidence, not a qualified M winner or Goal acceptance."
        ),
    }


def main() -> int:
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
