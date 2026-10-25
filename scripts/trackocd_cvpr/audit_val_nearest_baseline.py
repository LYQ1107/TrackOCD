#!/usr/bin/env python3
"""Extract a compact receipt for the sealed TAO Val nearest OCD baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.val-nearest-ocd-baseline.v1"
TABLE = "outputs/trackocd_v2_runtime/tables/pred_nearest.json"
TRUTH_AUDIT = "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit(repo: Path) -> dict[str, Any]:
    table_path = repo / TABLE
    truth_path = repo / TRUTH_AUDIT
    table = json.loads(table_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    nearest = truth.get("ocd_nearest") or {}
    evaluator_subset = table.get("evaluator_subset") or {}
    model_input = table.get("model_input_contract") or {}
    parameters = table.get("fixed_method_parameters") or {}
    prefix_metrics = nearest.get("prefix_metrics") or {}
    source_safe = (
        table.get("method") == "nearest"
        and table.get("prefixes") == [1, 2, 4, 8, 16]
        and nearest.get("status") == "SEALED_EXISTING_VAL_BASELINE"
        and nearest.get("test_semantic_accessed") is False
        and nearest.get("method_selection_performed") is False
        and evaluator_subset.get("gt_matching_used_as_model_input") is False
        and evaluator_subset.get("historical_matched_stream_used_for_inference") is False
        and model_input.get("gt_category_id") is False
        and model_input.get("gt_split") is False
        and model_input.get("temporal_iou_match") is False
        and model_input.get("text_or_category_logits") is False
    )
    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SEALED_VAL_NEAREST_OCD_BASELINE" if source_safe else "INCOMPLETE_VAL_NEAREST_BASELINE",
        "scope": {
            "val_only": True,
            "sealed_existing_result": True,
            "new_inference_started": False,
            "training_started": False,
            "tao_test_read": False,
            "model_selection_performed": False,
            "external_processes_touched": False,
        },
        "source": {
            "table_path": TABLE,
            "table_bytes": table_path.stat().st_size,
            "table_sha256": sha256_file(table_path),
            "truth_audit_path": TRUTH_AUDIT,
            "truth_audit_bytes": truth_path.stat().st_size,
            "truth_audit_sha256": sha256_file(truth_path),
            "frontend": nearest.get("frontend"),
            "source_kind": nearest.get("source_kind"),
        },
        "protocol": {
            "method": table.get("method"),
            "prefixes": table.get("prefixes"),
            "fixed_gt_denominator": nearest.get("fixed_gt_denominator"),
            "post_decision_join_rows": nearest.get("post_decision_join_rows"),
            "unmatched_gt_deferred": nearest.get("unmatched_gt_deferred"),
            "evaluator_subset": {
                key: evaluator_subset.get(key)
                for key in (
                    "fixed_gt_denominator_rows",
                    "public_decisions_filtered_only_after_causal_pass",
                )
            },
            "fixed_method_parameters": parameters,
            "model_input_contract": model_input,
        },
        "prefix_metrics_fraction": {
            prefix: {
                key: metrics.get(key)
                for key in (
                    "all_acc",
                    "old_acc",
                    "new_acc",
                    "h_score",
                    "commit_ct",
                    "false_assignment_rate",
                )
            }
            for prefix, metrics in sorted(
                prefix_metrics.items(), key=lambda item: int(str(item[0]).lstrip("p"))
            )
        },
        "interpretation": nearest.get("interpretation"),
        "qualification_boundary": {
            "is_formal_frontend_selection": False,
            "is_predicted_track_training": False,
            "is_goal_a_b_c_acceptance": False,
            "is_new_rerun": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/val_nearest_ocd_metrics_receipt_20261026.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
