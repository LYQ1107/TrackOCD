#!/usr/bin/env python3
"""Bounded CPU diagnostic of an inspected historical source, not a repair.

Only the AST-matched empty-memo score statements are executed. No upstream
module, checkpoint, annotation, tracker or GPU job is imported or launched.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
TRACKER_SHA256 = "0d68554296749207625bbbe8b828176bba873fa9a031ccaf1cf9dc0f9434d698"
EXPECTED_EMPTY_PREFIX = """conf_list = bboxes[:, 4]
for idx, item1 in enumerate(conf_list):
    conf_list[idx] = 0.7501
init_inds = (ids == -2) & (bboxes[:, 4] > self.init_score_thr).cpu()
"""


def diagnose_source(source: str) -> dict:
    """Check and run only three fixed statements; fail on any AST deviation."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IDOL_Tracker"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one IDOL_Tracker")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "match"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one match method")
    empty = [n for n in ast.walk(methods[0]) if isinstance(n, ast.If)
             and ast.dump(n.test) == ast.dump(ast.parse("self.empty", mode="eval").body)]
    if len(empty) != 1:
        raise ValueError("Expected exactly one empty-memo branch")
    module = ast.Module(body=empty[0].body[:3], type_ignores=[])
    if ast.dump(module) != ast.dump(ast.parse(EXPECTED_EMPTY_PREFIX)):
        raise ValueError("Inspected source statements changed; refuse execution")
    bboxes = torch.zeros((3, 5), dtype=torch.float32, device="cpu")
    bboxes[:, 4] = torch.tensor([0.01, 0.19, 0.9])
    before = bboxes[:, 4].clone()
    scope = {"bboxes": bboxes, "ids": torch.full((3,), -2, dtype=torch.long),
             "self": SimpleNamespace(init_score_thr=0.2)}
    exec(compile(module, "<verified-empty-memo-prefix>", "exec"),
         {"__builtins__": {"enumerate": enumerate, "__import__": builtins.__import__}}, scope)
    alias = scope["conf_list"].untyped_storage().data_ptr() == bboxes.untyped_storage().data_ptr()
    probabilities = [0.0, 0.01, 0.19, 0.9, 1.0]
    recompressed = [1 / (1 + math.exp(-p)) for p in probabilities]
    return {
        "schema_version": "trackocd.core.simowt_score_contract.v1",
        "status": "PASS_ENGINEERING_STATIC_DIAGNOSTIC_NOT_FRONTEND_QUALIFICATION",
        "execution_scope": "Three AST-verified statements on three synthetic CPU candidates",
        "device": "cpu",
        "source_empty_branch_first_line": empty[0].lineno,
        "initial_scores": before.tolist(),
        "scores_after_empty_branch": bboxes[:, 4].tolist(),
        "score_column_and_conf_list_share_storage": alias,
        "initial_candidates_passing_0_2": int((before > 0.2).sum()),
        "candidates_passing_0_2_after_source_prefix": int(scope["init_inds"].sum()),
        "already_probability_examples": probabilities,
        "after_second_sigmoid": recompressed,
        "second_sigmoid_range_on_unit_interval": [0.5, 1 / (1 + math.exp(-1))],
        "all_examples_exceed_registered_select_0_1": all(s > 0.1 for s in recompressed),
        "all_examples_exceed_registered_addnew_0_2": all(s > 0.2 for s in recompressed),
        "interpretation": "These gates cannot reject low raw probabilities after recompression; mask NMS and association still operate",
        "historical_stream_scores_or_fragmentation_cause_verified": False,
        "frozen_stream_mutated": False,
        "frontend_inference_or_training_started": False,
        "test_annotation_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/trackocd_core/audit/nas_simowt_source/tracker.py")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trackocd_core/audit/simowt_score_contract.json")
    args = parser.parse_args()
    if args.source.stat().st_size > 65536:
        raise ValueError("Refuse an unexpectedly large source")
    payload = args.source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TRACKER_SHA256:
        raise ValueError("NAS tracker source checksum mismatch")
    result = diagnose_source(payload.decode("utf-8"))
    result.update({"source_sha256": digest, "source_bytes": len(payload)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
