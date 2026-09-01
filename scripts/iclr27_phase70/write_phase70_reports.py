#!/usr/bin/env python3
"""Generate the Phase70 integration/final reports from frozen artifacts.

Numbers are read from the immutable validation aggregate and the checkpoint
validation gate.  The writer never reads sealed labels and never runs an
evaluator; it records blocked downstream stages explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
VAL = ROOT / "outputs/iclr27_phase70/validation/joint_d_repair1"
STAGE = VAL / "step_5000_metrics.json"
CONTRACT = ROOT / "outputs/iclr27_phase70/audit/phase70_contract.json"
PHASE69 = ROOT / "outputs/iclr27_phase69/metrics/phase69_aggregate.json"
Q0_RECALL = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/proposal_recall.json"
Q0_TE = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json"
DECISION = ROOT / "outputs/iclr27_phase70/final_decision.json"
INTEGRATION = ROOT / "docs/iclr27_phase69/PHASE70_OVTR_OCD_INTEGRATION_REPORT.md"
FINAL = ROOT / "docs/iclr27_phase70/PHASE70_MOT_OCD_FINAL_EVALUATION_REPORT.md"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: pathlib.Path, obj: object) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def fmt(v: object, digits: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def load() -> tuple[dict, dict, dict, dict, dict]:
    return (
        json.loads(STAGE.read_text()),
        json.loads(CONTRACT.read_text()),
        json.loads(PHASE69.read_text()),
        json.loads(Q0_RECALL.read_text()),
        json.loads(Q0_TE.read_text()),
    )


def fold_table(stage: dict) -> str:
    lines = [
        "| fold | updates | pred rows | top20 R@.3 | top20 R@.5 | top20 R@.7 | mean best IoU | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage["folds"]:
        te = row["trackeval_macro"]
        lines.append(
            f"| {row['fold']} | {row['checkpoint_steps']} | {row['prediction']['count']} | "
            f"{fmt(row['proposal_top20_recall']['0.3'])} | {fmt(row['proposal_top20_recall']['0.5'])} | "
            f"{fmt(row['proposal_top20_recall']['0.7'])} | {fmt(row['proposal_top20_mean_best_iou'])} | "
            f"{fmt(te['HOTA'])} | {fmt(te['DetA'])} | {fmt(te['AssA'])} | {fmt(te['MOTA'])} | "
            f"{fmt(te['IDF1'])} | {fmt(te['IDSW'])} | {fmt(te['Frag'])} |"
        )
    a = stage["aggregate"]
    lines.append(
        f"| **mean** | — | — | {fmt(a['top20_recall_iou03'])} | {fmt(a['top20_recall_iou05'])} | "
        f"{fmt(a['top20_recall_iou07'])} | {fmt(a['top20_mean_best_iou'])} | "
        f"{fmt(a['trackeval_macro_HOTA'])} | {fmt(a['trackeval_macro_DetA'])} | {fmt(a['trackeval_macro_AssA'])} | "
        f"{fmt(a['trackeval_macro_MOTA'])} | {fmt(a['trackeval_macro_IDF1'])} | {fmt(a['trackeval_macro_IDSW'])} | "
        f"{fmt(a['trackeval_macro_Frag'])} |"
    )
    return "\n".join(lines)


def write_reports(stage: dict, contract: dict, phase69: dict, q0: dict, q0te: dict) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    a = stage["aggregate"]
    q0_top = q0["recall"]["topk"]["20"]["thresholds"]
    q0m = q0te["macro"]
    checkpoints = "\n".join(
        f"- fold{item['fold']}: `{item['path']}`; updates={item['steps']}; bytes={item['bytes']}; SHA256 `{item['sha256']}`; completion marker `{item['completion_marker']}`"
        for item in stage["checkpoints"]
    )
    table = fold_table(stage)
    common = f"""# Phase70 — OVTR OCD integration and MOT sanity validation

**Generated:** {now}  
**Project:** `{ROOT}`  
**Route:** frozen Phase69 OVTR/DSCT physical initialization → Phase70 semantic_b → assign/create c → joint d repair1  
**Status:** **STOPPED BEFORE OCD** after the mandatory TRAIN-disjoint proposal/MOT sanity gate.

## Executive decision

The repair1 workers completed and produced four non-empty checkpoints, but the
first full-sequence validation is not a usable MOT candidate.  The aggregate
top-20 IoU≥0.5 recall is **{a['top20_recall_iou05']:.6f}**, versus the frozen Q0
reference **{q0_top['0.5']['recall']:.6f}**; macro HOTA/DetA/AssA/IDF1 are
`{a['trackeval_macro_HOTA']:.6f}/{a['trackeval_macro_DetA']:.6f}/{a['trackeval_macro_AssA']:.6f}/{a['trackeval_macro_IDF1']:.6f}`
versus Q0 `{q0m['HOTA']:.6f}/{q0m['DetA']:.6f}/{q0m['AssA']:.6f}/{q0m['IDF1']:.6f}`.
This fails the preregistered sanity gate by a large margin.  Consequently no
longer training, checkpoint selection, retrieval claim, controller replay or
sealed evaluation is legal for this route.

Decision code: **`P70_GATE_MOT_SANITY_FAIL_STOP_BEFORE_OCD`**.

## Frozen contract and data boundary

- Phase70 contract: `{contract['protocol']}`; causal prefixes `{contract['causal_prefixes']}`;
  event comparator contains 76 positive and 76 negative events, but these event
  labels were not model inputs.
- Inference inputs are restricted to RGB/query representation, bbox geometry,
  motion/history, causal quality and support metadata.  Category/text names,
  semantic IDs, physical IDs as features, future frames/tracks, held GT,
  DEV+/Q1/public new-model labels and controller actions are forbidden.
- Physical proposal/tracker was frozen for this route; semantic outputs cannot
  mutate physical parent assignment.  The exact 768-D bridge/raw fallback and
  Phase19R controller paths are recorded in `{CONTRACT}`.
- Validation annotations are used only for post-hoc TRAIN-disjoint scoring.  No
  public/Q1/DEV+/sealed label was accessed.

## Phase67–69 context

- Phase67 selected the local Q0 OVTR asset lineage (upstream commit
  `500e72c19bf5f7f8717546911a5639fdc26bfee5`, MIT) rather than another
  from-scratch detector.  Official/local checkpoint hashes and CLIP/text
  isolation are in the Phase67 report and `outputs/iclr27_phase67/audit/ovtr_assets.json`.
- Phase68 reproduced the real Q0 full-sequence stream (1,268,113 rows) with
  top-20 IoU≥0.5 recall `0.629993` and complete TAO/TrackEval artifacts.
- Phase69 adapted Q0 for class-agnostic DSCT over seven epochs × 15,000
  iterations/fold, but averaged top-20 IoU≥0.5 recall `0.067490` and macro
  HOTA `0.050312` (Gate M69 FAIL).  It remains a frozen diagnostic
  initialization, not a successful MOT baseline.

## Phase70 training and recovery

The route kept the OVTR physical graph and added the registered semantic/state
stages; it did not introduce a new backbone or detector.  Formal repair1 used
batch 1, one worker, GPUs 4–7 (one fold per GPU), and the inherited Q0/Phase69
initialization.  f0/f1/f3 stopped at 4,000 updates and f2 at 5,000 updates;
the worker logs report 48:01–60:49 wall time.  DSCT/semantic gradients and
losses were logged, but they are not evidence of MOT validity.

Final checkpoints (all four `.done` markers are present):

{checkpoints}

The first formal supervisor had a `/data1` zero-free-space write failure;
intermediate semantic_b/assign_c checkpoints were moved (not copied) to
`/home/user/trackocd_phase70_archive/checkpoints/` and symlinked back.  The
invalid `--dsct_stage joint_d` smoke and the corrected `d` smoke/targeted
markers are retained.  A prior validation launch path error is retained in
`outputs/iclr27_phase70/completion/joint_d_repair1_validation_path_error.json`;
the corrected validation then completed atomically.  No OOM/swap event was
observed.  An accidental duplicate analyzer PID was explicitly SIGTERM-ed
after process-tree verification; no broad kill or external process touch
occurred.

## Stage validation: complete full-sequence table

The validation uses the immutable OVTR validation annotations (112,798 rows)
and emits a real full-sequence prediction per fold.  TrackEval is the pinned
per-class TAO adapter; macro and count summaries are preserved in
`validation_aggregate.json` and are not confused with TrackOCD Commit-CT.

{table}

Aggregate source: `{VAL / 'validation_aggregate.json'}`.  The immutable atomic
stage artifact is `{STAGE}` with marker `{VAL / 'step_5000.done'}`.

## Sanity gate and downstream statuses

| check | result |
|---|---:|
| proposal top20 IoU≥0.5 non-degradation vs Q0 | **FAIL** (`{a['top20_recall_iou05']:.6f}` vs `{q0_top['0.5']['recall']:.6f}`) |
| HOTA/DetA/AssA/IDF1 non-degradation vs Q0 | **FAIL** |
| all fold prediction streams/TrackEval artifacts | PASS (diagnostic completeness only) |
| retrieval R70 | **NOT RUN** — blocked by proposal/MOT sanity |
| controller positive/negative causal events | **NOT RUN** — blocked by proposal/MOT sanity |
| sealed/public evaluation | **NOT RUN** |

The required positive/negative action, false-commit, premature, unresolved and
known/novel fields are therefore `NOT_RUN`, not zero and not inferred from
training loss.  Running the frozen controller on this invalid physical stream
would conflate a detector failure with OCD behavior and violate the requested
checkpoint-selection rule.

## Reproduction commands

```bash
cd {ROOT}
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase70/write_stage_validation.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase70/aggregate_validation.py
bash scripts/iclr27_phase70/run_full_sequence_eval_repair1.sh
bash scripts/iclr27_phase70/run_trackeval_validation.sh
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase70/write_phase70_reports.py
```

These commands reuse the recorded checkpoints and validation annotations;
they do not read sealed labels or launch another training run.

## Final route decision

The Phase70 semantic/controller route is **not** an end-to-end MOT+OCD result.
It is stopped at the first actionable physical sanity failure.  The evidence
does not show that the OVTR architecture is universally unsuitable; it shows
that this Phase69-initialized joint repair collapsed the full-sequence proposal
stream.  A future route must first restore a validated class-agnostic detector
and association/lifecycle baseline (with real full-sequence TrackEval) before
semantic/controller experiments.  No threshold, StateMemory, controller or
modern-backbone lottery is justified by this failed checkpoint.
"""

    integration = common.replace(
        "# Phase70 — OVTR OCD integration and MOT sanity validation",
        "# Phase70 — OVTR OCD integration report",
        1,
    )
    final = common.replace(
        "# Phase70 — OVTR OCD integration and MOT sanity validation",
        "# Phase70 — MOT+OCD final evaluation report",
        1,
    )
    final += """

## Gate summary

- **M67 lineage:** PASS (read-only provenance audit).
- **M68 Q0 full-sequence reproduction:** PASS as a historical MOT comparator.
- **M69 class-agnostic adaptation:** FAIL (recall and TrackEval degradation).
- **M70 repair1 proposal/MOT sanity:** **FAIL**.
- **R70 correspondence:** NOT RUN because the physical candidate failed sanity.
- **C70 causal controller:** NOT RUN; no Commit-CT is claimed.
- **S70 sealed:** NOT RUN; public/Q1/DEV+ labels remain sealed.

This is a negative Phase70 route result, not a claim that the complete TrackOCD
task is mathematically impossible.  It does establish that the current
Phase69-initialized semantic/state repair cannot be selected for longer
training or OCD evaluation under the fixed causal protocol.
"""
    atomic_text(INTEGRATION, integration)
    atomic_text(FINAL, final)

    decision = {
        "phase": 70,
        "decision": "P70_GATE_MOT_SANITY_FAIL_STOP_BEFORE_OCD",
        "generated": now,
        "training": {
            "tag": "joint_d_repair1",
            "fold_updates": {"0": 4000, "1": 4000, "2": 5000, "3": 4000},
            "gpu_map": {"fold0": 4, "fold1": 5, "fold2": 6, "fold3": 7},
            "batch_size": 1,
            "num_workers": 1,
            "checkpoints": stage["checkpoints"],
            "completion_markers": [str(ROOT / f"outputs/iclr27_phase70/completion/joint_d_repair1_f{i}.done") for i in range(4)],
        },
        "validation": {
            "artifact": str(STAGE),
            "artifact_sha256": sha256(STAGE),
            "done_marker": str(VAL / "step_5000.done"),
            "aggregate": a,
            "per_fold": [
                {
                    "fold": row["fold"],
                    "updates": row["checkpoint_steps"],
                    "prediction": row["prediction"],
                    "top20_recall": row["proposal_top20_recall"],
                    "trackeval_macro": row["trackeval_macro"],
                    "trackeval_count_sums": row["trackeval_count_sums"],
                }
                for row in stage["folds"]
            ],
            "q0_top20_recall_iou05": q0_top["0.5"]["recall"],
            "q0_trackeval_macro": {k: q0m[k] for k in ("HOTA", "DetA", "AssA", "IDF1")},
            "proposal_top20_iou05_non_degraded": False,
            "mot_non_degraded": False,
        },
        "gates": {
            "M67_lineage": "PASS",
            "M68_q0_full_sequence": "PASS_COMPARATOR",
            "M69_class_agnostic_adaptation": "FAIL",
            "M70_proposal_mot_sanity": "FAIL",
            "R70_correspondence": "NOT_RUN",
            "C70_controller": "NOT_RUN",
            "S70_sealed": "NOT_RUN",
        },
        "downstream": {
            "retrieval": "NOT_RUN_BLOCKED_BY_PROPOSAL_SANITY",
            "controller": "NOT_RUN_BLOCKED_BY_PROPOSAL_SANITY",
            "persistent_commit_ct": "NOT_RUN",
            "sealed_public_q1_accessed": False,
            "held_gt_used_as_model_input": False,
            "labels_used_for_model": False,
        },
        "resources": {
            "gpu_map": "fold0:4 fold1:5 fold2:6 fold3:7",
            "ram_policy": ">=25% headroom; preflights retained",
            "disk_event": "temporary /data1 zero-free-space write failure; intermediates archived to /home/user/trackocd_phase70_archive and symlinked back",
            "oom": False,
            "external_process_kill": False,
            "task_owned_duplicate_analyzer_pid": 27529,
            "task_owned_duplicate_action": "SIGTERM after process-tree verification; PID 26618 retained",
        },
        "lineage": {
            "phase67_assets": str(ROOT / "outputs/iclr27_phase67/audit/ovtr_assets.json"),
            "phase68_baseline": str(ROOT / "outputs/iclr27_phase68/audit/full_sequence_baseline.json"),
            "phase69_aggregate": str(PHASE69),
            "phase70_contract": str(CONTRACT),
            "symlink_archive": "/home/user/trackocd_phase70_archive/checkpoints",
        },
        "next_action": "Do not extend joint_d_repair1 and do not run semantic/controller/sealed evaluation; next authorized work must establish a validated pretrained class-agnostic detector plus physical association/lifecycle baseline.",
    }
    atomic_json(DECISION, decision)
    print(json.dumps({"integration": str(INTEGRATION), "final": str(FINAL), "decision": str(DECISION), "decision_code": decision["decision"]}, indent=2))


if __name__ == "__main__":
    stage, contract, phase69, q0, q0te = load()
    write_reports(stage, contract, phase69, q0, q0te)
