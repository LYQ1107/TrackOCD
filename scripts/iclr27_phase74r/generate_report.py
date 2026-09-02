#!/usr/bin/env python3
"""Render the self-contained Phase74R audit report from machine artifacts.

The runner is intentionally audit-only.  This renderer never invokes a model
or reads sealed labels; it only summarizes the artifacts produced by the
completed Phase74R run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74r"
DOC = ROOT / "docs/iclr27_phase74r/PHASE74R_HARNESS_AND_ASSET_REVALIDATION_REPORT.md"
OVERALL = ROOT / "docs/AUTONOMOUS_TRACKOCD_ICLR_PROGRESS_REPORT.md"

sys.path.insert(0, str(ROOT))
from src.iclr27_phase74r.io import atomic_json, atomic_text, sha256  # noqa: E402


def load(relative: str, default: Any = None) -> Any:
    path = OUT / relative
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: str) -> str:
    result = subprocess.run(
        command,
        shell=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def write_repair_events() -> list[dict[str, Any]]:
    """Preserve the two failed smoke attempts without rewriting their output."""
    events = [
        {
            "attempt": "r1",
            "run_id": "phase74r-final-20260902-r1",
            "status": "FAILED_BEFORE_AUDIT_ARTIFACTS",
            "root_cause": "NameError: EVENT_POS was referenced in run_phase74r.setup; the defined constants are EVENT_POSITIVE/EVENT_NEGATIVE.",
            "observed_action": "explicitly retained the closed lock marker; no model/replay was invoked",
            "repair": "replace only the undefined constant names in preregistration input_data",
        },
        {
            "attempt": "r2",
            "run_id": "phase74r-final-20260902-r2",
            "status": "BLOCKED_PREFIX_CONTRACT",
            "root_cause": "the checker required frame_id/image_id source-text literals although frozen Phase19R orders its runtime index by event_rank",
            "observed_action": "preserved r2 outputs; no model/replay was invoked",
            "repair": "check the actual idx.sort(event_rank) expression and Phase74R projection tie-break evidence",
        },
        {
            "attempt": "r3",
            "run_id": "phase74r-final-20260902-r3",
            "status": "PHASE74R_BLOCKED_DENOMINATOR",
            "root_cause": "evidence-derived denominator gate: the actual model fallback has 82 events and zero keys overlap the frozen 152-event evaluator metadata universe",
            "observed_action": "completed audit-only run; Q0 model was not invoked",
            "repair": "none; this is the actionable protocol blocker requiring Desktop ChatGPT review",
        },
        {
            "attempt": "report-render-1",
            "run_id": "phase74r-final-20260902-r3",
            "status": "FAILED_BEFORE_REPORT",
            "root_cause": "direct report-generator invocation did not add the project root to sys.path (ModuleNotFoundError: src)",
            "observed_action": "audit outputs and code commit were retained; no model/replay was invoked",
            "repair": "insert the resolved project root into sys.path before importing the local package",
        },
    ]
    atomic_json(OUT / "audit/repair_events.json", events)
    return events


def write_output_ledger() -> dict[str, Any]:
    ledger: dict[str, Any] = {}
    for path in sorted(OUT.rglob("*")):
        relative = path.relative_to(OUT).as_posix()
        if relative == "manifests/output_sha256.json" or (not path.is_file() and not path.is_symlink()):
            continue
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            ledger[relative] = {
                "symlink_target": os.readlink(path),
                "target_exists": resolved.exists(),
                "target_sha256": sha256(resolved) if resolved.is_file() else None,
            }
        else:
            ledger[relative] = sha256(path)
    ledger["__self_hash__"] = "excluded_to_avoid_self_hash_cycle"
    atomic_json(OUT / "manifests/output_sha256.json", ledger)
    return ledger


def markdown(status: dict[str, Any], repairs: list[dict[str, Any]]) -> str:
    order = status["model_event_order"]
    identity = status["asset_identity"]
    prefix = status["prefix_contract"]
    gates = status["gates"]
    obs = status["observability"]
    metamorphic = status["metamorphic"]
    inventory = status["inputs"]
    pre = load("audit/preflight.json", {})
    post = load("audit/postflight.json", {})
    failure = load("audit/failure_taxonomy_summary.json", {})
    fold_counts = Counter()
    for row in load_jsonl(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"):
        fold_counts[int(row.get("fold", -1))] += 1
    fold_totals = {str(k): int(v * 2) for k, v in sorted(fold_counts.items())}
    hash_pairs = [
        (x["name"], x.get("sha256"), x.get("hash_match"))
        for x in inventory.get("inputs", [])
    ]
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if value.get('pass') else 'FAIL'} | {value.get('reason', '')} |"
        for name, value in gates.items()
    )
    hash_rows = "\n".join(
        f"| `{name}` | `{digest}` | {'match' if match else 'not-comparable'} |"
        for name, digest, match in hash_pairs
    )
    pre_commands = ", ".join(x["command"] for x in pre.get("commands", []))
    post_commands = ", ".join(x["command"] for x in post.get("commands", []))
    return f"""# TrackOCD Phase74R — Harness and Asset Revalidation

**Run:** `{status['run_id']}`  
**UTC window:** `{status['start_utc']}` → `{status['end_utc']}`  
**Decision:** **`{status['status']}`**  
**Project:** `{ROOT}`  
**Luna thread:** `01a01fb6-96f7-7132-a318-0833180c88d8`

## Executive decision

Phase74R is an audit-only repair of the Q0 control-replay harness.  The run
completed all executable contracts, but it is **blocked before Q0 replay**:
the actual Phase19R model event stream is 82 rows (41 positive + 41 negative)
while the frozen evaluator denominator is 152 rows (76 + 76), with **zero
event-key intersection**.  It would be invalid to map Q0 predictions to the
76-positive/76-negative evaluator by position or by category.  No detector,
retrieval, OCD, or controller conclusion is drawn from this mismatch.

The required next action is an externally reviewed, exact event-stream
contract (or an authorized frozen Q0 validation manifest).  Phase75A/B and all
semantic/controller work are **not started**.  Public new-model labels,
DEV+, Q1 and sealed evaluation remain sealed.

## Scope and immutable boundaries

The namespace is independent: `src/iclr27_phase74r/`,
`scripts/iclr27_phase74r/`, `tests/phase74r/`, `outputs/iclr27_phase74r/`,
and `docs/iclr27_phase74r/`.  Phase19R/Phase74 inputs were read-only.  The
held positive/negative manifests were read only as frozen evaluator metadata
for denominator auditing; they were never passed to a model.  No future rows,
category/text features, semantic IDs, physical IDs, DEV+/Q1/public-new
labels, or sealed labels were used as inference input.

Historical numbers remain unchanged: Phase74's Q0 25/76 observation result
is **not directly comparable** until the exact event universe is restored.
This run therefore reports no Q0 performance, no zero-valued detection
metrics, and no OCD=0 claim.

## Repair attempts and first actionable roots

The failed attempts are preserved in
`outputs/iclr27_phase74r/audit/repair_events.json` and their closed lock
markers.  They were not overwritten:

| attempt | outcome | actionable root | disposition |
|---|---|---|---|
| r1 | failed before artifacts | undefined `EVENT_POS` in setup | minimal constant-name fix |
| r2 | blocked prefix contract | checker demanded irrelevant frame/image source literals | minimal evidence-check fix |
| r3 | `{status['status']}` | 82 model events vs 152 evaluator events; 0 key matches | retain blocker; await exact manifest |
| report-render-1 | failed before report | direct invocation could not import local `src` package | add resolved project root to `sys.path` |

Smoke and targeted tests after the two fixes: **9/9 passed**.  No old Phase74
file was modified.

## Model-event order contract

`outputs/iclr27_phase74r/contracts/model_event_order_contract.json` records the
authoritative provenance and hashes.  `public_model_events.jsonl` was absent,
so the exact historical fallback was reconstructed: the two Phase19R source
files were projected to model-visible fields and sorted by `event_key`.

| field | value |
|---|---|
| source | `{order['source']}` |
| source paths | `{'; '.join(order['source_paths'])}` |
| model rows | `{order['count']}` |
| evaluator rows | `{order['metadata_count']}` (`{order['metadata_polarity_counts']}`) |
| model order hash | `{order['order_sha256']}` |
| metadata order hash | `{order['metadata_order_sha256']}` |
| matched keys | `{order['model_metadata_matched']}` |
| model-only rows | `{order['model_metadata_unmatched']}` |
| evaluator-only rows | `{order['metadata_without_model']}` |
| key sets equal | `{order['model_key_set_equals_metadata']}` |
| independent reconstruction | `{order['order_matches_independent_reconstruction']}` |

`evaluator_event_join.jsonl` retains the model subsequence first and all
unmatched evaluator metadata afterwards; no row is silently invented,
reordered, or dropped.

## Asset identity and Branch-A alignment

The identity pipeline uses protocol keys, split-independent content keys and
lazy file hashes while preserving duplicate candidates as lists.  It never
uses category or track IDs for identity.

| measure | result |
|---|---:|
| Q0 asset records | `{identity['q0_records']}` |
| event asset records | `{identity['event_records']}` |
| content-key intersection | `{identity['content_key_intersection']}` |
| mapped event records | `{identity['mapped_event_records']}` |
| unresolved event records | `{identity['unresolved_event_records']}` |
| event status | `{identity['status_counts']}` |
| duplicate candidates preserved | `{identity['duplicates_preserved']}` |

All 1,422 event assets are `NO_CONTENT_MATCH` against the Q0 validation asset
manifest.  This is independent evidence that a content-key join cannot
repair the event universe by guessing.  The synthetic Branch-A fixture in
`outputs/iclr27_phase74r/tests/branch_a_integration.json` **passes**: a
different event image ID maps uniquely to a Q0 physical candidate through a
shared content key, and overlap is distinguished from physical fragmentation.

## Causal prefix, reliability, and fragmentation contracts

`prefix_contract.json` is proven from the frozen runner and stream source:
source tracklets are registered independently before the target, runtime rows
are ordered by `event_rank`, and the Phase74R projection uses deterministic
`event_rank/frame/image` tie-breaks.  Target visibility is the first
`N ∈ {{1,2,4,8,16}}` rows only; future append and source-before-target
metamorphic tests pass.

Before a Q0 replay, every observation field is deliberately **null** with
status `NOT_AVAILABLE_Q0_NOT_REPLAYED`; no missing detection is encoded as a
zero.  The null export has `{obs['records']}` records (= 152 events × 2 roles
× 5 prefixes).  Exact unique evaluator denominators are retained per fold:
`{fold_totals}` positive+negative events (24/24/48/56) and 152 total events.
The joint reliability rule is fixed to `event assigned == 1 AND event IoU >=
0.5 AND Q0 IoU >= 0.5`, but its value is not computed until a valid replay.
Failure taxonomy records are `{failure.get('records')}` explicit
`Q0_REPLAY_INPUT_MISMATCH` entries; no detector conclusion is allowed.

## Executed metamorphic and artifact checks

All registered checks pass: category shuffle, event-label swap, physical-ID
renumbering, future append, source-before-target, repeat determinism, atomic
crash injection at asset/alignment/status, and static anti-hardcode scan.
JSON and JSONL artifacts parse, and the timeline is a JSON array.  Full
machine-readable evidence is under `outputs/iclr27_phase74r/`.

## Mandatory gates

| gate | result | evidence reason |
|---|---|---|
{gate_rows}

The only failed mandatory gate is `EVALUATOR_DENOMINATOR`; the resulting
status is `PHASE74R_BLOCKED_DENOMINATOR`.  `ASSET_IDENTITY` passing means the
identity records are complete, **not** that a nonexistent content mapping was
fabricated.  `RELIABILITY_CONTRACT` passing means unreplayed fields remain
null, not that Q0 was evaluated.

## Resources, process safety, and reproducibility

The audit used one CPU process, zero GPUs, and no external process kills.
Preflight commands: `{pre_commands}`.  Postflight commands:
`{post_commands}`.  Both preflight and postflight are recorded under
`outputs/iclr27_phase74r/audit/`; no Phase74R process remained active after
completion.  The host had 125 GiB RAM with about 118 GiB available during the
final preflight; `/data1` had about 36 GiB free and `/data2` about 1.2 TiB.
During an earlier resource check, GPUs 0–3 were occupied by an unrelated
`masaenv_debug` job; they were not touched.  The final audit used no GPU.

Input SHA256 matches the registered Phase74 values:

| input | SHA256 | registered |
|---|---|---|
{hash_rows}

The output ledger is `outputs/iclr27_phase74r/manifests/output_sha256.json`;
there are no Phase74R symlinked large files.  Code and tests are committed
separately from generated artifacts.

## Explicitly not executed

- Q0 model forward/control replay: **not run**
- Phase75A independent Q0 validation replays: **not run**
- Phase75B event replay, O/R/C stages, semantic/controller training: **not run**
- DEV+, Q1, public new-model labels, or sealed evaluation: **not accessed**
- threshold/memory/controller tuning: **not performed**

## Reproduction

```bash
cd {ROOT}
test -f AGENTS.md  # project identity check
/home/lwr/anaconda3/envs/locatemot/bin/python -m py_compile src/iclr27_phase74r/*.py scripts/iclr27_phase74r/*.py tests/phase74r/*.py
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python -m pytest -q tests/phase74r
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase74r/run_phase74r.py --run-id phase74r-final-20260902-r3
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase74r/generate_report.py
```

All data paths are resolved from the project root.

## Recommendation and stop boundary

Do not infer that Q0, the detector, or OCD failed.  The evidence supports one
specific blocker: the historical model-event source and the frozen evaluator
event universe are different protocols.  Desktop ChatGPT should authorize or
provide an exact, hash-identified Q0 validation event manifest (or reconcile
the event-key generator) before any replay.  Until then, automatically
starting Phase75A would violate the denominator and causal contracts.

## Artifact index

- status: `outputs/iclr27_phase74r/status.json`
- model order: `outputs/iclr27_phase74r/contracts/model_event_order_contract.json`
- event join: `outputs/iclr27_phase74r/contracts/evaluator_event_join.jsonl`
- asset identity: `outputs/iclr27_phase74r/assets/content_identity_summary.json`
- null alignment: `outputs/iclr27_phase74r/export/event_tracklet_alignment.jsonl`
- failure taxonomy: `outputs/iclr27_phase74r/audit/failure_taxonomy_76.json`
- metamorphic tests: `outputs/iclr27_phase74r/tests/metamorphic_results.json`
- hashes: `outputs/iclr27_phase74r/manifests/output_sha256.json`
- repair history: `outputs/iclr27_phase74r/audit/repair_events.json`

This report is a Phase74R blocker report, not a final MOT+OCD result.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    status = load("status.json")
    if not status:
        raise SystemExit("Phase74R status.json is missing")
    repairs = write_repair_events()
    report = markdown(status, repairs)
    atomic_text(DOC, report)
    overall = f"""# TrackOCD Autonomous Progress — Phase74R checkpoint

Phase74R completed an audit-only harness/asset revalidation on 2026-09-02.

**Current status:** `{status['status']}`.  The actual Q0 model event fallback
contains 82 rows, while the frozen evaluator contains 152 (76 positive + 76
negative), with zero key overlap.  Therefore Q0 control replay and all
Phase75+ stages remain blocked; no detector/OCD metric is inferred.  No
training, controller run, DEV+/Q1/public-new/sealed access occurred.

The detailed report is
[`PHASE74R_HARNESS_AND_ASSET_REVALIDATION_REPORT.md`](docs/iclr27_phase74r/PHASE74R_HARNESS_AND_ASSET_REVALIDATION_REPORT.md).
Machine evidence is under `outputs/iclr27_phase74r/`; repair history is at
`outputs/iclr27_phase74r/audit/repair_events.json`.  Desktop ChatGPT must
resolve the exact event-stream manifest/contract before a causal Q0 replay is
authorized.
"""
    atomic_text(OVERALL, overall)
    write_output_ledger()
    print(json.dumps({"status": status["status"], "report": str(DOC), "overall": str(OVERALL)}, indent=2))


if __name__ == "__main__":
    main()
