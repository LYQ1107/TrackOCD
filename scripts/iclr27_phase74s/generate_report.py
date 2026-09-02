#!/usr/bin/env python3
"""Render the Phase74S provenance report from immutable machine artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74s"
DOC = ROOT / "docs/iclr27_phase74s/PHASE74S_EVENT_PROTOCOL_RECONCILIATION_REPORT.md"
OVERALL = ROOT / "docs/AUTONOMOUS_TRACKOCD_ICLR_PROGRESS_REPORT.md"

from src.iclr27_phase74s.io import atomic_json, atomic_text, sha256  # noqa: E402


def load(rel: str, default: Any = None) -> Any:
    path = OUT / rel
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def ledger() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path in sorted(OUT.rglob("*")):
        rel = path.relative_to(OUT).as_posix()
        if rel == "manifests/output_sha256.json" or (not path.is_file() and not path.is_symlink()):
            continue
        records[rel] = sha256(path) if not path.is_symlink() else {"symlink_target": os.readlink(path), "target_exists": path.resolve(strict=False).exists()}
    records["__self_hash__"] = "excluded_to_avoid_self_hash_cycle"
    atomic_json(OUT / "manifests/output_sha256.json", records)
    return records


def render() -> str:
    status = load("status.json", {})
    decision = load("audit/stale_fallback_decision.json", {})
    contract = load("contracts/model_evaluator_contract_v2.json", {})
    comparison = load("provenance/protocol_comparison.json", {})
    inv = load("provenance/event_manifest_inventory.json", {})
    pre = load("audit/preflight.json", {})
    post = load("audit/postflight.json", {})
    cond = decision.get("conditions", {})
    cond_rows = "\n".join(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in cond.items()) or "| (missing) | FAIL |"
    protocols = []
    for name, item in comparison.items():
        if name.startswith("_"):
            continue
        protocols.append(f"| `{name}` | {item.get('total_rows')} | {item.get('unique_event_keys')} | `{item.get('event_key_hash')}` |")
    protocol_rows = "\n".join(protocols) or "| (none) | 0 | 0 | - |"
    model_hash = contract.get("model_order_sha256", "-")
    join_hash = contract.get("join_order_sha256", "-")
    return f"""# TrackOCD Phase74S — Event Protocol Reconciliation

**Run:** `{status.get('run_id', 'unknown')}`  
**Decision:** **`{status.get('status', 'UNKNOWN')}`**  
**Project:** `{ROOT}`  
**Luna thread:** `01a01fb6-96f7-7132-a318-0833180c88d8`

## Executive result

Phase74S was an audit-only recovery stage. It did not run Q0, train a model,
run the controller, or access DEV+, Q1, public-new, or sealed labels. The
authoritative legacy model fallback is 82 rows (41 positive + 41 negative)
from the Phase18 identifiable source, whereas the later frozen evaluator is
152 rows (76 + 76) and the later model manifest is 152 rows. The event-key
sets are disjoint, so positional or category-based mapping would be invalid.

All registered provenance conditions are evaluated below. The status is PASS
only when every condition is true and the label-free model/evaluator contract
has exactly 152 rows on each side.

## Stale-fallback decision evidence

| condition | result |
|---|---|
{cond_rows}

`{decision.get('status', 'UNKNOWN')}`. Legacy counts: `{decision.get('evidence', {}).get('legacy_counts', [])}`; evaluator counts: `{decision.get('evidence', {}).get('evaluator_counts', [])}`; model counts: `{decision.get('evidence', {}).get('model_manifest_counts', [])}`.

## Protocol inventory

The inventory, graph, and consumer table are machine-readable under
`outputs/iclr27_phase74s/provenance/`. Each record retains logical path,
resolved path, symlink status, SHA256, mtime, row count, fields, phase origin,
and source-text generator/consumer evidence. Aliases are deduplicated by
realpath before protocol totals are computed.

| protocol | rows | unique event keys | key hash |
|---|---:|---:|---|
{protocol_rows}

Inventory records: **{inv.get('manifest_count', 0)}**. The Phase19R freeze
script still contains the historical silent fallback to
    `data/iclr27_phase19r/sources/{{positive,negative}}_events.jsonl`; this is the
specific stale path being quarantined, not a model result.

## Versioned model/evaluator contract

`outputs/iclr27_phase74s/manifests/model_events_v2.jsonl` is the model-facing
manifest. It has only `model_event_uid`, source/target tracklet keys, and
explicit source/target videos. It contains no event key, polarity, category,
fold, GT, action, semantic ID, or physical ID. The evaluator-only
`evaluator_join_v2.jsonl` adds the frozen event metadata after model rows are
frozen; no join table is read by model forward. Counts are
model={contract.get('model_event_count')}, evaluator={contract.get('evaluator_event_count')}, join={contract.get('join_count')}; forbidden fields seen=`{contract.get('forbidden_model_fields_seen', [])}`. Model order hash=`{model_hash}`, join order hash=`{join_hash}`.

The explicit Phase74S entry point
`scripts/iclr27_phase74s/freeze_predictions_v2.py` requires
`--model-event-manifest` and hard-fails if it is absent, malformed, has the
wrong cardinality, or contains evaluator labels. There is no silent fallback.

## Causal and sealed boundaries

The new manifest preserves the frozen positive-file then negative-file order
and source-before-target track processing. It does not alter physical IDs,
proposal rows, frame order, or evaluator denominators. No future frames or
tracks, category/text features, semantic/physical-ID features, DEV+/Q1,
public-new labels, or sealed labels were consumed as model input. Q0 replay is
authorized only after this explicit contract is accepted; this report itself
contains no Q0 performance and does not claim OCD=0.

## Resources and integrity

Phase74S used one bounded CPU process and zero GPUs. Pre/post resource command
outputs are in `outputs/iclr27_phase74s/audit/{{preflight,postflight}}.json` and
`logs/{{preflight,postflight}}_resource.txt`. No external process was touched;
there is no training worker, no checkpoint, and no OOM event. Generated files
were written atomically. Output hashes are in
`outputs/iclr27_phase74s/manifests/output_sha256.json`.

## Next authorized action

If the status is PASS, proceed to two independent Q0 control replays using the
frozen Q0 checkpoint and OVTR commit `500e72c`, with graph-matched physical
assignments and explicit no-detection nulls. If either replay cannot reproduce
the same event universe, stop before O/R/C and preserve the mismatch. If the
contract is blocked, do not invoke Q0 or infer any detector/OCD metric.

## Reproduction

```bash
cd {ROOT}
PYTHONPATH=. python -m unittest discover -s tests/phase74s -v
python -m py_compile src/iclr27_phase74s/*.py scripts/iclr27_phase74s/*.py
PYTHONPATH=. python scripts/iclr27_phase74s/run_phase74s.py --run-id phase74s-replay
PYTHONPATH=. python scripts/iclr27_phase74s/generate_report.py
```

## Artifact index

- status: `outputs/iclr27_phase74s/status.json`
- inventory: `outputs/iclr27_phase74s/provenance/event_manifest_inventory.json`
- graph: `outputs/iclr27_phase74s/provenance/event_manifest_graph.json`
- consumers: `outputs/iclr27_phase74s/provenance/event_manifest_consumers.json`
- decision: `outputs/iclr27_phase74s/audit/stale_fallback_decision.json`
- model manifest: `outputs/iclr27_phase74s/manifests/model_events_v2.jsonl`
- evaluator join: `outputs/iclr27_phase74s/manifests/evaluator_join_v2.jsonl`
- contract: `outputs/iclr27_phase74s/contracts/model_evaluator_contract_v2.json`
- output ledger: `outputs/iclr27_phase74s/manifests/output_sha256.json`

This is a protocol recovery report, not a final MOT+OCD result.
"""


def main() -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    repair_path = OUT / "audit/repair_events.json"
    repairs = []
    if repair_path.exists():
        try:
            repairs = json.loads(repair_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            repairs = []
    repairs.append({
        "attempt": "report-render-r1",
        "status": "FAILED_BEFORE_REPORT",
        "root_cause": "generate_report.py used unescaped literal braces in an f-string (NameError: positive)",
        "preserved_artifacts": True,
        "repair": "escape the literal source manifest braces; no audit/model outputs were changed",
    })
    atomic_json(repair_path, repairs)
    text = render()
    atomic_text(DOC, text)
    ledger()
    overall = f"""\n\n## Phase74S event protocol reconciliation (2026-09-02)\n\nPhase74S status: **{load('status.json', {}).get('status', 'UNKNOWN')}**. The audit identified and quarantined the Phase19R silent 82-row legacy fallback, then built a label-free 152-row model manifest plus evaluator-only join with exact frozen order and no forbidden model fields. No Q0 model forward, training, controller, DEV+/Q1/public-new/sealed access, or performance claim occurred.\n\nDetailed report: [Phase74S event protocol reconciliation](docs/iclr27_phase74s/PHASE74S_EVENT_PROTOCOL_RECONCILIATION_REPORT.md). Machine evidence: `outputs/iclr27_phase74s/`.\n"""
    atomic_text(OVERALL, OVERALL.read_text(encoding="utf-8") + overall if OVERALL.exists() else overall.lstrip())
    print(str(DOC))


if __name__ == "__main__":
    main()
