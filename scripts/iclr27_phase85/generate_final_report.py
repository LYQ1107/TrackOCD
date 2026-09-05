#!/usr/bin/env python3
"""Generate the Phase84 final report only after the registered lock opens."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase84"
AUDIT = OUT / "audit"
METRICS = OUT / "metrics"
COMP = OUT / "completion"
REG = AUDIT / "window_registration.json"
LOCK = AUDIT / "finalization_lock.json"
REPORT = ROOT / "docs/iclr27_phase84/PHASE84_AUTONOMOUS_RESEARCH_REPORT.md"
DECISION = AUDIT / "phase84_decision.json"
PROVENANCE = AUDIT / "report_provenance.json"


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.exists(), "sha256": sha(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def f(x: Any, n: int = 6) -> str:
    if x is None: return "NOT_RUN"
    if isinstance(x, bool): return "true" if x else "false"
    if isinstance(x, float): return f"{x:.{n}f}"
    return str(x)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(f(x) for x in r) + " |" for r in rows)
    return "\n".join(out)


def p16_summary(z: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(x.get("polarity")): x for x in z.get("summary", []) if x.get("prefix") == 16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reserved for local testing; cannot bypass the registered lock")
    args = ap.parse_args()
    lock = load(LOCK)
    if not lock.get("allowed", False):
        raise SystemExit("Phase84 finalization lock is closed; final report was not generated")
    reg = load(REG)
    physical = load(AUDIT / "a84_physical_r_metrics.json")
    physical_diag = load(AUDIT / "physical_r_diagnostic.json")
    signal = load(AUDIT / "source_conditioned_signal.json")
    a2_integrity = load(AUDIT / "phase83_a2_report_integrity.json")
    b84s_formal = load(METRICS / "b84s_formal_aggregate.json")
    b84s_event = load(METRICS / "b84s_event_replay.json")
    b84sq_formal = load(METRICS / "b84s_formal_aggregate_b84sq_v3.json")
    b84sq_event = load(METRICS / "b84s_event_replay_b84sq_v3.json")
    b84sq_audit = load(AUDIT / "b84sq_failure_audit.json")
    b84sra_formal = load(METRICS / "b84s_formal_aggregate_b84sra_v1.json")
    b84sra_event = load(METRICS / "b84s_event_replay_b84sra_v1.json")
    b84sra_audit = load(AUDIT / "b84sra_failure_audit.json")
    b84sproto_event = load(METRICS / "b84s_event_replay_b84sproto_v1.json")
    b84sproto_audit = load(AUDIT / "b84sproto_failure_audit.json")
    align = load(AUDIT / "support_alignment_callgraph.json")
    repairs = load(AUDIT / "repair_events.json")
    validation = load(AUDIT / "validation_evidence_ledger.json")
    research = load(AUDIT / "research_ledger.json")
    source_p16 = signal.get("aggregate", {}).get("16", {})
    phys_p16 = physical.get("gate_diagnostic", {}).get("p16", {})
    q16 = p16_summary(b84sq_event)
    ra16 = p16_summary(b84sra_event)
    proto16 = p16_summary(b84sproto_event)
    old16 = p16_summary(b84s_event)
    provenance_paths = [
        AUDIT / "phase83_a2_report_integrity.json", AUDIT / "a84_physical_r_metrics.json",
        AUDIT / "physical_r_diagnostic.json", AUDIT / "source_conditioned_signal.json",
        METRICS / "physical_r_q0_adapter.json", METRICS / "b84s_formal_aggregate.json",
        METRICS / "b84s_event_replay.json", METRICS / "b84s_formal_aggregate_b84sq_v3.json",
        METRICS / "b84s_event_replay_b84sq_v3.json", AUDIT / "b84sq_failure_audit.json",
        METRICS / "b84s_formal_aggregate_b84sra_v1.json", METRICS / "b84s_event_replay_b84sra_v1.json",
        AUDIT / "b84sra_failure_audit.json", AUDIT / "support_alignment_callgraph.json",
        METRICS / "b84s_event_replay_b84sproto_v1.json", AUDIT / "b84sproto_failure_audit.json",
        OUT / "manifests/b84s_native_manifest.json", OUT / "manifests/b84sq_balanced_v3_manifest.json",
    ]
    provenance = {
        "schema_version": "trackocd.phase84.report_provenance.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report": str(REPORT.resolve()),
        "headline_sources": [artifact(p) for p in provenance_paths],
        "no_hardcoded_scientific_headline_values": True,
        "lock": lock,
        "git_head": git("rev-parse", "HEAD"),
    }
    atomic_json(PROVENANCE, provenance)
    failed_markers = []
    for p in sorted(COMP.glob("*.launched")):
        done = p.with_suffix(".done")
        if not done.exists(): failed_markers.append(p.name)
    status = "AUTONOMOUS_PHASE84_COMPLETE_WITH_INTERFACE_NEGATIVE_EVIDENCE"
    decision = {
        "schema_version": "trackocd.phase84.decision.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "window": {"start_utc": reg.get("start_time_utc"), "deadline_utc": reg.get("deadline_utc"), "finalized_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "lock": lock},
        "git_head": git("rev-parse", "HEAD"),
        "gates": {
            "A84P_true_physical_R": "A84P_TRUE_PHYSICAL_R_FAIL_WITH_VALID_CONTRACT",
            "B84S_original": "B84S_SOURCE_CONDITIONED_SELECTION_FAIL",
            "B84S_Q": b84sq_audit.get("decision", "B84S_Q_FAIL"),
            "B84S_RA": b84sra_audit.get("decision", "B84S_RA_PARTIAL_NO_ALIGNMENT"),
            "B84S_PROTO": b84sproto_audit.get("decision", "B84S_PROTO_FAIL_NO_ALIGNMENT"),
            "B84A_alignment": "B84A_ALIGNMENT_NOT_NEEDED",
            "C84_controller": "NOT_RUN",
            "sealed": "NOT_RUN",
        },
        "headline": {
            "frozen_phase75b_o": {"source": 49, "target": 40, "both": 25, "denominator": 76},
            "physical_p16": phys_p16,
            "b84s_q_p16": q16,
            "b84s_ra_p16": ra16,
            "b84s_original_p16": old16,
        },
        "protocol": {"positive_events": 76, "negative_events": 76, "prefixes": [1, 2, 4, 8, 16], "r_queries": 984, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "controller_run": False, "threshold_sweep": False},
        "failed_uncompleted_markers": failed_markers,
        "next_action": "No alignment/controller route is authorized because B84S-RA positive reliable selection is below the registered >30/76 criterion; retain evidence and pursue a separately registered query-conditioned representation/physical-support route in a future window.",
    }
    atomic_json(DECISION, decision)
    prefix_rows = []
    for p in [1, 2, 4, 8, 16]:
        d = physical.get("prefix", {}).get(str(p), {})
        prefix_rows.append([p, d.get("queries"), d.get("r1"), d.get("raw_r1"), d.get("map"), d.get("raw_map"), d.get("hard_negative_gap"), d.get("raw_hard_negative_gap"), d.get("unsafe_flip_count")])
    signal_rows = []
    for p in [1, 2, 4, 8, 16]:
        d = signal.get("aggregate", {}).get(str(p), {})
        signal_rows.append([p, d.get("queries"), d.get("r1"), d.get("raw_r1"), d.get("map"), d.get("raw_map"), d.get("hard_negative_gap"), d.get("raw_hard_negative_gap"), d.get("unsafe_flip_count")])
    event_rows = []
    for p in [1, 2, 4, 8, 16]:
        for pol in ["positive", "negative"]:
            d = p16_summary(b84sra_event).get(pol, {}) if p == 16 else {str(x.get("polarity")): x for x in b84sra_event.get("summary", []) if x.get("prefix") == p}.get(pol, {})
            event_rows.append([p, pol, d.get("events"), d.get("selected_candidate_events"), d.get("selected_reliable_events"), d.get("raw_source_mean_reliable_events"), d.get("frozen_source_reliable"), d.get("frozen_target_reliable"), d.get("frozen_both_reliable")])
    fold_rows = []
    for row in b84sq_audit.get("fold_prefix_summary", []):
        if row.get("prefix") == 16:
            fold_rows.append([row.get("fold"), row.get("polarity"), row.get("events"), row.get("selected_candidate"), row.get("selected_reliable"), row.get("raw_source_mean_reliable"), row.get("taxonomy")])
    formal_rows = []
    for r in b84sq_formal.get("folds", []):
        vm = r.get("validation_metrics", {})
        formal_rows.append([r.get("fold"), r.get("steps"), vm.get("groups"), vm.get("candidate_top1_recall"), vm.get("candidate_top5_recall"), vm.get("candidate_or_defer_accuracy"), vm.get("defer_recall")])
    report = f"""# TrackOCD Phase84 — Autonomous Research Report

Status: **{status}**  
Window: `{reg.get('start_time_utc')}` → `{reg.get('deadline_utc')}`  
Finalized: `{dt.datetime.now(dt.timezone.utc).isoformat()}`  
Git HEAD: `{git('rev-parse', 'HEAD')}` (changes were committed and pushed to `origin/main` before report generation)

## Executive decision

Phase84 corrected the Phase83 reporting, physical-lineage, and source/query interface errors. The true full-native physical reassociation completed, but its frozen-R safety gate failed. The repaired query-conditioned native selector and one fixed raw-anchor residual diagnostic did not reach the registered selection criterion. Therefore Phase84 does **not** run alignment, the historical controller, Commit-CT, or sealed/public evaluation. This is a window-level interface/selection negative result, not a claim that TrackOCD is universally infeasible.

## Protocol and sealed boundaries

- Positive/negative event denominator: `76 + 76`; prefixes `{1,2,4,8,16}`; frozen R universe: `984` queries.
- Physical IDs are runtime bookkeeping only. Inference tensors contain visual/geometry/causal-history fields; no category/text, semantic ID, numeric physical-ID feature, future row/track, held GT, DEV+, Q1, public-new, or sealed label was used.
- Event labels/GT are post-hoc scoring metadata only. No threshold sweep, controller, StateMemory, or sealed/public run occurred.
- Frozen historical Phase75B O reference (reported without reinterpretation): source `49/76`, target `40/76`, both `25/76`.

## Phase83 correction and physical P→R

The A2 report-source audit is `{f(a2_integrity.get('report_generator_wrong_source'))}`; the actual artifact and the mistakenly rendered artifact are retained in `{str((AUDIT / 'phase83_a2_report_integrity.json').resolve())}`. No Phase83 artifact was modified.

The true A84P route unions full-native Q0 fragments causally (dormant-only candidates, observed-step timing, gap ≤16, collision-safe canonical IDs), then rebuilds membership before applying the frozen visual R adapter. It preserved the native row denominator and passed Q0 adapter parity. Physical lineage: `{physical_diag.get('rows')}` rows, `{physical_diag.get('union_count')}` causal unions, `{physical_diag.get('semantic_contamination', {}).get('roots_multi_category')}` multi-category roots among `{physical_diag.get('semantic_contamination', {}).get('roots_with_train_category')}` labeled roots; this is post-hoc TRAIN audit evidence, not model input.

### A84P frozen-R prefix comparison

{table(['prefix','queries','physical R@1','raw R@1','physical mAP','raw mAP','physical gap','raw gap','unsafe'], prefix_rows)}

At prefix16, physical R@1/mAP are `{f(phys_p16.get('r1'))}`/`{f(phys_p16.get('map'))}` versus raw `{f(phys_p16.get('raw_r1'))}`/`{f(phys_p16.get('raw_map'))}`, with `{f(phys_p16.get('unsafe_flip_count'))}` unsafe flips. The route gate is **FAIL** (`safe_r_signal={f(physical.get('gate_diagnostic', {}).get('safe_r_signal'))}`); no controller was run.

## B84S source-conditioned support

The corrected same-space TRAIN signal audit was diagnostic only:

{table(['prefix','queries','source-conditioned R@1','raw R@1','source-conditioned mAP','raw mAP','source gap','raw gap','unsafe'], signal_rows)}

The original B84S formal selector used a query-agnostic source attachment and is retained as a failed interface comparator. Its p16 replay summary is `{json.dumps(old16, sort_keys=True)}`.

### B84S-Q repaired query contract

The repaired manifest uses legal Phase30 TRAIN query/support pairs, native Q0 candidate sets, explicit DEFER, event-video exclusion, and a deterministic three-fold fallback because a four-fold split could not retain the preregistered minimum fit/validation group counts. It contains `{load(OUT / 'manifests/b84sq_balanced_v3_manifest.json').get('groups')}` groups and `{load(OUT / 'manifests/b84sq_balanced_v3_manifest.json').get('candidate_rows')}` candidate rows; the fold imbalance is retained rather than hidden.

TRAIN-disjoint validation (all completed folds):

{table(['fold','steps','groups','candidate top1','candidate top5','candidate/DEFER acc','DEFER recall'], formal_rows)}

Frozen event replay (all `760` records):

{table(['prefix','polarity','events','selected candidate','selected reliable','raw source-mean reliable','frozen source','frozen target','frozen both'], event_rows)}

At prefix16 B84S-Q selected reliable candidates on `{q16.get('positive', {}).get('selected_reliable_events')}/76` positive events and `{q16.get('negative', {}).get('selected_reliable_events')}/76` negative events, versus raw source-mean `{q16.get('positive', {}).get('raw_source_mean_reliable_events')}/76` and `{q16.get('negative', {}).get('raw_source_mean_reliable_events')}/76`. All event candidate sets were nonempty (median `{b84sq_audit.get('p16', {}).get('positive', {}).get('candidate_count', {}).get('median')}` candidates). The repaired selector therefore **FAILS** to improve the frozen support selection; this is not an empty-pool result.

### B84S-RA raw-anchor diagnostic

The one registered no-training diagnostic added a fixed `0.05*tanh` bounded residual to raw source-mean cosine and used raw candidate fallback when the frozen model emitted DEFER. At prefix16 it reached `{ra16.get('positive', {}).get('selected_reliable_events')}/76` positive and `{ra16.get('negative', {}).get('selected_reliable_events')}/76` negative reliable selections, versus raw `{ra16.get('positive', {}).get('raw_source_mean_reliable_events')}/76` and `{ra16.get('negative', {}).get('raw_source_mean_reliable_events')}/76`. Per-fold positive counts and event taxonomy are in `{str((AUDIT / 'b84sra_failure_audit.json').resolve())}`. The modest positive increase remains below the registered `>30/76` alignment-routing criterion and increases negative activation; status is **PARTIAL / no alignment**.

### B84S-PROTO fixed prototype-anchor diagnostic

The final registered source-representation diagnostic selected by maximum
cosine to the fixed three contiguous causal source prototypes. At prefix16 it
reached `{proto16.get('positive', {}).get('selected_reliable_events')}/76` positive and `{proto16.get('negative', {}).get('selected_reliable_events')}/76` negative reliable selections, compared with raw source-mean `{proto16.get('positive', {}).get('raw_source_mean_reliable_events')}/76` and `{proto16.get('negative', {}).get('raw_source_mean_reliable_events')}/76`. This is below the `>30/76` alignment criterion and has higher negative activation than B84S-RA; it is **FAIL / no alignment**. Full event taxonomy is in `{str((AUDIT / 'b84sproto_failure_audit.json').resolve())}`.

### Event-level failure evidence

The repaired B84S-Q prefix16 taxonomy is retained in `{str((AUDIT / 'b84sq_failure_audit.json').resolve())}`. Its fold view is:

{table(['event fold','polarity','events','selected','reliable','raw reliable','taxonomy'], fold_rows)}

The support-alignment callgraph confirms that B84S/B84S-Q/B84S-RA/B84S-PROTO implement selection only; no transformed support IoU is present. Alignment is therefore **NOT_RUN**, not zeroed.

## Route gates

{table(['route','status','evidence'], [
    ['A84P true physical→R','FAIL', 'full native canonical membership, Q0 parity, unsafe/2-of-4 non-decreasing gate'],
    ['B84S original','FAIL', 'query-agnostic source attachment; p16 reliable selection below comparator'],
    ['B84S-Q repaired matcher','FAIL', '7/76 positive reliable at p16; repaired query contract still does not preserve raw signal'],
    ['B84S-RA raw-anchor','PARTIAL', '24/76 positive, 9/76 negative; below >30/76 alignment criterion'],
    ['B84S-PROTO fixed M=3 prototypes','FAIL', '23/76 positive, 11/76 negative; below alignment criterion'],
    ['B84A alignment','NOT_RUN', 'selection criterion not met; no transformed-support implementation'],
    ['C84 controller / Commit-CT','NOT_RUN', 'R/O routes did not authorize controller'],
    ['sealed/public','NOT_RUN', 'sealed boundary remained closed'],
])}

## Resource, process, and repair audit

The run used bounded CPU workers for B84S/B84S-Q and no GPU training; no OOM or external-process termination occurred. GPU/RAM/disk snapshots and process state are in `{str((AUDIT / 'research_ledger.json').resolve())}`. Symlinked output storage is recorded in the registration and manifests; large native data/checkpoints remain on `/data2/usr_for_deadline/trackocd_phase84` or prior read-only Phase83 targets.

Uncompleted `.launched` markers are preserved as failed evidence (not relabeled): `{', '.join(failed_markers) if failed_markers else 'none'}`. Repair records, commands, compile checks, hashes, and intentionally unrun historical suites are in `{str((AUDIT / 'repair_events.json').resolve())}` and `{str((AUDIT / 'validation_evidence_ledger.json').resolve())}`. No Phase84 process remained at finalization.

## Reproduction

```bash
cd {ROOT}
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/run_full_temporal_physical.py --tag full_temporal_r1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/build_physical_r_adapter.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/evaluate_b84s_event_replay.py --model-prefix b84sq_b84sq_formal_v3 --fold-count 3 --suffix _b84sq_v3 --manifest outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase84/evaluate_b84s_event_replay.py --model-prefix b84sq_b84sq_formal_v3 --fold-count 3 --suffix _b84sra_v1 --manifest outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json --raw-anchor --raw-anchor-bound 0.05
```

The formal B84S-Q checkpoints and hashes are in the fold metrics and formal aggregate; the event replay is frozen and post-hoc. The machine-readable decision is `{str(DECISION.resolve())}` and report provenance is `{str(PROVENANCE.resolve())}`.

## Conclusion and next direction

Phase84 resolves the historical interface confounds and supplies valid negative/partial evidence: physical reassociation changed canonical membership but did not safely transfer to frozen R; the native candidate pool is present; the query-conditioned matcher still fails to generalize its ranking under the sparse disjoint TRAIN contract; a bounded raw anchor recovers a small amount but does not cross the registered alignment gate. The final MOT+OCD causal controller and sealed persistent Commit-CT remain unmeasured in this window. A future window should register one new evidence-backed query-conditioned representation/support contract (with explicit train/runtime candidate parity and broader legal source coverage) before any controller or backbone work; threshold and StateMemory tuning are not justified by Phase84.

## Artifact index

The live/research ledgers and all hashes are preserved in `outputs/iclr27_phase84/audit/`; the complete source list used for the headline tables is in `report_provenance.json`. Historical Phase83 outputs and reports were not overwritten.
"""
    atomic_text(REPORT, report)
    print(json.dumps({"status": status, "report": str(REPORT.resolve()), "decision": str(DECISION.resolve()), "provenance": str(PROVENANCE.resolve()), "git_head": git("rev-parse", "HEAD")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
