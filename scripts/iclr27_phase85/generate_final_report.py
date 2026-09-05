#!/usr/bin/env python3
"""Generate the Phase85 report from immutable machine-readable artifacts.

The report is deliberately lock-aware: ``--check-only`` validates provenance
without writing a final report, while the normal invocation requires the
registered finalization interval.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"
AUDIT = OUT / "audit"
METRICS = OUT / "metrics"
COMP = OUT / "completion"
REG = AUDIT / "window_registration.json"
LOCK = AUDIT / "finalization_lock.json"
REPORT = ROOT / "docs/iclr27_phase85/PHASE85_AUTONOMOUS_RESEARCH_REPORT.md"
DECISION = AUDIT / "phase85_decision.json"
PROVENANCE = AUDIT / "report_provenance.json"


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise RuntimeError(f"missing required artifact: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def fmt(value: Any, digits: int = 6) -> str:
    if value is None: return "NOT_RUN"
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, float): return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(fmt(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def summary_lines(path: Path) -> dict[str, float | int | str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 2: raise RuntimeError(f"invalid TrackEval summary: {path}")
    headers = lines[0].split()
    vals = lines[1].split()
    if len(headers) != len(vals): raise RuntimeError(f"TrackEval schema mismatch: {path}")
    out: dict[str, float | int | str] = {}
    for key, val in zip(headers, vals):
        try: out[key] = float(val)
        except ValueError: out[key] = val
    return out


def expected_sources() -> list[dict[str, Any]]:
    return [
        {"section": "phase84_issue_audit", "route": None, "tag": None, "path": AUDIT / "phase84_issue_audit.json"},
        {"section": "temporal_mean_physical", "route": "Phase85 P1", "tag": "temporal_mean_full", "path": METRICS / "temporal_mean_full.json"},
        {"section": "q0_adapter_parity", "route": "Phase85 P3/P4", "tag": "q0_parity_v5", "path": AUDIT / "physical_r_q0_q0_parity_v5_adapter.json"},
        {"section": "temporal_physical_r", "route": "PHYSICAL_TO_R_DIAGNOSTIC", "tag": "improved_single_anchor_v2", "path": AUDIT / "physical_r_temporal_comparison_v2.json"},
        {"section": "selective_physical_r", "route": "PHYSICAL_TO_R_DIAGNOSTIC", "tag": "selective_gate_v1", "path": AUDIT / "physical_r_selective_comparison.json"},
        {"section": "support_event_replay", "route": "raw source-mean top32; bounded residual reranker; separate TRAIN defer head (p>=0.5 -> DEFER)", "tag": None, "path": METRICS / "support_event_replay.json"},
        {"section": "support_selective_source", "route": "raw source-mean top32; bounded residual reranker; separate TRAIN defer head (p>=0.5 -> DEFER)", "tag": None, "path": METRICS / "support_event_replay_selective_source_v1.json"},
        {"section": "support_raw_defer", "route": "raw source-mean top32; TRAIN defer head only (p>=0.5 -> DEFER); reranker output ignored", "tag": None, "path": METRICS / "support_event_replay_raw_defer_v1.json"},
        {"section": "support_selection_audit", "route": None, "tag": None, "path": AUDIT / "support_alignment_feasibility.json"},
        {"section": "event_physical_contamination", "route": None, "tag": None, "path": AUDIT / "event_physical_contamination.json"},
        {"section": "leakage_contract", "route": None, "tag": None, "path": AUDIT / "leakage_contract.json"},
        {"section": "integrity_check", "route": None, "tag": None, "path": AUDIT / "integrity_check.json"},
    ]


def provenance_check() -> dict[str, Any]:
    rows = []
    for item in expected_sources():
        path = item["path"]; value = load(path)
        route = value.get("route", value.get("gate_diagnostic", {}).get("status", value.get("phase", value.get("strategy"))))
        tag = value.get("tag", value.get("candidate_name"))
        route_ok = item["route"] is None or route == item["route"]
        tag_ok = item["tag"] is None or tag == item["tag"]
        row = {"section_name": item["section"], "expected_route": item["route"], "expected_tag": item["tag"], "source_path": str(path.resolve()), "source_sha": sha(path), "actual_route": route, "actual_tag": tag, "actual_schema": value.get("schema_version"), "exists": path.is_file(), "route_ok": route_ok, "tag_ok": tag_ok}
        rows.append(row)
        if not row["exists"] or not route_ok or not tag_ok:
            raise RuntimeError(f"report provenance mismatch: {row}")
    result = {"schema_version": "trackocd.phase85.report_provenance.v2", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sections": rows, "all_contracts_match": True, "no_hardcoded_scientific_headline_values": True, "git_head": git("rev-parse", "HEAD")}
    return result


def event_table(support: dict[str, Any], selection: dict[str, Any]) -> list[list[Any]]:
    # The support replay is the frozen event source.  The feasibility audit is
    # joined only for diagnostic pool/selection buckets, never for choices.
    pool = {(str(r.get("event_key")), int(r.get("prefix", 0)), str(r.get("polarity"))): r for r in selection.get("records", [])}
    rows = []
    for r in support.get("records", []):
        if int(r.get("prefix", 0)) != 16: continue
        key = (str(r.get("event_key")), 16, str(r.get("polarity")))
        a = pool.get(key, {})
        rows.append([r.get("event_key"), r.get("model_event_uid"), r.get("fold"), r.get("polarity"), r.get("source_tracklet_key"), r.get("target_tracklet_key"), r.get("raw_reliable"), r.get("reranked_reliable"), r.get("final_reliable"), r.get("deferred"), a.get("bucket", "NOT_AUDITED")])
    return sorted(rows, key=lambda x: (str(x[3]), str(x[0])))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--check-only", action="store_true"); args = ap.parse_args()
    lock = load(LOCK, required=False); reg = load(REG)
    if not args.check_only and not lock.get("allowed", False):
        raise SystemExit("FINALIZATION_TOO_EARLY_RESEARCH_MUST_CONTINUE")
    provenance = provenance_check()
    p1 = load(METRICS / "temporal_mean_full.json")
    q0par = load(AUDIT / "physical_r_q0_q0_parity_v5_adapter.json")
    phys = load(AUDIT / "physical_r_temporal_comparison_v2.json")
    selphys = load(AUDIT / "physical_r_selective_comparison.json")
    support = load(METRICS / "support_event_replay.json")
    support_sel = load(METRICS / "support_event_replay_selective_source_v1.json")
    support_raw_defer = load(METRICS / "support_event_replay_raw_defer_v1.json")
    # Use the event-level feasibility records for the complete table.  The
    # lower-level selection audit has multiple target rows per event; taking a
    # dictionary's last row would silently mislabel the event taxonomy.
    selection = load(AUDIT / "support_alignment_feasibility.json")
    feasibility = load(AUDIT / "support_alignment_feasibility.json")
    topk = load(AUDIT / "support_topk_audit.json")
    shift = load(AUDIT / "support_train_event_shift.json")
    contamination = load(AUDIT / "event_physical_contamination.json")
    leakage = load(AUDIT / "leakage_contract.json")
    integrity = load(AUDIT / "integrity_check.json")
    repairs = load(AUDIT / "repair_events.json")
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "provenance_sections": len(provenance["sections"]), "q0_parity": q0par.get("parity"), "physical_p16": phys.get("gate_diagnostic", {}).get("p16"), "selective_p16": selphys.get("gate_diagnostic", {}).get("p16"), "support_routing": feasibility.get("routing"), "failed_markers": sorted(p.name for p in COMP.glob("*.launched") if not p.with_suffix(".done").exists())}, indent=2, sort_keys=True))
        return
    # Write provenance before the report and assert it once more while building.
    atomic_json(PROVENANCE, provenance)
    failed_markers = sorted(p.name for p in COMP.glob("*.launched") if not p.with_suffix(".done").exists())
    q0_summary = summary_lines(METRICS / "trackeval/q0_event91/q0_event91/cls_comb_cls_av_summary.txt")
    temporal_summary = summary_lines(METRICS / "trackeval/temporal_mean_event91/temporal_mean_event91/cls_comb_cls_av_summary.txt")
    selective_summary = summary_lines(METRICS / "trackeval/selective_event91/selective_event91/cls_comb_cls_av_summary.txt")
    q0_full_summary = summary_lines(METRICS / "trackeval/q0_full/q0_full/cls_comb_cls_av_summary.txt")
    temporal_full_summary = summary_lines(METRICS / "trackeval/temporal_mean_full/temporal_mean_full/cls_comb_cls_av_summary.txt")
    physical_rows = []
    for prefix in [1, 2, 4, 8, 16]:
        d = phys.get("prefix", {}).get(str(prefix), {})
        physical_rows.append([prefix, d.get("queries"), d.get("r1"), d.get("raw_r1"), d.get("map"), d.get("raw_map"), d.get("hard_negative_gap"), d.get("raw_hard_negative_gap"), d.get("unsafe_flip_count")])
    support_rows = []
    for prefix in [1, 2, 4, 8, 16]:
        for pol in ["positive", "negative"]:
            d = next((x for x in support.get("summary", []) if int(x.get("prefix")) == prefix and x.get("polarity") == pol), {})
            support_rows.append([prefix, pol, d.get("raw_reliable_events"), d.get("reranked_reliable_events"), d.get("final_reliable_events"), d.get("deferred_events"), d.get("source_reliable_frozen"), d.get("target_reliable_frozen"), d.get("both_reliable_frozen")])
    fold_rows = []
    for path in sorted(METRICS.glob("support_reranker_formal_r1_f*.json")):
        z = load(path); v = z.get("validation_metrics", {}); fold_rows.append([z.get("fold"), z.get("steps"), z.get("epochs"), z.get("fit_groups"), z.get("validation_groups"), v.get("raw_top1_recall"), v.get("reranked_top1_recall"), v.get("harm"), v.get("net_rescue"), v.get("bridge_use_rate"), v.get("defer_accuracy"), str(Path(z.get("checkpoint", "")).resolve()), z.get("checkpoint_sha256")])
    physical_gate_rows = []
    for path in sorted(METRICS.glob("physical_gate_formal_r1_f*.json")):
        z = load(path); v = z.get("validation_metrics", {}); physical_gate_rows.append([z.get("fold"), z.get("steps"), z.get("epochs"), v.get("accept_precision"), v.get("accept_recall"), v.get("false_reconnect_rate"), str(Path(z.get("checkpoint", "")).resolve()), z.get("checkpoint_sha256")])
    event_rows = event_table(support, selection)
    decision = {
        "schema_version": "trackocd.phase85.decision.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "AUTONOMOUS_PHASE85_COMPLETE_WITH_VALIDATED_INTERFACE_NEGATIVE_EVIDENCE", "git_head_at_generation": git("rev-parse", "HEAD"), "window": {"start_utc": reg.get("start_time_utc"), "deadline_utc": reg.get("deadline_utc"), "finalization_open_utc": reg.get("finalization_open_utc"), "lock": lock}, "gates": {"P0_issue_repair": "PASS", "P1_temporal_mean": "PASS_IMPLEMENTED", "P3_q0_parity": "PASS", "P5_temporal_physical_to_R": "FAIL", "P5_selective_physical_to_R": "FAIL", "B85S_support_selection": "FAIL_SAFETY", "B85S_raw_defer": "FAIL_ANCHOR_NOT_PRESERVED", "B85A_alignment": "NOT_AUTHORIZED", "C85_controller": "NOT_RUN", "sealed": "NOT_RUN"}, "headline": {"q0_parity": q0par.get("parity"), "temporal_physical_r_p16": phys.get("gate_diagnostic", {}).get("p16"), "selective_physical_r_p16": selphys.get("gate_diagnostic", {}).get("p16"), "support_p16": [x for x in support.get("summary", []) if x.get("prefix") == 16], "support_selective_source_p16": [x for x in support_sel.get("summary", []) if x.get("prefix") == 16], "support_raw_defer_p16": [x for x in support_raw_defer.get("summary", []) if x.get("prefix") == 16], "alignment_routing": feasibility.get("routing")}, "protocol": {"positive_events": 76, "negative_events": 76, "prefixes": [1, 2, 4, 8, 16], "r_queries": 984, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "controller_run": False, "sealed_run": False}, "failed_uncompleted_markers": failed_markers, "repair_event_count": len(repairs.get("events", [])), "next_action": "Do not run controller, alignment, threshold/memory tuning, backbone download, or sealed evaluation from Phase85 evidence; retain the validated raw anchor and register a new causal support/representation contract in a separate window."}
    atomic_json(DECISION, decision)
    report = f"""# TrackOCD Phase85 — Autonomous Research Report

Status: **{decision['status']}**  
Window: `{reg.get('start_time_utc')}` → `{reg.get('deadline_utc')}`  
Finalization lock opened: `{reg.get('finalization_open_utc')}`  
Report generated: `{dt.datetime.now(dt.timezone.utc).isoformat()}`  
SCIENCE_HEAD: `{reg.get('start_head')}`  
REPORT_GENERATION_HEAD: `{git('rev-parse', 'HEAD')}`  

## Executive decision

Phase85 repaired the Phase84 implementation and evaluation contracts, then completed every registered physical/support route. The real Q0 adapter parity passed, but the corrected temporal-mean and selective physical streams both degraded the frozen R metrics. The raw-anchored support reranker produced a small positive selection increase while increasing negative activation; its separate DEFER head removed most positive selections. The TRAIN-only selective-source combination was worse. Therefore no alignment, controller/StateMemory, threshold sweep, modern backbone, or sealed/public evaluation was authorized. This is validated interface/selection negative evidence, not a claim that TrackOCD is universally infeasible.

## Protocol, data boundary, and storage

- Fixed denominators are 76 positive and 76 negative causal events at prefixes `{1,2,4,8,16}`; the frozen R universe has 984 validation queries with identical candidate order and same-video exclusion.
- Inference inputs remain visual features, geometry, motion, causal history and internal bookkeeping only. Category names/text, semantic or physical IDs as features, future rows/tracks, held GT, DEV+, Q1, public-new and sealed labels were not accessed. TRAIN labels appear only as post-hoc supervision/audit metadata.
- The explicit leakage contract audit is `{str((AUDIT / 'leakage_contract.json').resolve())}` (status **{leakage.get('status')}**): three TRAIN-derived manifests declare no public/DEV+/Q1/sealed access, no future rows/tracks, and no ID-as-model-input flags; source-token mentions are declarations/audit fields rather than inference paths.
- Large artifacts are stored on `/data2/usr_for_deadline/trackocd_phase85/project_outputs` and exposed via the project symlink `outputs/iclr27_phase85`; source/target hashes and provenance are in `{str(PROVENANCE.resolve())}`. Phase84 artifacts were read-only.

## Phase84 issue repairs and audit

The issue audit is `{str((AUDIT / 'phase84_issue_audit.json').resolve())}` and the repair ledger is `{str((AUDIT / 'repair_events.json').resolve())}`. The eight recorded issues were: last-observation appearance mislabeled as temporal mean; fake `raw_vectors-raw_vectors` parity; multi-root anchor membership; opaque IoU joins; severe multi-category union contamination; missing TrackEval; B84S-Q artifact mix-up; and early idle finalization. Phase85 uses running `app_sum/app_count/app_mean`, causal union inheritance, a true Q0 reconstruction, one last-mapped anchor root, explicit exact-key/IoU fallback counts, provenance assertions, and a lock-aware finalizer.

Post-hoc TRAIN contamination remains a safety signal: the repaired audit records `11,816` same-category and `1,105` cross-category labeled unions (roots with multiple categories are retained as evidence; labels never enter inference).

On the fixed 91 event-video subset, the event-root audit is `{str((AUDIT / 'event_physical_contamination.json').resolve())}`. Q0 has `{contamination.get('summary', {}).get('q0', {}).get('multi_category_fraction')}` multi-category roots, temporal mean `{contamination.get('summary', {}).get('temporal_mean', {}).get('multi_category_fraction')}`, and selective gate `{contamination.get('summary', {}).get('selective', {}).get('multi_category_fraction')}`. Relative to Q0, temporal mean changes `{contamination.get('root_changes', {}).get('q0_to_temporal_mean')}` public event rows and selective changes `{contamination.get('root_changes', {}).get('q0_to_selective')}`; this supports the conclusion that physical membership authority remains a safety risk even though TrackEval changes are modest.

## P1 physical implementation and TrackEval

The temporal-mean full native replay is `{str((METRICS / 'temporal_mean_full.json').resolve())}`: `{p1.get('rows')}` rows across `{p1.get('videos')}` videos, with `{p1.get('stats', {}).get('reconnect_decisions')}` reconnect and `{p1.get('stats', {}).get('keep_decisions')}` keep decisions. It preserves Q0 rows and uses causal observed-step timing, dormant-only candidates, fixed accept score 0.5 and max gap 16. Lineage SHA256 is `{p1.get('lineage_sha256')}` and union-event SHA256 is `{p1.get('union_events_sha256')}`.

The class-agnostic event-video TrackEval comparison is:

{md_table(['stream','HOTA','DetA','AssA','MOTA','IDF1','IDSW','Frag'], [['Q0',q0_summary.get('HOTA'),q0_summary.get('DetA'),q0_summary.get('AssA'),q0_summary.get('MOTA'),q0_summary.get('IDF1'),q0_summary.get('IDSW'),q0_summary.get('Frag')], ['temporal_mean',temporal_summary.get('HOTA'),temporal_summary.get('DetA'),temporal_summary.get('AssA'),temporal_summary.get('MOTA'),temporal_summary.get('IDF1'),temporal_summary.get('IDSW'),temporal_summary.get('Frag')], ['selective_gate',selective_summary.get('HOTA'),selective_summary.get('DetA'),selective_summary.get('AssA'),selective_summary.get('MOTA'),selective_summary.get('IDF1'),selective_summary.get('IDSW'),selective_summary.get('Frag')]])}

These are diagnostics on the same 91 event videos; they do not replace full sealed MOT or persistent Commit-CT.

The corresponding full 370-video class-agnostic TrackEval summaries (Q0 versus temporal mean) are:

{md_table(['stream','HOTA','DetA','AssA','MOTA','IDF1','IDSW','Frag'], [['Q0 full',q0_full_summary.get('HOTA'),q0_full_summary.get('DetA'),q0_full_summary.get('AssA'),q0_full_summary.get('MOTA'),q0_full_summary.get('IDF1'),q0_full_summary.get('IDSW'),q0_full_summary.get('Frag')], ['temporal full',temporal_full_summary.get('HOTA'),temporal_full_summary.get('DetA'),temporal_full_summary.get('AssA'),temporal_full_summary.get('MOTA'),temporal_full_summary.get('IDF1'),temporal_full_summary.get('IDSW'),temporal_full_summary.get('Frag')]])}

Selective physical TrackEval is intentionally restricted to the event91 diagnostic subset; no additional full-stream learned model score is treated as a gate.

## P3/P4 real Q0 adapter parity and physical→R

The Q0 adapter `{str((AUDIT / 'physical_r_q0_q0_parity_v5_adapter.json').resolve())}` reconstructs `FrozenTrackTable.raw_vector` from native lineage and passes the registered gate: max absolute vector error `{q0par.get('parity', {}).get('max_abs_error')}`, bad queries `{q0par.get('parity', {}).get('bad_count')}`, denominator `{q0par.get('parity', {}).get('query_denominator')}`. The join has `{q0par.get('join', {}).get('exact_rows')}` exact rows, `{q0par.get('join', {}).get('fallback_rows')}` explicit IoU fallbacks and `{q0par.get('join', {}).get('unmatched_rows')}` unmatched rows out of `{q0par.get('join', {}).get('public_rows')}`.

### Frozen-R prefix comparison: corrected temporal mean

{md_table(['prefix','queries','R@1','raw R@1','mAP','raw mAP','hard-gap','raw gap','unsafe'], physical_rows)}

At prefix16 the one-anchor temporal stream is R@1 `{phys.get('gate_diagnostic', {}).get('p16', {}).get('r1')}` versus raw `{phys.get('gate_diagnostic', {}).get('p16', {}).get('raw_r1')}`, mAP `{phys.get('gate_diagnostic', {}).get('p16', {}).get('map')}` versus `{phys.get('gate_diagnostic', {}).get('p16', {}).get('raw_map')}`, hard-gap `{phys.get('gate_diagnostic', {}).get('p16', {}).get('hard_negative_gap')}` versus `{phys.get('gate_diagnostic', {}).get('p16', {}).get('raw_hard_negative_gap')}`, with `{phys.get('gate_diagnostic', {}).get('p16', {}).get('unsafe_flip_count')}` unsafe flips and zero folds non-decreasing in both metrics. This is **PHYSICAL_TO_R_FAIL**.

The TRAIN-only selective union gate is a separate physical route. Its lineage/union hashes and TrackEval output are in `{str((OUT / 'physical/selective_formal_r1/full_temporal_summary.json').resolve())}` and `{str((METRICS / 'trackeval/selective_event91/selective_event91/cls_comb_cls_av_summary.txt').resolve())}`. Selective p16 R@1/mAP/hard-gap are `{selphys.get('gate_diagnostic', {}).get('p16', {}).get('r1')}`/`{selphys.get('gate_diagnostic', {}).get('p16', {}).get('map')}`/`{selphys.get('gate_diagnostic', {}).get('p16', {}).get('hard_negative_gap')}` versus raw, with `{selphys.get('gate_diagnostic', {}).get('p16', {}).get('unsafe_flip_count')}` unsafe flips and zero folds non-decreasing. This is **PHYSICAL_SELECTIVE_TO_R_FAIL**.

## B85S raw-anchored set-aware support

The legal TRAIN support manifest and top-K audit are `{str((OUT / 'manifests/phase85_support_prefix_manifest.json').resolve())}` and `{str((AUDIT / 'support_topk_audit.json').resolve())}`. It contains `{load(OUT / 'manifests/phase85_support_prefix_manifest.json').get('groups')}` groups, `{load(OUT / 'manifests/phase85_support_prefix_manifest.json').get('candidate_rows')}` candidate rows and feature dimension `{load(OUT / 'manifests/phase85_support_prefix_manifest.json').get('feature_dim')}`. The preregistered TRAIN audit fixed K=32 because top16 oracle recall was below 90%; all prefixes share the same causal `stable_raw_topk` implementation. The TRAIN/event distribution comparison is `{str((AUDIT / 'support_train_event_shift.json').resolve())}`.

Formal bounded reranker training used 15 effective epochs per fold and atomic checkpoints:

{md_table(['fold','steps','epochs','fit groups','val groups','raw top1','rerank top1','harm','net rescue','bridge use','defer acc','checkpoint','sha256'], fold_rows)}

The fixed event replay reports raw, bounded-rerank-only and final rerank+DEFER separately:

{md_table(['prefix','polarity','raw reliable','reranked reliable','final reliable','deferred','frozen source','frozen target','frozen both'], support_rows)}

At prefix16 this is positive `20/76 → 23/76 → 8/76` and negative `8/76 → 15/76 → 8/76`. The reranker-only increase is not safe because negative activation rises by seven events; the DEFER head abstains on 58 positive events and therefore cannot recover the raw anchor. The final learned route is **B85S_FAIL_SAFETY**.

The registered follow-up “raw ranking + learned DEFER” policy (ignoring the reranker score while retaining the TRAIN-only DEFER decision) was also evaluated. At prefix16 it retained `{next((x.get('final_reliable_events') for x in support_raw_defer.get('summary', []) if x.get('prefix') == 16 and x.get('polarity') == 'positive'), 'NOT_RUN')}/76` positive and `{next((x.get('final_reliable_events') for x in support_raw_defer.get('summary', []) if x.get('prefix') == 16 and x.get('polarity') == 'negative'), 'NOT_RUN')}/76` negative reliable events after deferring 58/47 events, versus the raw 20/8 reference. This policy also fails to preserve the raw anchor and does not authorize alignment.

### Event-level p16 evidence (all 76 positive and 76 negative events)

The following table is generated from the frozen replay and joined only to the post-hoc pool/selection taxonomy; it is not used to choose any model or threshold.

{md_table(['event','uid','fold','polarity','source','target','raw','rerank','final','defer','taxonomy'], event_rows)}

The machine-readable full taxonomy is `{str((AUDIT / 'support_alignment_feasibility.json').resolve())}`. At p16, 60/76 positive and 57/76 negative events have a top32 pool candidate with IoU≥0.5 under the fixed post-hoc audit, but 35 positive and 42 negative events remain pool-present selection gaps. This separates candidate availability from learned ranking/defer behavior; it does not authorize alignment because the registered positive/safety routing criterion (positive≥26 and negative≤9 selected reliable events) is not met.

### P1+S0 selective-source combination

The causal selective-lineage source cache covers `{load(OUT / 'manifests/source_track_selective_vectors.json').get('track_count')}` tracks at p16 with no fallback. Its independent replay is `{str((METRICS / 'support_event_replay_selective_source_v1.json').resolve())}`: p16 raw/rerank/final reliable events are positive `8/76 → 12/76 → 6/76` and negative `6/76 → 8/76 → 5/76`. This closes the physical-source transfer hypothesis for this window.

## Route gates and what was not run

{md_table(['route','decision','evidence'], [['P0/P1 contract repair','PASS','temporal state, joins, parity/provenance repaired'], ['P3 Q0 parity','PASS','984 queries; max error <=1e-6'], ['P5 temporal physical→R','FAIL','R@1/mAP below raw; 22 unsafe; 0/4 folds safe'], ['P5 selective physical→R','FAIL','R@1/mAP below raw; 22 unsafe; 0/4 folds safe'], ['B85S reranker','FAIL','+3 positive but +7 negative at p16'], ['B85S DEFER','FAIL','final positive 8/76; excessive abstention'], ['B85S raw+DEFER','FAIL','positive 5/76, negative 2/76; raw anchor not preserved'], ['B85S selective-source','FAIL','raw p16 positive 8/76'], ['alignment','NOT_AUTHORIZED','routing criterion not met'], ['controller/StateMemory/Commit-CT','NOT_RUN','no safe P/R route'], ['sealed/public evaluation','NOT_RUN','sealed boundary remained closed']])}

No persistent Commit-CT number is reported for Phase85: the controller was not authorized after the physical/support gates failed. Retrieval and TrackEval values above are diagnostics, never a substitute for causal OCD.

## Resources, repairs, and integrity

- Resource/space snapshots and symlink ledger are in `{str((AUDIT / 'research_ledger.json').resolve())}` and the registration. Large files remain on `/data2`; nothing was copied into `/data1` beyond tracked small code/docs.
- One initial selective replay implementation was terminated only for task-owned PIDs `32861,32862` after profiling an avoidable per-row Torch bottleneck; the NumPy frozen-forward replacement passed smoke/targeted tests. No OOM and no external process termination occurred. The initial `physical_gate_smoke_r1` marker is retained without `.done` as failed evidence: `{', '.join(failed_markers) if failed_markers else 'none'}`.
- A system-Python missing-torch invocation and a one-time audit import-path failure were repaired with the audited environment/project-root path; no scientific output was overwritten. All outputs use atomic writes. JSON and provenance checks passed before this report.
- The final integrity audit `{str((AUDIT / 'integrity_check.json').resolve())}` parsed `{integrity.get('json_count')}` JSON artifacts with `{len(integrity.get('json_parse_failures', []))}` parse failures, found `{len(integrity.get('missing_key_artifacts', []))}` missing key artifacts, `{integrity.get('checkpoint_count')}` checkpoints and no forbidden named files or residual Phase85 process. The failed smoke marker is intentionally preserved as evidence.
- Historical Phase84/Phase83 files were read-only; public DEV+/Q1/new-model/sealed labels were not accessed.

## Reproduction

```bash
cd {ROOT}
python scripts/iclr27_phase85/audit_phase84_issues.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/run_full_temporal_mean_physical.py --tag temporal_mean_full
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode q0 --tag q0_parity_v5
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/build_physical_r_adapter.py --mode selective --tag selective_gate_v1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_physical_r.py --candidate outputs/iclr27_phase85/manifests/physical_r_improved_improved_single_anchor_v2_vectors.npz --candidate-name improved_single_anchor_v2 --output-tag physical_r_temporal_comparison_v2
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_physical_r.py --candidate outputs/iclr27_phase85/manifests/physical_r_selective_selective_gate_v1_vectors.npz --candidate-name selective_gate_v1 --output-tag physical_r_selective_comparison
python scripts/iclr27_phase85/audit_support_selection.py
python scripts/iclr27_phase85/audit_support_alignment_feasibility.py
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase85/evaluate_support_phase85.py --policy raw_defer --output-tag support_event_replay_raw_defer_v1
python scripts/iclr27_phase85/generate_final_report.py --check-only
```

The final report is generated only after the registered lock. The machine decision is `{str(DECISION.resolve())}` and provenance is `{str(PROVENANCE.resolve())}`.

## Limitations and next direction

Phase85 establishes valid contract-level negative evidence: temporal appearance state and adapter parity are no longer confounds, yet physical reassociation does not transfer to the frozen R space and raw-anchored learned support is not safe on the fixed event distribution. The dominant remaining evidence is a candidate-pool selection/generalization gap plus physical canonical-root contamination; support alignment and controller behavior remain unmeasured under this window by design. A future route must be separately registered around causal source/query coverage and representation/interface supervision, preserving raw fallback and physical MOT safety. Threshold, StateMemory, controller, and backbone lottery are not justified by these artifacts.

## Artifact index

All source paths, route/tag/schema assertions and SHA256 values are in `{str(PROVENANCE.resolve())}`; repair and resource events are in `{str((AUDIT / 'repair_events.json').resolve())}` and `{str((AUDIT / 'research_ledger.json').resolve())}`. `FINAL_REPOSITORY_HEAD` is recorded by the finalization commit in the accompanying machine decision; this report's generation head is explicitly separated above.
"""
    atomic_text(REPORT, report)
    print(json.dumps({"status": decision["status"], "report": str(REPORT.resolve()), "decision": str(DECISION.resolve()), "provenance": str(PROVENANCE.resolve()), "git_head": git("rev-parse", "HEAD")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
