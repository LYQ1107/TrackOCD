#!/usr/bin/env python3
"""Generate the two Phase83 reports from immutable JSON artifacts."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
DOC = ROOT / "docs/iclr27_phase83"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f: return json.load(f)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def f(x: Any, n: int = 4) -> str:
    return "n/a" if x is None else f"{float(x):.{n}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def physical_report(x: dict[str, Any]) -> str:
    rows=[]
    d=x["sections"]
    for mode in ("exact_mixed","mapped_subset"):
        for p in (1,2,4,8,16):
            if str(p) not in d[mode]["prefix"]: continue
            z=d[mode]["prefix"][str(p)]; rows.append([mode,p,z["queries"],f(z["raw_r1"]),f(z["r1"]),f(z["r1"]-z["raw_r1"]),f(z["raw_map"]),f(z["map"]),f(z["hard_negative_gap"]),f(z["raw_hard_negative_gap"]),z["unsafe_flip_count"]])
    fold=[]
    for z in d["exact_mixed"]["folds"]:
        if z["prefix"]==16:
            m=z["metrics"]; fold.append([z["fold"],m["queries"],f(m["raw_r1"]),f(m["r1"]),f(m["map"]-m["raw_map"]),f(m["hard_negative_gap"]-m["raw_hard_negative_gap"]),m["unsafe_flip_count"],z["inventory"]["keys_evaluable"]])
    mp=x["mapping"]; ev=x["event_diagnostic"]
    return f"""# Phase83 Physical→R report

**Generated (UTC):** {dt.datetime.now(dt.timezone.utc).isoformat()}  
**Source commit:** `{commit()}`  
**Route:** parameter-free causal temporal-appearance mean from the Phase82R native physical lineage, compared with the frozen Phase75D raw-DINO scorer. This is a TRAIN-disjoint retrieval diagnostic, not an OCD Commit-CT result.

## Contract and inputs

The candidate universe is the exact Phase30 validation universe (all validation tracks except the query itself and same-video tracks). Q0 row keys, candidate order, prefixes `{1,2,4,8,16}`, 984-query fold denominator, and scorer were not changed. Native rows are matched only on the same `(video_id,image_id)` and bbox IoU ≥ 0.5; temporal means use matched current/past rows only. Unmapped tracks are reported explicitly; the exact-universe view falls back to raw only for those tracks and is therefore a diagnostic, not a claim of a new full physical model.

Native lineage: `{mp['native_path']}` (SHA256 `{mp['native_sha256']}`); native features `{mp['native_feature_path']}` (SHA256 `{mp['native_feature_sha256']}`); corrected public CSV SHA256 `{mp['public_csv_sha256']}`. Public/DEV+/Q1/sealed labels were not accessed for model selection; event labels below are post-hoc mapping diagnostics only.

## Mapping coverage

{mp['row_best_iou_ge_0.5']}/{mp['public_rows']} public rows had a native bbox match at IoU≥0.5; {mp['mapped_track_count']}/{mp['track_count']} tracks ({f(mp['mapped_track_fraction']*100,2)}%) were usable. The native stream covers only 91 event videos. In the 76-event post-hoc diagnostic, source mapped={ev['source_mapped']}/76, target mapped={ev['target_mapped']}/76, both={ev['both_mapped']}/76; mean raw cosine={f(ev['mean_raw_cosine'])}, temporal cosine={f(ev['mean_temporal_cosine'])}. This temporal event cosine is lower, not higher.

## Exact-universe and mapped-subset results

""" + md_table(["view","prefix","queries","raw R@1","temporal R@1","ΔR@1","raw mAP","temporal mAP","temporal gap","raw gap","unsafe"],rows) + """

Fold p16 exact-mixed comparison:

""" + md_table(["fold","queries","raw R@1","temporal R@1","ΔmAP","Δhard-gap","unsafe","evaluable"],fold) + f"""

The full exact-mixed p16 aggregate is R@1 `{f(d['exact_mixed']['prefix']['16']['r1'])}` vs raw `{f(d['exact_mixed']['prefix']['16']['raw_r1'])}`, mAP `{f(d['exact_mixed']['prefix']['16']['map'])}` vs `{f(d['exact_mixed']['prefix']['16']['raw_map'])}`, hard-gap `{f(d['exact_mixed']['prefix']['16']['hard_negative_gap'])}` vs `{f(d['exact_mixed']['prefix']['16']['raw_hard_negative_gap'])}`, with `{d['exact_mixed']['prefix']['16']['unsafe_flip_count']}` unsafe flips. Only {d['exact_mixed']['gate_diagnostic']['folds_non_decrease']}/4 folds were non-decreasing in both R@1 and mAP. The mapped-only p16 view has a changed denominator ({d['mapped_subset']['prefix']['16']['queries']} queries) and also decreases R@1/mAP.

The first physical-R invocation was explicitly SIGTERM-ed at task-owned PID 17813 (and its wait shell 17963) after profiling showed repeated per-pair track averaging; it had produced no artifact. Caching raw vectors was the smallest repair, followed by the fold0 smoke/targeted run and the formal run. No external process was touched and no OOM occurred.

## R83 decision

`R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT`. The temporal physical lineage does not improve the frozen raw correspondence signal under the exact full validation contract and has non-zero unsafe flips. No representation training, controller, StateMemory, threshold sweep, DEV+/Q1/public-new or sealed evaluation was run. The registered next work remains the independent O-support route; any later C route is **NOT_RUN**.

Reproduce: `python scripts/iclr27_phase83/run_physical_r.py --run-id phase83-physical-r-temporal-20260904-full`.
"""


def o_report(tax: dict[str, Any], summ: dict[str, Any], inv: dict[str, Any], replay: dict[str, Any], aggregate: dict[str, Any]) -> str:
    pc=summ["pool_upper_bound"]; cats=summ["event_failure_counts"]
    rows=[]
    for z in replay["prefix_summary"]: rows.append([z["prefix"],z["positive_events"],z["negative_events"],z["frozen_both_reliable"],z["learned_both_support_reliable"],z["learned_both_support_selected"],z["negative_both_support_selected"],z["negative_both_support_reliable"]])
    folds=[]
    for fold_key,z in aggregate["folds"].items(): folds.append([fold_key,z["steps"],z["fit_metrics"]["rows"],z["fit_metrics"]["positive_rate"],f(z["validation_metrics"]["roc_auc"]),f(z["validation_metrics"]["f1"]),f(z["loss_first"]),f(z["loss_last"])])
    return f"""# Phase83 O-support report

**Generated (UTC):** {dt.datetime.now(dt.timezone.utc).isoformat()}  
**Source commit:** `{commit()}`  
**Route:** `support_quality_router_v1`, one hidden-128 class-agnostic MLP; no detector/proposal/physical tracker/controller changes.

## Frozen O contract and callgraph

The Phase75B evaluator, 76 positive + 76 negative events, prefixes `{1,2,4,8,16}`, row key and denominator were read-only. `assigned`, `row_iou`, and `track_temporal_iou` are upstream/evaluator metadata used only for TRAIN labels and post-hoc scoring; they are not router inputs. The callgraph and source hashes are in `outputs/iclr27_phase83/audit/support_assignment_callgraph.json`.

Frozen Phase75B p16 event reliability is source {pc['frozen_event_source_reliable']}/76, target {pc['frozen_event_target_reliable']}/76, both {pc['frozen_event_both_reliable']}/76. The causal proposal-pool upper bound (native max IoU≥0.5, diagnostic only) is source {pc['source_max_iou_ge_0.5']}/76, target {pc['target_max_iou_ge_0.5']}/76, both {pc['both_max_iou_ge_0.5']}/76. No proposal is missing at p16.

## 76-event failure taxonomy (p16)

""" + md_table(["class","events"],[[k,v] for k,v in sorted(cats.items())]) + f"""

The 36 E/C cases have a good native pool but no reliable frozen assigned row, while B is a genuinely insufficient pool. Full evidence for every event/side (candidate counts, scores, frame IDs, IoUs and row details) is in `outputs/iclr27_phase83/audit/failure_taxonomy_76.json` and CSV. Prefix aggregation is in `observability_by_prefix.json`.

## TRAIN-only router data and training

Rows came only from non-event public TRAIN videos and roles `known_bank`/`novel_correspondence_train`; all 91 event videos were excluded. Inputs are score, normalized geometry, causal age/stability, history/gap, proposal density/ambiguity and current-vs-causal DINO cosine statistics. GT/assigned/IoU fields form the TRAIN target `assigned==1 AND row_iou>=0.5` only. No category, text, physical/semantic ID, future, event polarity or StateMemory field enters the tensor. Four folds use the frozen video/category-disjoint manifests; scaling is fit on each fold's TRAIN split and threshold is the preregistered p≥0.5.

""" + md_table(["fold","steps","fit rows","fit pos rate","val ROC-AUC","val F1","first loss","last loss"],folds) + f"""

Smoke (100 updates) and fold0 targeted (500 updates) produced finite checkpoints and atomic markers. Formal folds (1000 updates) were CPU bounded because GPUs 0–3 were external jobs and the route does not require GPU; no OOM occurred. Checkpoints and hashes are recorded in `outputs/iclr27_phase83/metrics/support_router_aggregate_formal.json` and `manifests/support_router_inventory_formal.json`.

## Frozen 76+76 event replay

""" + md_table(["prefix","pos","neg","frozen both","learned reliable both","learned selected both","negative selected both","negative reliable both"],rows) + f"""

At p16 the learned router selected support on 46/76 positive events but only 8/76 were both-side reliable (frozen=25/76); it selected support on 52/76 negative events (10/76 had both-side reliable rows). This is over-activation and does not improve O; it cannot be used to claim C or Commit-CT progress.

## O83 decision and reproduction

`O83_FAIL_NO_SUPPORT_GAIN`. The router does not approach the 61/76 pool upper bound, reduces strict both reliability from 25/76 to 8/76, and activates on many negative events. No threshold was tuned on held events, no controller/StateMemory/backbone/public/sealed path was run, and no event was removed. The next action is a contract-level support/proposal assignment investigation, not another router variant.

Reproduce: `python scripts/iclr27_phase83/train_support_router.py --folds 0 --steps 100 --tag smoke`; `python scripts/iclr27_phase83/train_support_router.py --folds 0 --steps 500 --tag targeted`; `python scripts/iclr27_phase83/train_support_router.py --folds 0,1,2,3 --steps 1000 --tag formal`.
"""


def main() -> None:
    physical=load(OUT/"metrics/physical_r_temporal.json"); tax=load(OUT/"audit/failure_taxonomy_76.json"); summ=load(OUT/"audit/failure_taxonomy_summary.json"); inv=load(OUT/"manifests/support_router_inventory_formal.json"); rep=load(OUT/"metrics/o_support_replay_formal.json"); agg=load(OUT/"metrics/support_router_aggregate_formal.json")
    atomic_text(DOC/"PHYSICAL_TO_R_REPORT.md",physical_report(physical)); atomic_text(DOC/"O_SUPPORT_REPORT.md",o_report(tax,summ,inv,rep,agg))
    auto=ROOT/"docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md"; old=auto.read_text(encoding="utf-8")
    section=f"""

## Phase83 — dual-path physical→R and O-support (2026-09-04)

- Phase83 was registered from commit `{commit()}` with Q0/Phase75B rows, 76+76 event denominator, prefixes and evaluator frozen. Outputs use the `/data2/usr_for_deadline/trackocd_phase83` symlink; no DEV+/Q1/public-new/sealed labels were accessed.
- Branch A mapped the Phase82R native temporal-appearance-mean lineage to corrected public rows. Only 1,046/6,213 tracks had usable matches (74/76 event pairs). Exact TRAIN validation p16 R@1 was 0.882735 vs raw 0.893219, mAP 0.847251 vs 0.848374, hard-gap 0.198022 vs 0.189559, with 5 unsafe flips; only one fold was non-decreasing in both R@1/mAP. Event-pair temporal cosine (0.197574) was below raw (0.271772). Decision: `R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT`; no downstream C run.
- Branch B read-only taxonomy reproduced the proposal-pool upper bound source/target/both 72/64/61 and frozen event reliability 49/40/25. Main p16 classes were B=15, D=18, E=36, G=7; no missing proposals. The TRAIN-only hidden-128 router completed 100-step smoke, 500-step fold0 targeted and 4×1000-step formal runs with atomic checkpoints. Learned p16 both-side reliable support was 8/76 (frozen 25/76); support was selected on 46/76 positives and 52/76 negatives. Decision: `O83_FAIL_NO_SUPPORT_GAIN`; no controller/StateMemory/threshold/backbone/sealed path was run.
- All code changes were committed/pushed before execution. One task-owned physical-R PID was stopped for the documented complexity repair; no external process or GPU job was touched and no OOM occurred. GPU 0–3 remained external and GPU4–7 were not needed. Primary artifacts: `outputs/iclr27_phase83/audit/{{support_assignment_callgraph,failure_taxonomy_76,failure_taxonomy_summary,observability_by_prefix}}.json`, `outputs/iclr27_phase83/metrics/{{physical_r_temporal,o_support_replay_formal}}.json`, and `docs/iclr27_phase83/{{PHYSICAL_TO_R_REPORT,O_SUPPORT_REPORT}}.md`.
- Phase83 closes this window's two registered diagnostics without claiming OCD success. Persistent Commit-CT, unchanged-controller C83 and sealed evaluation remain `NOT_RUN`; a later route must repair the support/assignment interface or provide a separately registered legal proposal source rather than tune thresholds or declare retrieval/O oracle success.
"""
    if "## Phase83 — dual-path physical→R and O-support" not in old: atomic_text(auto,old.rstrip()+section+"\n")
    atomic_json(OUT/"audit/phase83_decision.json", {"schema_version":"trackocd.phase83.decision.v1","phase":"Phase83","status":"AUTONOMOUS_10H_COMPLETE_WITH_NEGATIVE_EVIDENCE","decision_code":"P83_DUAL_PATH_R_AND_O_FAIL_NO_DOWNSTREAM","R83":"FAIL","O83":"FAIL","C83":"NOT_RUN","sealed":"NOT_RUN","physical_r_p16":physical["sections"]["exact_mixed"]["prefix"]["16"],"o_support_p16":next(z for z in rep["prefix_summary"] if z["prefix"]==16),"pool_ceiling":summ["pool_upper_bound"],"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"outputs": [str((DOC/"PHYSICAL_TO_R_REPORT.md").resolve()),str((DOC/"O_SUPPORT_REPORT.md").resolve())],"next_action":"hold for a new support/assignment contract; no gate/controller/backbone lottery"})
    atomic_json(OUT/"audit/validation_evidence_ledger.json", {"phase":"Phase83","artifacts":[str((OUT/"audit/failure_taxonomy_76.json").resolve()),str((OUT/"metrics/physical_r_temporal.json").resolve()),str((OUT/"metrics/o_support_replay_formal.json").resolve())],"hashes": {str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [OUT/"audit/failure_taxonomy_76.json",OUT/"metrics/physical_r_temporal.json",OUT/"metrics/o_support_replay_formal.json"]},"status":"COMPLETE","public_dev_q1_sealed_accessed":False})
    atomic_json(OUT/"status.json", {"phase":"Phase83","status":"AUTONOMOUS_10H_COMPLETE_WITH_NEGATIVE_EVIDENCE","run_id":"phase83-20260904","next_action":"hold for a new support/assignment contract; no controller/StateMemory/threshold/backbone lottery","R83":"FAIL","O83":"FAIL","C83":"NOT_RUN","sealed":"NOT_RUN","physical_r":physical["sections"]["exact_mixed"]["prefix"].get("16"),"o_support":next((z for z in rep["prefix_summary"] if z["prefix"]==16), None),"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"task_owned_process_termination":{"pid":17813,"wait_shell_pid":17963,"reason":"quadratic raw-vector recomputation before cache repair","external_processes_touched":False},"reports":[str((DOC/"PHYSICAL_TO_R_REPORT.md").resolve()),str((DOC/"O_SUPPORT_REPORT.md").resolve())]})
    print(json.dumps({"status":"REPORTS_GENERATED","commit":commit(),"reports":[str(DOC/"PHYSICAL_TO_R_REPORT.md"),str(DOC/"O_SUPPORT_REPORT.md")],"decision":str(OUT/"audit/phase83_decision.json")},indent=2))


if __name__ == "__main__": main()
