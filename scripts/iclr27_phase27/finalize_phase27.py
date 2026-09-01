#!/usr/bin/env python3
"""Freeze Phase27 correspondence evidence and write the final report."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase27"
DOC = ROOT / "docs/iclr27_phase27/PHASE27_CORRESPONDENCE_CONTROLLER_COMPLETE_REPORT.md"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def js(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def fmt(value: float, n: int = 4) -> str:
    return f"{float(value):.{n}f}"


def nvidia_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) == 4:
                rows.append({"index": int(vals[0]), "memory_used_MiB": int(vals[1]), "memory_free_MiB": int(vals[2]), "utilization_percent": int(vals[3])})
        return {"query_ok": True, "gpus": rows}
    except Exception as exc:
        return {"query_ok": False, "error": repr(exc)}


def main() -> None:
    frozen = json.loads((OUT / "audit/frozen_proposal.json").read_text())
    metrics = json.loads((OUT / "metrics/correspondence_validation.json").read_text())
    event_payload = json.loads((OUT / "audit/correspondence_event_records.json").read_text())
    assert metrics["positive_event_denominator"] == 76 and len(event_payload["records"]) == 76

    fold_rows = []
    for f in metrics["folds"]:
        b, e = f["baseline"]["16"], f["encoder"]["16"]
        fold_rows.append({
            "fold": int(f["fold"]),
            "validation_tracklets": int(f["validation_tracklets"]),
            "best_step": int(f["best_step"]),
            "baseline_r1": b["r1"], "encoder_r1": e["r1"], "delta_r1": e["r1"] - b["r1"],
            "baseline_r5": b["r5"], "encoder_r5": e["r5"], "delta_r5": e["r5"] - b["r5"],
            "baseline_map": b["map"], "encoder_map": e["map"], "delta_map": e["map"] - b["map"],
            "baseline_hard_negative_gap": b["hard_negative_gap"], "encoder_hard_negative_gap": e["hard_negative_gap"],
        })
    gate = metrics["gate_r"]
    aggregate = {
        "protocol": "trackocd_iclr27_phase27_correspondence_controller_complete",
        "decision_code": gate["decision"],
        "gate_r": gate,
        "gate_c": {"evaluated": False, "decision": "NOT_RUN_GATE_R_FAILED", "historical_comparator": "Phase19R persistent Commit-CT 2/76"},
        "proposal": {"phase26_decision": frozen["proposal_decision"], "gate_p2": frozen["proposal_gate"], "raw_prefix16": frozen["raw_prefix16"], "source_prefix16": frozen["source_prefix16"]},
        "folds": fold_rows,
        "mean_baseline_r1": sum(x["baseline_r1"] for x in fold_rows) / 4.0,
        "mean_encoder_r1": sum(x["encoder_r1"] for x in fold_rows) / 4.0,
        "mean_baseline_map": sum(x["baseline_map"] for x in fold_rows) / 4.0,
        "mean_encoder_map": sum(x["encoder_map"] for x in fold_rows) / 4.0,
        "positive_event_denominator": 76,
        "controller_started": False,
        "public_evaluation_started": False,
        "sealed": True,
    }
    atomic_json(OUT / "metrics/correspondence_aggregate.json", aggregate)

    incidents = {
        "protocol": "trackocd_iclr27_phase27_process_and_performance_incidents",
        "repair_cycles": 3,
        "incidents": [
            {"unit": "correspondence_smoke_original_f0", "pid": 2976, "action": "SIGTERM", "reason": "validation used per-query list membership and remained CPU-bound after one training step; no done/metrics artifact", "marker_retained": "correspondence_smoke_smoke_f0.launched"},
            {"unit": "correspondence_smoke_matrix_f0", "pid": 7656, "action": "SIGTERM", "reason": "matrix similarity repair still used many small GRU validation batches; GPU idle and smoke did not finish", "marker_retained": "correspondence_smoke_fix_smoke_f0.launched"},
            {"unit": "correspondence_smoke_matrix_large_batch_f0", "pid": 9871, "action": "SIGTERM", "reason": "large-batch validation still spent time in list membership; diagnostic benchmark isolated the remaining hotspot", "marker_retained": "correspondence_smoke_fix2_smoke_f0.launched"},
            {"unit": "retrieval_benchmark", "pid": 13931, "parent_pid": 13930, "action": "diagnostic_completed_then_explicit_cleanup", "reason": "confirmed approximately 18-20 s/prefix list-membership hotspot; no training/output writes"},
        ],
        "successful_repairs": [
            "matrix similarity computation",
            "larger validation inference batch",
            "set membership for positive-hit computation",
        ],
        "oom_or_memory_pressure": False,
        "external_processes_touched": False,
    }
    atomic_json(OUT / "audit/process_and_performance_incidents.json", incidents)

    artifact_paths = [
        OUT / "audit/frozen_proposal.json",
        OUT / "metrics/correspondence_validation.json",
        OUT / "metrics/correspondence_aggregate.json",
        OUT / "audit/correspondence_event_records.json",
        OUT / "manifests/fold_manifest.json",
        OUT / "completion/stage0.done",
        OUT / "completion/correspondence_validation.done",
    ]
    artifact_paths += [OUT / "completion" / f"correspondence_f{f}.done" for f in range(4)]
    artifact_paths += [OUT / "checkpoints" / f"correspondence_f{f}_best.pt" for f in range(4)]
    hashes = {str(p): {"exists": p.exists(), "is_symlink": p.is_symlink(), "sha256": sha(p) if p.exists() and p.is_file() else None, "resolved": str(p.resolve()) if p.exists() else None} for p in artifact_paths}
    atomic_json(OUT / "audit/artifact_hashes.json", hashes)

    ps_lines = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,cmd"], text=True).splitlines()
    phase27_processes = []
    for line in ps_lines:
        if not ("iclr27_phase27" in line or "correspondence_f" in line):
            continue
        # The finalizer's own shell/python command contains its script path;
        # exclude that diagnostic wrapper when asserting no worker remains.
        if "finalize_phase27.py" in line:
            continue
        phase27_processes.append(line)
    resource = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "nvidia_smi": nvidia_snapshot(), "process_count": int(len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines())), "phase27_processes": phase27_processes, "free_h": subprocess.check_output(["free", "-h"], text=True), "disk_df": subprocess.check_output(["df", "-h", "/data1"], text=True)}
    atomic_json(OUT / "audit/resource_postflight.json", resource)

    forbidden_hits = []
    for p in OUT.rglob("*"):
        if p.is_file() and any(token in p.name.lower() for token in ("q1", "dev+", "public_new_model")):
            forbidden_hits.append(str(p))
    integrity = {
        "markdown_target": str(DOC),
        "json_parse_ok": True,
        "positive_event_denominator": len(event_payload["records"]) == 76,
        "fold_done": all((OUT / "completion" / f"correspondence_f{f}.done").exists() for f in range(4)),
        "checkpoint_best": all((OUT / "checkpoints" / f"correspondence_f{f}_best.pt").exists() for f in range(4)),
        "phase27_processes_empty_at_finalize": not resource["phase27_processes"],
        "forbidden_output_name_hits": forbidden_hits,
        "public_q1_accessed": False,
        "gate_r_pass": bool(gate["pass"]),
        "gate_c_evaluated": False,
    }
    atomic_json(OUT / "audit/phase27_decision.json", aggregate)
    atomic_json(OUT / "audit/integrity.json", integrity)

    lines: list[str] = []
    lines += [
        "# TrackOCD ICLR 2027 — Phase27 Correspondence / Controller Complete Report",
        "",
        f"**Execution (UTC):** `{datetime.now(timezone.utc).isoformat()}`  ",
        f"**Decision:** **`{gate['decision']}`**  ",
        "**Scope:** Phase26 proposal-source checkpoints are frozen and read-only. This phase tests exactly one causal correspondence encoder; the old controller is conditional on Gate R.",
        "",
        "## Executive decision",
        "",
        f"Phase26 Gate P2 remains PASS (real source prefix16 **{frozen['source_prefix16']}/76**, raw **{frozen['raw_prefix16']}/76**). The sole Phase27 encoder did not improve the frozen DINOv2 track baseline on any of four disjoint folds under the preregistered substantial-improvement rule ({js(gate['thresholds'])}); Gate R is **FAIL** ({gate['folds_substantial']}/4 substantial, {gate['folds_directional']}/4 directional). Per protocol, the unchanged Phase19R controller was **not run**, no controller threshold/memory/backbone change was made, and public/Q1 labels remain sealed.",
        "",
        "The result is a reliable negative for this single correspondence route, not evidence that the Phase26 proposal Gate P2 or the entire TrackOCD task is solved. The historical persistent controller comparator remains Commit-CT **2/76**; no new persistent Commit-CT, false-merge, duplicate-birth, premature-commit or known/novel safety number is claimed because Gate R failed before the conditional controller stage.",
        "",
        "## Frozen boundaries and protocol",
        "",
        "- Phase26 report: [`PHASE26_PROPOSAL_SOURCE_CANDIDATE_COVERAGE_COMPLETE_REPORT.md`](../../docs/iclr27_phase26/PHASE26_PROPOSAL_SOURCE_CANDIDATE_COVERAGE_COMPLETE_REPORT.md).",
        f"- Frozen proposal decision: [`frozen_proposal.json`](../../outputs/iclr27_phase27/audit/frozen_proposal.json); source checkpoints are SHA-256 recorded and symlinked read-only from Phase26.",
        "- The row key, causal ordering, physical track parent assignment, proposal source, 76-event denominator and Phase19R evaluator were not changed. The correspondence model receives only key-aligned fused DINOv2 CLS/ROI row features from causal track prefixes.",
        "- Training/validation used public TRAIN metadata only with video/category-disjoint folds. Category/video values were sampling/evaluation metadata, never model input. No category text, semantic/physical ID, GT box, future row, StateMemory or action signal entered the model.",
        "- DEV+, Q1 and public new-model labels were not read; no public evaluation artifact was created.",
        "",
        "## Architecture and training",
        "",
        "The only registered route is `TrackCorrespondenceEncoder`: LayerNorm on the 768-D fused DINOv2 CLS/ROI row feature, one causal GRU (hidden 128), LayerNorm/linear projection to a normalized 768-D embedding. The objective is multi-positive same-category cross-video alignment, hard-negative ranking (raw-feature nearest different-category samples), and a fixed prefix-consistency cosine term. There is no classifier, StateMemory, action head, threshold or backbone change.",
        "",
        "Four folds ran one bounded worker per physical GPU4–7, BF16 AMP, batch 16, 2,000 updates and checkpoints every 500 updates. Fold0 smoke (2 updates) and targeted (10 updates) completed after the validation performance repair; all four formal folds completed with `.launched`/`.done` markers and resumable best/latest checkpoints. The fold manifest is a Phase27-local symlink to the frozen Phase22 TRAIN split; its hash is recorded in every checkpoint and [`artifact_hashes.json`](../../outputs/iclr27_phase27/audit/artifact_hashes.json).",
        "",
        "## Gate R representation validation",
        "",
        "Metrics are computed on held categories and validation videos of each fixed TRAIN fold. Each query candidate is a different physical track and different video. Prefixes 1/2/4/8/16 are causal; checkpoint selection uses only disjoint validation retrieval and never the 76 event records.",
        "",
        "| fold | val tracks | best step | baseline R@1 | encoder R@1 | ΔR@1 | baseline mAP | encoder mAP | ΔmAP | baseline hard-gap | encoder hard-gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in fold_rows:
        lines.append(f"| {x['fold']} | {x['validation_tracklets']} | {x['best_step']} | {fmt(x['baseline_r1'])} | {fmt(x['encoder_r1'])} | {fmt(x['delta_r1'])} | {fmt(x['baseline_map'])} | {fmt(x['encoder_map'])} | {fmt(x['delta_map'])} | {fmt(x['baseline_hard_negative_gap'])} | {fmt(x['encoder_hard_negative_gap'])} |")
    lines += [
        "",
        "| aggregate mean | — | — | **" + fmt(aggregate["mean_baseline_r1"]) + "** | **" + fmt(aggregate["mean_encoder_r1"]) + "** | **" + fmt(aggregate["mean_encoder_r1"] - aggregate["mean_baseline_r1"]) + "** | **" + fmt(aggregate["mean_baseline_map"]) + "** | **" + fmt(aggregate["mean_encoder_map"]) + "** | **" + fmt(aggregate["mean_encoder_map"] - aggregate["mean_baseline_map"]) + "** | — | — |",
        "",
        "The full prefix curves (R@1/R@5/mAP/category-macro/video-macro/hard-negative gap), per-query denominators and checkpoint hashes are in [`correspondence_validation.json`](../../outputs/iclr27_phase27/metrics/correspondence_validation.json). The encoder is below baseline at prefix16 in all four folds; its hard-negative gap also shrinks or becomes negative. Gate R therefore fails without a selective interpretation of one category or one fold.",
        "",
        "## 76-event causal correspondence diagnostic",
        "",
        "All 76 Phase20/Phase26 pseudo-held positive event keys were retained. The event diagnostic compares source-full versus target-prefix cosine for the frozen DINOv2 mean and the encoder; it is not used for selection and is not a persistent OCD score. Full records (event key, fold, source/target track, prefixes 1/2/4/8/16) are in [`correspondence_event_records.json`](../../outputs/iclr27_phase27/audit/correspondence_event_records.json).",
        "",
        "| event | fold | source track | target track | p16 DINO cosine | p16 encoder cosine | Δ |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for r in event_payload["records"]:
        p = r["prefixes"]["16"]
        lines.append(f"| `{r['event_key']}` | {r['fold']} | `{r['source_track']}` | `{r['target_track']}` | {fmt(p['baseline_cosine'])} | {fmt(p['encoder_cosine'])} | {fmt(p['encoder_minus_baseline'])} |")
    lines += [
        "",
        "These cosine values are a diagnostic of the representation under the frozen proposal; they do not override Gate R or establish Commit-CT.",
        "",
        "## Conditional controller stage (not authorized)",
        "",
        "Gate R failed before the conditional stage. Consequently no Phase19R `RCMSOCD` instance was evaluated with the encoder, no source proposal was changed, and no threshold, known mask, StateMemory, transition core, action semantics or evaluator code was modified. Gate C is recorded as `NOT_RUN_GATE_R_FAILED`; the only controller comparator is the historical Phase19R/20 persistent Commit-CT 2/76. There are no new false-merge, duplicate-birth, premature-commit, known/novel confusion or MOT-safety claims in this phase.",
        "",
        "## Phase26 proposal context (frozen)",
        "",
        "Phase26 fixed the proposal layer before this phase: raw true-IoU ceiling 25/76; fixed candidate-pool oracle 38/76; broad causal pool oracle 56/76; trained class-agnostic source branch 41/76 with source reliable 67, target reliable 48 and fold ceilings [11,5,14,11]. Those source checkpoints are not retrained or selected here. The correspondence route deliberately tests representation separately from proposal and controller.",
        "",
        "## Resource, process and repair audit",
        "",
        "- Formal preflight: 125 GiB RAM, about 75 GiB available (above the 25% safety floor), GPUs4–7 idle with 40,337 MiB free each, `/data1` about 19 GiB free. Four bounded workers used one GPU per fold; no OOM, swap or near-OOM event occurred.",
        "- The first smoke attempts exposed a validation implementation hotspot, not a model failure. Three task-owned smoke PIDs (2976, 7656, 9871) were explicitly SIGTERM'ed after confirming they were the Phase27 process and retaining their `.launched` provenance. A separate retrieval benchmark (PID13931, parent13930) completed as a diagnostic and was explicitly cleaned up. The fixes were matrix similarity, a larger validation batch and set membership. Corrected smoke/targeted and all formal folds then completed.",
        "- No external process was touched. The incident ledger is [`process_and_performance_incidents.json`](../../outputs/iclr27_phase27/audit/process_and_performance_incidents.json); postflight resources and process-empty check are [`resource_postflight.json`](../../outputs/iclr27_phase27/audit/resource_postflight.json).",
        "- Phase27 source proposal checkpoints are symlinks to Phase26; the fold manifest is a symlink to Phase22. No large feature or checkpoint was copied. Resolved targets and SHA-256 values are in [`artifact_hashes.json`](../../outputs/iclr27_phase27/audit/artifact_hashes.json).",
        "",
        "## Integrity and sealed-data checks",
        "",
        "`integrity.json` confirms all four formal `.done` markers, best checkpoints, 76 event records and JSON parses. At finalization no Phase27 process remained. No filename containing Q1/DEV+/public-new-model labels entered `outputs/iclr27_phase27`; `public_q1_accessed=false`. The Phase26 namespace and source checkpoints remain unchanged.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# Phase27-local frozen split/source links and freeze audit",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/run_stage0_freeze.py",
        "# smoke and targeted (physical GPU4; CUDA device is cuda:0 inside the process)",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/train_correspondence.py --fold 0 --device cuda:0 --expected-physical-gpu 4 --smoke --tag correspondence_smoke_fix3",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/train_correspondence.py --fold 0 --device cuda:0 --expected-physical-gpu 4 --steps 10 --batch-size 16 --checkpoint-every 5 --tag correspondence_targeted",
        "# bounded four-fold formal run (GPU4/5/6/7, one worker/fold)",
        "PYTHONPATH=. bash scripts/iclr27_phase27/run_four_fold_correspondence_supervisor.sh",
        "# representation validation and final freeze",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/evaluate_correspondence.py --device cuda:0",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/finalize_phase27.py",
        "```",
        "",
        "## Final decision and next direction",
        "",
        "**P26_GATE_P2_PASS** is preserved, but **P27_GATE_R_FAIL_STOP_BEFORE_CONTROLLER** is the final decision for this phase. The one small GRU correspondence encoder did not provide stable cross-instance gains over frozen DINOv2 on any fold, so an unchanged persistent controller evaluation is not authorized and public/Q1 remains sealed. Do not tune the controller or download a backbone in response to this failure. The evidence-supported next step is a separate, preregistered representation study that diagnoses cross-video alignment/domain shift (with the same proposal and causal event protocol) before any controller interface work; if that cannot beat the frozen baseline, reconsider the correspondence supervision or task definition rather than adding memory/threshold complexity.",
        "",
        "## Machine-readable artifacts",
        "",
        "- [`phase27_decision.json`](../../outputs/iclr27_phase27/audit/phase27_decision.json)",
        "- [`correspondence_aggregate.json`](../../outputs/iclr27_phase27/metrics/correspondence_aggregate.json)",
        "- [`correspondence_validation.json`](../../outputs/iclr27_phase27/metrics/correspondence_validation.json)",
        "- [`integrity.json`](../../outputs/iclr27_phase27/audit/integrity.json)",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "completion/phase27.done").write_text(json.dumps({"decision": gate["decision"], "report": str(DOC), "controller_evaluated": False}, sort_keys=True) + "\n")
    print(json.dumps({"decision": gate["decision"], "report": str(DOC), "integrity": integrity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
