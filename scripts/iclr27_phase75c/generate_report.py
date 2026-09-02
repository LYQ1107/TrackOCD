#!/usr/bin/env python3
"""Render the self-contained Phase75C report from frozen metrics.

The renderer deliberately reads only the Phase75C TRAIN-disjoint metric file
and already audited local reports.  It never opens the 152-event evaluator or
any sealed/DEV+/Q1 artifact.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75c"
METRICS = OUT / "metrics/r_retrieval.json"
REPORT = ROOT / "docs/iclr27_phase75c/PHASE75C_GROUNDED_CORRESPONDENCE_REPORT.md"
LITERATURE = ROOT / "outputs/literature_review/phase75c_grounded_correspondence.json"
PROGRESS = ROOT / "docs/AUTONOMOUS_TRACKOCD_ICLR_PROGRESS_REPORT.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def pct(value: Any) -> str:
    return f"{float(value):.4f}"


def literature_payload() -> dict[str, Any]:
    return {
        "phase": "Phase75C",
        "selected": {
            "method": "Grounded Correspondence / Rethinking Temporal Consistency in Video Object-Centric Learning",
            "paper_url": "https://arxiv.org/abs/2605.03650",
            "repo_url": "https://github.com/LiZhYun/ICML2026-RethinkingOCL",
            "commit": "5d345268797425558b449337519af3ab24aeb6f1",
            "license": "MIT (repository LICENSE)",
            "component": "frozen DINOv2 saliency/feature correspondence with deterministic parameter-free matching",
            "integration_point": "causal prefix aggregation before a future controller compatibility route",
            "why_selected": "It tests temporal correspondence without another learned GRU/MLP or forbidden text/ID inputs.",
            "not_a_solution": "The official object-centric code is not a drop-in TAO MOT or TrackOCD controller and is evaluated only as a bounded representation route.",
        },
        "audited_candidates": [
            {
                "method": "OVTR",
                "paper_url": "https://arxiv.org/abs/2503.10616",
                "repo_url": "https://github.com/jinyanglii/OVTR",
                "commit": "500e72c19bf5f7f8717546911a5639fdc26bfee5",
                "license": "MIT",
                "boundary": "causal persistent-query MOT reference; released open-vocabulary branch uses CLIP/text/category logits, so text branch is isolated",
            },
            {
                "method": "MOTIP-2",
                "paper_url": "https://arxiv.org/abs/2403.16848",
                "repo_url": "https://github.com/GISer-WB/MOTIP-2",
                "commit": "012856c1dc13b324064e79339ae71054518d1b5e",
                "license": "repository terms audited in Phase57",
                "boundary": "physical trajectory/query memory reference; identity prompts are not semantic correspondence inputs",
            },
            {
                "method": "ObjectRelator",
                "paper_url": "https://arxiv.org/abs/2411.19083",
                "repo_url": "https://github.com/insait-institute/ObjectRelator",
                "commit": "25ecbc086cc812304de97764aa21f4bb8e0e6360",
                "license": "repository terms audited in Phase57",
                "boundary": "paired/static cross-view object correspondence; no causal MOT lifecycle and text modality is not used",
            },
            {
                "method": "C3Po",
                "paper_url": "https://arxiv.org/abs/2511.18559",
                "repo_url": "https://github.com/c3po-correspondence/C3Po",
                "commit": "21254a078435451e99d2feabd5db9334c02d8483",
                "license": "repository terms audited in Phase57",
                "boundary": "static geometric correspondence with DUSt3R/pointmap assumptions; not an online semantic controller",
            },
            {
                "method": "MASA",
                "paper_url": "https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.pdf",
                "repo_url": "https://github.com/siyuanliii/masa",
                "commit": "c5472b9c7615f35abdf1188cb1a0c5408fe50d66",
                "license": "repository terms audited in Phase57",
                "boundary": "association adapter over external proposals; does not supply the required prior-video causal semantic state",
            },
        ],
        "source": "Phase57/Phase51 local official-method audits plus the selected repository HEAD audit",
    }


def render(metrics: dict[str, Any]) -> str:
    agg = metrics["aggregate"]
    folds = metrics["folds"]
    gate = metrics["gate_r"]
    rows = gate["fold_rows"]
    lines = [
        "# Phase75C — Grounded Correspondence R report",
        "",
        f"Generated UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        f"**{gate['decision']}**.  This is a frozen, zero-parameter, TRAIN-disjoint representation diagnostic. "
        "Because the registered R gate is not met, unchanged-controller C replay is **NOT RUN** and no sealed/public/Q1 evaluation is authorized.",
        "",
        "The route is not interpreted as detector failure or OCD=0. Phase75B established the physical observability anchor; this route asks only whether deterministic causal aggregation improves the already strong raw DINOv2 correspondence baseline.",
        "",
        "## Frozen provenance and boundaries",
        "",
        f"- Phase75B positive p16 observability: **25/76 both reliable** (raw Q0 TRAIN event replay; O gate passed at its preregistered anchor).",
        f"- Input CSV SHA256: `{metrics['input_csv_sha256']}`; feature NPZ SHA256: `{metrics['input_feature_sha256']}`.",
        f"- Aligned feature rows: {metrics['input_rows']}; labelled tracklets available to TRAIN validation: {metrics['input_tracklets_with_labels'] }.",
        "- Feature: exact five-field-key-aligned Phase15S frozen DINOv2 CLS/ROI, fused as 0.8 CLS + 0.2 ROI. Metadata labels are used only to score disjoint TRAIN validation positives/negatives.",
        "- Prefixes: 1, 2, 4, 8, 16; candidates are different physical tracks in different videos.",
        "- Not read as model inputs: 152-event evaluator outcomes, held event GT, DEV+, Q1, public-new labels, future rows/tracks, category text, physical IDs or semantic IDs.",
        "",
        "## Official method audit",
        "",
        "The selected reference is [Grounded Correspondence / Rethinking Temporal Consistency](https://github.com/LiZhYun/ICML2026-RethinkingOCL), associated with [arXiv:2605.03650](https://arxiv.org/abs/2605.03650), repository commit `5d345268797425558b449337519af3ab24aeb6f1`, MIT. Its frozen-feature correspondence idea is used only as a parameter-free causal aggregation diagnostic. It is not claimed to solve TrackOCD.",
        "",
        "Other audited references are retained as boundaries: OVTR (persistent-query MOT but CLIP/text category branch), MOTIP-2/MeMOTR/MOTR (physical-ID trajectory supervision), ObjectRelator and C3Po (paired/static correspondence), and MASA (external-proposal association adapter). None combines the no-text/no-ID prior-video semantic state and persistent Commit/Defer contract.",
        "",
        "## Method and R protocol",
        "",
        "For each causal prefix, every normalized fused frame vector receives a frozen within-prefix agreement weight (temperature 0.20); the weighted 768-D vector is L2-normalized. A deterministic one-to-one Hungarian score is emitted only as a diagnostic. There are zero learnable parameters, no checkpoint, no threshold sweep, and no controller/state-memory modification.",
        "",
        "### Prefix aggregate (mean over the four fixed TRAIN-disjoint folds)",
        "",
        "| prefix | raw R@1 | grounded R@1 | raw mAP | grounded mAP | raw hard gap | grounded hard gap | unsafe flips | queries |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in (1, 2, 4, 8, 16):
        x = agg[str(p)]
        lines.append(f"| {p} | {pct(x['raw']['r1'])} | {pct(x['grounded']['r1'])} | {pct(x['raw']['map'])} | {pct(x['grounded']['map'])} | {pct(x['raw']['hard_negative_gap'])} | {pct(x['grounded']['hard_negative_gap'])} | {x['unsafe_flip_count']} ({pct(x['unsafe_flip_rate'])}) | {x['queries']} |")
    lines += [
        "",
        "### Prefix-16 per-fold gate rows",
        "",
        "| fold | raw R@1 | grounded R@1 | ΔR@1 | raw mAP | grounded mAP | ΔmAP | raw hard gap | grounded hard gap | unsafe flips | substantial |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(f"| {r['fold']} | {pct(r['raw_r1'])} | {pct(r['grounded_r1'])} | {pct(r['delta_r1'])} | {pct(r['raw_map'])} | {pct(r['grounded_map'])} | {pct(r['delta_map'])} | {pct(r['raw_hard_gap'])} | {pct(r['grounded_hard_gap'])} | {r['unsafe_flip']} | {r['substantial']} |")
    lines += [
        "",
        f"Registered thresholds: ΔR@1 ≥ {gate['thresholds']['r1_delta']:.2f}, ΔmAP ≥ {gate['thresholds']['map_delta']:.2f}, at least {gate['thresholds']['minimum_folds']} folds, unsafe flips = {gate['thresholds']['unsafe_flip']}. The observed substantial-fold count is {gate['folds_substantial']}; directional-fold count is {gate['folds_directional']}; unsafe flips are {gate['unsafe_flip_count']}.",
        "",
        "### Fold/prefix validation inventory",
        "",
        "| fold | validation tracklets | fit multi-positive episodes | fit hard-negative episodes | validation categories | validation videos |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        lines.append(f"| {f['fold']} | {f['validation_tracklets']} | {f['fit_positive_episodes']} | {f['fit_hard_negative_episodes']} | {len(f['validation_categories'])} | {len(f['validation_videos'])} |")
    lines += [
        "",
        "Grounded-to-prefix16 cosine consistency is retained in the machine-readable metric file under each fold. The deterministic Hungarian score is likewise diagnostic only and cannot select a model or alter the event denominator.",
        "",
        "## Resource and repair record",
        "",
        "- This R route is CPU-only and zero-parameter; no training worker or GPU was launched. The contract smoke is bounded and checks finite 768-D output, exact prefix causality, and forbidden-input metadata.",
        "- Phase75B resource preflight remained the physical-stream record: ~125 GiB RAM with >115 GiB available, GPU workers limited to the registered Q0 replay, `/data1` capacity monitored and large replay artifacts archived by symlink under `/data2/usr_for_deadline/trackocd_phase75b/`.",
        "- Earlier accidental CPU matching snippets (PIDs 23159, 24457–24458, 25434–25435, 29310–29311) were task-owned diagnostics, explicitly terminated, produced no artifacts, and are retained in `research_log.md`; no external process was touched.",
        "- No OOM, swap, duplicate supervisor, or sealed-data access occurred for Phase75C.",
        "",
        "## C / sealed status",
        "",
        "`event_evaluator.status = NOT_RUN` because R did not pass. This is intentional protocol compliance: no controller, StateMemory, threshold or Commit-CT claim is made for this route, and no public/sealed labels were read.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase75c/run_contract_smoke.py",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase75c/run_r_retrieval.py --run-id phase75c-r-20260902-r1",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase75c/generate_report.py",
        "```",
        "Outputs: `outputs/iclr27_phase75c/metrics/r_retrieval.json`, `outputs/iclr27_phase75c/audit/gate_rows.json`, `outputs/iclr27_phase75c/audit/contract_smoke.json`, and `outputs/iclr27_phase75c/completion/r_retrieval.done`. The metrics file SHA256 is computed after generation and recorded in the status audit.",
        "",
        "## Next authorized action",
        "",
        "Preserve Q0/Phase75B physical lineage and the R negative evidence. Do not tune thresholds, memory or controller and do not open a second representation/backbone lottery. Desktop ChatGPT must authorize a distinct, evidence-backed route before any controller compatibility replay; the current Grounded route cannot be used to claim TrackOCD success.",
        "",
    ]
    return "\n".join(lines)


def update_progress(metrics: dict[str, Any]) -> None:
    marker = "## Phase75C grounded correspondence R (2026-09-02)"
    old = PROGRESS.read_text(encoding="utf-8") if PROGRESS.exists() else "# TrackOCD progress\n"
    if marker in old:
        return
    gate = metrics["gate_r"]
    section = "\n" + marker + "\n\n" + (
        f"Phase75C ran the single registered Grounded Correspondence route on four TRAIN-derived video/category-disjoint validation folds. "
        f"The frozen consistency-weighted DINOv2 aggregation produced {gate['folds_substantial']}/4 substantial folds and {gate['unsafe_flip_count']} unsafe flips; decision **{gate['decision']}**. "
        "Because Gate R failed, controller compatibility and sealed evaluation remain unrun. The route is a bounded representation result, not OCD=0 or final MOT+OCD."
    ) + "\n"
    atomic_text(PROGRESS, old.rstrip() + "\n" + section)


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    atomic_json(LITERATURE, literature_payload())
    atomic_text(REPORT, render(metrics))
    update_progress(metrics)
    audit = {
        "status": "PASS_REPORT_RENDERED",
        "metrics_sha256": sha256(METRICS),
        "report_sha256": sha256(REPORT),
        "literature_sha256": sha256(LITERATURE),
        "gate_decision": metrics["gate_r"]["decision"],
        "controller_status": metrics["event_evaluator"]["status"],
    }
    atomic_json(OUT / "audit/report_integrity.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
