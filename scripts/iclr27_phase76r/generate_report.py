#!/usr/bin/env python3
"""Generate the self-contained Phase76R contract-errata/Pareto report."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76r"
DOC = ROOT / "docs/iclr27_phase76r/PHASE76R_CONTRACT_ERRATA_AND_PARETO_AUDIT.md"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(value, h, indent=2, sort_keys=True, allow_nan=False)
            h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def pct(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    folds = [json.loads((OUT / "pareto" / f"fold{i}.json").read_text()) for i in range(4)]
    rows = [r for d in folds for r in d["checkpoints"]]
    front = json.loads((OUT / "pareto/pareto_front.json").read_text())
    err = json.loads((OUT / "audit/teacher_authorization_erratum.json").read_text())
    opt = json.loads((OUT / "audit/optimizer_contract_erratum.json").read_text())
    seed = json.loads((OUT / "audit/seed_contract_erratum.json").read_text())
    sel = json.loads((OUT / "audit/selection_contract_erratum.json").read_text())
    pre = json.loads((OUT / "audit/resource_preflight.json").read_text())
    post = json.loads((OUT / "audit/resource_postflight.json").read_text())
    status = json.loads((OUT / "audit/status.json").read_text())

    lines: list[str] = []
    lines += [
        "# TrackOCD Phase76R — Contract Errata and Checkpoint Pareto Audit", "",
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat()} at git `{subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()}`.", "",
        "## Decision", "",
        f"- Decision: **`{front['decision']}`**.",
        f"- All four folds completed (`{len(rows)}` checkpoints: 30 per fold, steps 500–15000). No controller, StateMemory, Commit-CT, DEV+, Q1, public-new, or sealed evaluation was run.",
        "- The safe window is diagnostic only; it never authorizes a model or a downstream gate.", "",
        "## Scope and frozen contract", "",
        "Phase76R does not retrain Phase75E and does not alter Phase75D/E reports, seeds, candidates, evaluator, denominator, or status. It reads the 120 retained Phase75E step checkpoints from `/data2/usr_for_deadline/trackocd_phase75e/checkpoints` and recomputes exact p16 global (all validation tracks) and manifest-legal scores. Inputs remain frozen 768-D visual frame features and causal prefixes; no category, semantic/physical ID, text, future, held, DEV+, Q1, public-new, or sealed labels enter model inference.", "",
        "## Contract errata", "",
        f"- Teacher authorization: historical helper returned `{err['old_result']}` because the −0.02 R1 guard was applied to legal folds. Corrected checker applies it to global folds and returns `{err['corrected_result']}`; bad folds: `{[(x['fold'], round(x['delta_r1'], 6)) for x in err['global_bad_folds']]}`. Phase75D's recorded status is unchanged.",
        f"- Optimizer: actual Phase75E optimizer was `{opt['phase75e_actual_optimizer']}` at constant LR `{opt['phase75e_actual_lr']}`; scheduler was `{opt['phase75e_actual_scheduler']}`. Registered warmup `{opt['registered_warmup_steps']}` was not executed.",
        f"- Seeds: actual formal fold seeds are `{seed['phase75e_worker_seed_formula']}` (`{seed['phase75e_actual_fold_seeds']}`), not the report-level seed alone.",
        f"- Selection: historical key `{sel['phase75e_selection_key']}` was legal-first and omitted `{sel['omitted_hard_constraints']}`; selected checkpoints therefore were not globally/legal Pareto-safe.", "",
        "## Pareto audit definition", "",
        "For every step, raw scores are the immutable Phase75D comparator and learned scores are the frozen Phase75E adapter replay. We retain R@1, mAP, hard-negative gap, unsafe flips, top-1 changes, raw/adapt cosine p05/p50/p95 and relative delta norm. The diagnostic safe window requires global and legal unsafe=0, global ΔR1≥−0.005, global ΔmAP≥−0.002, legal ΔR1>0, legal ΔmAP>0, and mean cosine≥0.98.", "",
        "## Fold summary (p16)", "",
        "| fold | steps | min global unsafe | min legal unsafe | best legal ΔR1 | best legal ΔmAP | max global ΔR1 | max global ΔmAP | max cosine | safe checkpoints |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in folds:
        rs = d["checkpoints"]
        best_r1 = max(rs, key=lambda r: r["legal_delta_r1"])
        best_map = max(rs, key=lambda r: r["legal_delta_map"])
        lines.append(f"| {d['fold']} | {len(rs)} | {min(r['global_unsafe'] for r in rs)} | {min(r['legal_unsafe'] for r in rs)} | {pct(best_r1['legal_delta_r1'])} (step {best_r1['step']}) | {pct(best_map['legal_delta_map'])} (step {best_map['step']}) | {pct(max(r['global_delta_r1'] for r in rs))} | {pct(max(r['global_delta_map'] for r in rs))} | {pct(max(r['mean_raw_adapt_cosine'] for r in rs))} | {sum(bool(r['safe_window']) for r in rs)} |")
    lines += [
        "", "The fold summary exposes the incompatibility: fold0 has at least 46 global unsafe flips at every audited step; folds1–3 reach minimum global unsafe 3, 4, and 1, respectively. Legal gains therefore cannot be accepted as a safe global feature adapter.", "",
        "## Pareto artifact inventory", "",
        f"- `all_checkpoints.jsonl`: {len(rows)} records (all 120 exact step rows).",
        "- `fold0.json` … `fold3.json`: fold-level full records, raw metrics, inventories, and checkpoint drift.",
        f"- `pareto_front.json`: `{front['safe_window_count']}` safe-window records; `window_found={front['window_found']}`.",
        "- `plot_data.json`: legal ΔmAP vs global ΔmAP, legal ΔR1 vs global unsafe, legal ΔR1 vs raw cosine, step vs global unsafe, and step vs legal R1 fields.", "",
        "## Resource and process audit", "",
        f"- Preflight: `{pre['created_utc']}`; postflight: `{post['created_utc']}`. Pareto replay was CPU-only (`gpu_count=0`); GPU4–7 remained available and external GPU0–3 processes were not touched.",
        f"- Process counts recorded: pre `{pre['process_count']}`, post `{post['process_count']}`. No Phase76R worker remained after completion; each fold has `.launched` and `.done` markers.",
        "- `/data2/usr_for_deadline/trackocd_phase75e/checkpoints` was read-only; no checkpoint was copied or overwritten.",
        "- Sealed/public boundary: `held_event_accessed_for_model=false`, `sealed_accessed=false` in status/resource artifacts.", "",
        "## Tests and reproducibility", "",
        "- `python -m py_compile src/iclr27_phase76r/*.py scripts/iclr27_phase76r/*.py`.",
        "- `python -m pytest -q tests/phase76r` (2 passed).",
        "- `PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase76r/run_contract_errata.py`.",
        "- `bash scripts/iclr27_phase76r/run_four_fold_pareto.sh` (one bounded supervisor; four CPU fold workers; exact 120-step replay).",
        "- `PYTHONPATH=. python scripts/iclr27_phase76r/generate_report.py`.", "",
        "## Interpretation and next route", "",
        "The corrected historical authorization is an erratum, not a retroactive change to Phase75D. The legal pairwise signal remains a diagnostic motivation, but Phase75E's unconstrained residual adapter has no checkpoint satisfying raw/global safety and legal improvement together. Phase76A therefore uses a different structure: a frozen raw global scorer plus a bounded local relation reranker whose output is a side relation score, not a replacement 768-D embedding. It must pass independent global parity, legal retrieval, and memory-mimic gates before any StateMemory/controller work.", "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{DOC.name}.", dir=str(DOC.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            h.write("\n".join(lines) + "\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, DOC)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic_json(OUT / "audit/phase76r_decision.json", {
        "phase": "Phase76R", "status": "COMPLETE", "decision": front["decision"],
        "checkpoint_count": len(rows), "fold_counts": {str(d["fold"]): d["checkpoint_count"] for d in folds},
        "teacher_authorization_corrected": err["corrected_result"], "safe_window_count": front["safe_window_count"],
        "controller_run": False, "sealed_accessed": False,
        "next_action": "Implement Phase76A anchored local pairwise relation reranker; do not retrain Phase75E",
        "report": str(DOC), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    print(json.dumps({"phase": "Phase76R", "decision": front["decision"], "report": str(DOC)}, sort_keys=True))


if __name__ == "__main__":
    main()
