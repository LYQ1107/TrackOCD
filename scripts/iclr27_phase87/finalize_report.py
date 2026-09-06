#!/usr/bin/env python3
"""Generate the self-contained Phase87 report and machine-readable decision."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"
DOC = ROOT / "docs" / "iclr27_phase87" / "PHASE87_AUTONOMOUS_RESEARCH_REPORT.md"


def read(path: pathlib.Path):
    return json.loads(path.read_text())


def metric_rows(prefix: str, key: str = "metrics") -> list[dict]:
    rows = []
    for fold in range(4):
        data = read(OUT / "metrics" / f"{prefix}_f{fold}.json")
        rows.append({"fold": fold, **data[key]})
    return rows


def fmt(x, digits: int = 4):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key, "NA")) for key, _ in columns) + " |")
    return "\n".join(out)


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reg = read(OUT / "audit" / "window_registration.json")
    c0 = read(OUT / "audit" / "phase87_c0_decision.json")
    r1 = read(OUT / "audit" / "phase87_repair1_decision.json")
    c1 = read(OUT / "audit" / "phase87_c1_decision.json")
    contracts = read(OUT / "audit" / "contract_checks.json")
    event_manifest = read(OUT / "audit" / "causal_event_build.json")
    rf = read(OUT / "audit" / "phase86_rf_causal_correction.json")
    u2 = read(OUT / "audit" / "phase86_u2_evalmode_correction.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    c0_train = read(OUT / "metrics" / "c0_aggregate.json")["train_validation"]
    c1_val = metric_rows("c1_val")
    c0_diag = [{"fold": i, **r} for i, r in enumerate(c0["diagnostic_76_plus_76"]["folds"])]
    r1_diag = [{"fold": i, **r} for i, r in enumerate(r1["repair1"]["folds"])]
    c1_diag = [{"fold": i, **r} for i, r in enumerate(c1["aggregate"]["folds"])]
    c0_ckpt = [read(OUT / "metrics" / f"c0_formal_f{i}.json") for i in range(4)]
    r1_ckpt = [read(OUT / "metrics" / f"c0_repair1_f{i}.json") for i in range(4)]
    c1_ckpt = [read(OUT / "metrics" / f"c1_formal_f{i}.json") for i in range(4)]

    def ckpt_table(rows):
        return table(rows, [("fold", "fold"), ("updates", "updates"), ("loss_first", "loss first"), ("loss_last", "loss last"), ("grad_norm_mean", "mean grad norm"), ("checkpoint_sha256", "checkpoint SHA256")])

    report = f"""# TrackOCD Phase87 — Causal Persistent OCD Controller Redesign

**Status:** `PHASE87_REGISTERED_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL`  
**Generated (UTC):** {dt.datetime.now(dt.timezone.utc).isoformat()}  
**Window:** {reg.get('start_utc', reg.get('started_utc', 'see window_registration.json'))} → {reg['deadline_utc']}  
**Code head at report generation:** `{head}`

## Executive decision

Phase87 tested the one registered causal controller redesign on the frozen
Phase26/Phase86 physical feature stream. C0 learned a causal state/action policy
but failed safety; the one registered false-merge repair traded a small safety
improvement for lower precision; C1 added only legal prior-track support
features and still failed the pre-registered gate. The final C1 diagnostic was
18/76 Commit-CT, existing precision **0.1731**, and negative false-merge
**0.0789**. The precision and known-stream conditions therefore fail. No
unchanged-controller compatibility, public evaluation, or sealed evaluation
was run. These are controller/interface negative results, not a claim that the
entire TrackOCD task is impossible.

**Decision code:** `P87_CONTROLLER_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL_NO_SEALED`

## 1. Frozen boundary and Phase86 carry-over

- Phase86 remained read-only. No DEV+, Q1, public-new or sealed label was used
  for training, checkpoint selection, or inference; the 76+76 event labels were
  used only for the registered diagnostic scoring after models were frozen.
- No future row/track, category text, semantic ID, or physical ID entered a
  model tensor. Physical IDs were used only for causal eligibility bookkeeping.
- The Phase86 RF correction was independently re-evaluated with observed-step
  ordering; its exact p16 metrics and fold rows are preserved in
  `audit/phase86_rf_causal_correction.json`, with decision
  `RF_PHASE86_OBSERVED_STEP_CORRECTION_NEGATIVE`. All six frozen U2 eval-mode
  checkpoints had rescue=0 and harm=0; decision
  `U2_COMPRESSED_FEATURE_COLLAPSE_CONFIRMED`.
- The frozen physical stream, row order, event denominator, prefixes
  {{1,2,4,8,16}}, and feature manifest hash were not changed.

## 2. Architecture and causal contract

`raw 768-D track rows + 15 geometry fields → one-layer GRU (256) → bounded
0.05*tanh residual (768-D semantic state) → StateMemoryV2 (16 states × 4
prototypes) → relation features → joint EXISTING/NEW/DEFER/RESET logits`.

Candidate eligibility is exactly `birth_video != current_video AND
birth_track != current_track`. Target evidence is an EMA (0.70 previous,
0.30 current), with a per-track session and a transactional global memory
update only at track end. EXISTING updates the selected state; NEW creates a
state; DEFER/RESET and unresolved tracks do not mutate global memory. Empty
prototype slots are masked at the joint action interface. There is no
threshold-chain or `risk_decode` path. The full contract is in
`docs/iclr27_phase87/PHASE87_PREREGISTRATION.md`.

The targeted contract artifact is `{contracts['status']}`: joint action dim
{contracts['joint_action_dim']}, NEW/DEFER/RESET present, source-before-target
and monotonic prefix checks passed, candidate video/track AND passed, and
non-zero training gradient passed.

## 3. TRAIN event materialization

The manifest excluded 91 videos from the historical event observability file,
kept strict source-before-target chronology, and made category/ID fields
metadata-only. Counts were:

| fold | fit/train events | train positive/negative | validation events | validation positive/negative |
|---:|---:|---:|---:|---:|
| 0 | 33 tracks / 51 | 21 / 30 | 1628 | 748 / 880 |
| 1 | 524 tracks / 950 | 432 / 518 | 191 | 91 / 100 |
| 2 | 505 tracks / 916 | 417 / 499 | 164 | 75 / 89 |
| 3 | 554 tracks / 1011 | 463 / 548 | 82 | 37 / 45 |

No held-event overlap or forbidden model input was found. The baseline artifact
`outputs/iclr27_phase87/metrics/phase19r_baseline_reference.json` records the
safe all-DEFER TRAIN comparator and the historical Phase19R lineage; the old
Phase19R event schema is not silently treated as equal to this new manifest.

## 4. Resource and execution audit

The preflight reserved fold0–3 on GPU5–8, with a four-worker bound and estimated
4 GB peak RSS per worker against a 125 GB host (at least 25% available-memory
floor). Large checkpoints and outputs use the symlink
`outputs/iclr27_phase87 → /data2/usr_for_deadline/trackocd_phase87/project_outputs`.
External GPU0–4 processes were not touched. The only implementation incident
was a desktop wrapper leaving duplicate contract-check processes; explicit
task-owned PIDs were terminated, the valid artifact was kept, and no training
worker or external task was killed. The held evaluator initially failed on a
manifest field alias (`target_tracklet_key` vs `target_track_key`); a normalizer
was added, smoke and targeted replay passed, and the original failure log was
preserved. There was no OOM, no broad kill, and no seed/denominator change.

## 5. Training results

### C0 causal persistent controller (20,000 updates/fold)

{ckpt_table(c0_ckpt)}

TRAIN validation improved over all-DEFER in all folds, but safety was not
stable. The fold validation summary is in
`outputs/iclr27_phase87/metrics/c0_val_f{{0..3}}.json`.

### Repair1 (false-merge weight 2 → 3)

{ckpt_table(r1_ckpt)}

This was the only registered C0 repair. It was not selected from held events.

### C1 support integration (initialized from repair1)

{ckpt_table(c1_ckpt)}

C1 support inputs were eight causal values: source/query cosine, fixed
candidate-count normalization, source length, source variance, observation
quality, and prefix history consistency. They contain no category or identity
shortcut. C1 smoke and fold0 targeted tests completed before the four-fold
run.

## 6. TRAIN validation and 76+76 diagnostic replay

### C0 TRAIN validation

{table([{"fold": x["fold"], "CT": x["metrics"]["commit_ct_correct"], "eligible": x["metrics"]["commit_ct_eligible"], "precision": x["metrics"]["existing_precision"], "false_merge": x["metrics"]["negative_false_merge_rate"], "premature": x["metrics"]["premature_rate"], "unresolved": x["metrics"]["unresolved_rate"]} for x in c0_train], [("fold", "fold"), ("CT", "CT"), ("eligible", "eligible"), ("precision", "existing precision"), ("false_merge", "negative false merge"), ("premature", "premature"), ("unresolved", "unresolved")])}

### Frozen diagnostic event replay (positive=76, negative=76)

| route | fold | Commit-CT | existing precision | negative false merge | premature | unresolved |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| C0 | {r['fold']} | {r['commit_ct_correct']}/{r['commit_ct_eligible']} | {r['existing_precision']:.4f} | {r['negative_false_merge_rate']:.4f} | {r['premature_rate']:.4f} | {r['unresolved_rate']:.4f} |" for r in c0_diag) + "\n" + "\n".join(f"| repair1 | {r['fold']} | {r['commit_ct_correct']}/{r['commit_ct_eligible']} | {r['existing_precision']:.4f} | {r['negative_false_merge_rate']:.4f} | {r['premature_rate']:.4f} | {r['unresolved_rate']:.4f} |" for r in r1_diag) + "\n" + "\n".join(f"| C1 support | {r['fold']} | {r['commit_ct_correct']}/{r['commit_ct_eligible']} | {r['existing_precision']:.4f} | {r['negative_false_merge_rate']:.4f} | {r['premature_rate']:.4f} | {r['unresolved_rate']:.4f} |" for r in c1_diag) + f"""

Aggregate comparison:

| route | Commit-CT | existing precision | negative false merge | category coverage | video coverage |
|---|---:|---:|---:|---:|---:|
| C0 | {c0['diagnostic_76_plus_76']['commit_ct_correct']}/76 | {c0['diagnostic_76_plus_76']['existing_precision']:.4f} | {c0['diagnostic_76_plus_76']['negative_false_merge_rate']:.4f} | {c0['diagnostic_76_plus_76']['category_coverage']} | {c0['diagnostic_76_plus_76']['video_coverage']} |
| repair1 | {r1['repair1']['commit_ct_correct']}/76 | {r1['repair1']['existing_precision']:.4f} | {r1['repair1']['negative_false_merge_rate']:.4f} | {r1['repair1']['category_coverage']} | {r1['repair1']['video_coverage']} |
| C1 support | {c1['aggregate']['commit_ct_correct']}/76 | {c1['aggregate']['existing_precision']:.4f} | {c1['aggregate']['negative_false_merge_rate']:.4f} | {c1['aggregate']['category_coverage']} | {c1['aggregate']['video_coverage']} |

These replays are diagnostic OCD evidence, not a sealed score and not a
checkpoint-selection signal. Known micro/macro thresholds are marked not
applicable because this route intentionally evaluates novel events only; they
are therefore not silently reported as zero.

## 7. Gate decisions

| gate | result | evidence |
|---|---|---|
| Contract checks | PASS | joint action/causal/mask/gradient checks passed |
| C0 formal | FAIL | 19/76, precision .1450, false merge .1711 |
| repair1 formal | FAIL | 17/76, precision .1250, false merge .1447; recall/precision regressed |
| C1 formal | FAIL | 18/76, precision .1731, false merge .0789; known gate conditions unavailable |
| unchanged controller compatibility | NOT RUN | C1 Gate R/formal gate failed |
| public / sealed | NOT RUN | sealed boundary preserved |

The final formal decision is `C1_GATE_FAIL_STOP_PHASE87_CONTROLLER_ROUTES`.
The improvement in negative false merge under C1 did not solve the dominant
precision/unresolved trade-off. C1 therefore cannot legally be connected to
the frozen Phase19R controller, and no threshold, StateMemory, or controller
lottery is authorized under this phase.

## 8. Method audit and limitations

The read-only recent-method audit covers OVTR (`500e72c`, MIT), TRACT,
COVTrack, ObjectRelator (Apache-2.0), C3Po, AGE, and TALON with their official
URLs, revisions, and compatibility decisions in
`docs/iclr27_phase87/PHASE87_METHOD_AUDIT.md` and
`outputs/iclr27_phase87/audit/methods.json`. No external code or weights were
downloaded. None provides a verified text-free causal prior-video support set
with this 768-D transactional interface.

The main limitations are severe fold imbalance (fold0 has only 51 training
events), overfitting to the positive/negative action labels, unresolved
support quality, and a retrieval/controller interface whose precision gate is
not met. Training loss decreases sharply in some folds, but the replay safety
metrics show why loss cannot be treated as task success.

## 9. Reproduction and artifact ledger

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase87/contract_checks.py
/home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase87/run_autonomous.py
scripts/iclr27_phase87/run_c0_four_fold_supervisor.sh
scripts/iclr27_phase87/run_replay_supervisor.sh
scripts/iclr27_phase87/run_repair1_four_fold_supervisor.sh
scripts/iclr27_phase87/run_repair1_replay_supervisor.sh
scripts/iclr27_phase87/run_c1_four_fold_supervisor.sh
scripts/iclr27_phase87/run_c1_replay_supervisor.sh
```

Key machine-readable artifacts:

- `outputs/iclr27_phase87/audit/window_registration.json`
- `outputs/iclr27_phase87/audit/phase86_rf_causal_correction.json`
- `outputs/iclr27_phase87/audit/phase86_u2_evalmode_correction.json`
- `outputs/iclr27_phase87/audit/causal_event_build.json`
- `outputs/iclr27_phase87/audit/contract_checks.json`
- `outputs/iclr27_phase87/audit/phase87_c0_decision.json`
- `outputs/iclr27_phase87/audit/phase87_repair1_decision.json`
- `outputs/iclr27_phase87/audit/phase87_c1_decision.json`
- `outputs/iclr27_phase87/metrics/c0_aggregate.json`, `repair1_aggregate.json`, `c1_aggregate.json`
- checkpoints under `/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/`

The project output symlink target is `/data2/usr_for_deadline/trackocd_phase87/project_outputs`; no feature or checkpoint copy was made. At report time all
Phase87 workers and supervisors had exited, JSON artifacts parsed, and no
public/Q1/sealed artifact was created. Code was pushed to
`https://github.com/LYQ1107/TrackOCD` before report generation.

## 10. Next research action

Do not repeat C0/repair1/C1 weighting or threshold variants. The evidence
supports a new supervision/task-contract investigation: materialize balanced,
cross-fold causal support with a verifiable action utility target, or revisit
the physical-track/support observability interface. Any future route must be
registered separately and demonstrate safety/precision on TRAIN validation
before another controller compatibility attempt.
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    tmp = DOC.with_name(f".{DOC.name}.tmp"); tmp.write_text(report); os.replace(tmp, DOC)
    decision = {"schema_version": "trackocd.phase87.final_decision.v1", "phase": 87, "status": "PHASE87_REGISTERED_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL", "decision_code": "P87_CONTROLLER_ROUTES_EXHAUSTED_FORMAL_GATE_FAIL_NO_SEALED", "c0": c0, "repair1": r1, "c1": c1, "contract_checks": contracts, "rf_correction": rf, "u2_evalmode_correction": u2, "public_dev_q1_sealed_accessed": False, "controller_compatibility_run": False, "sealed_run": False, "code_head": head, "report": str(DOC.resolve()), "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    for path, payload in [(OUT / "audit" / "final_decision.json", decision), (OUT / "audit" / "report_provenance.json", {"phase": 87, "head": head, "report_sha256": sha(DOC), "method_audit": "docs/iclr27_phase87/PHASE87_METHOD_AUDIT.md", "outputs_root": str(OUT.resolve()), "public_dev_q1_sealed_accessed": False}), (OUT / "audit" / "status_ledger.json", {"phase": 87, "status": decision["status"], "routes": ["C0", "repair1", "C1"], "all_routes_complete": True, "next_action": "WAIT_FOR_NEW_AUTHORIZED_SUPPORT_CONTRACT", "sealed": False})]:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)
    print(str(DOC.resolve()))


if __name__ == "__main__":
    main()
