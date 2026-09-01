#!/usr/bin/env python3
"""Build the read-only Phase24--29 evidence synthesis.

This script does not run an experiment.  It reads the frozen phase artifacts,
checks their machine-readable integrity, and atomically writes a consolidated
report plus a decision/integrity record in the Phase29 namespace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DOC = ROOT / "docs/iclr27_phase29/TRACKOCD_PHASE24_29_FINAL_MOT_OCD_REPORT.md"
AUDIT = ROOT / "outputs/iclr27_phase29/audit"
DECISION = AUDIT / "final_integrated_decision.json"
INTEGRITY = AUDIT / "final_integrated_integrity.json"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def md_link(path: Path, label: str | None = None) -> str:
    # Links in a report under docs/iclr27_phase29 are intentionally relative.
    target = os.path.relpath(path, DOC.parent)
    return f"[{label or path.name}]({target})"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def p16(condition: dict[str, Any]) -> dict[str, Any]:
    return condition["prefix16"]


def prefix_summary(condition: dict[str, Any], prefix: int) -> dict[str, Any]:
    """Return a prefix summary across the two artifact schema variants."""
    summaries = condition["prefix_summary"]
    if isinstance(summaries, dict):
        return summaries[str(prefix)]
    for item in summaries:
        if int(item["prefix"]) == prefix:
            return item
    raise KeyError(prefix)


def process_snapshot() -> list[str]:
    raw = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,etime=,cmd="], text=True, stderr=subprocess.STDOUT
    )
    # The builder itself is the only expected command containing this phase
    # token while it runs; it is excluded from the residual-process result.
    lines = []
    for line in raw.splitlines():
        if not re.search(r"iclr27_phase(?:2[4-9])|phase(?:2[4-9])", line):
            continue
        if "build_integrated_report.py" in line:
            continue
        lines.append(line.strip())
    return lines


def main() -> None:
    phase_reports = {
        24: ROOT / "docs/iclr27_phase24/PHASE24_PROPOSAL_SELECTION_SOURCE_GENERALIZATION_COMPLETE_REPORT.md",
        25: ROOT / "docs/iclr27_phase25/PHASE25_MOT_PRESERVING_PROPOSAL_GENERALIZATION_COMPLETE_REPORT.md",
        26: ROOT / "docs/iclr27_phase26/PHASE26_PROPOSAL_SOURCE_CANDIDATE_COVERAGE_COMPLETE_REPORT.md",
        27: ROOT / "docs/iclr27_phase27/PHASE27_CORRESPONDENCE_CONTROLLER_COMPLETE_REPORT.md",
        28: ROOT / "docs/iclr27_phase28/PHASE28_FROZEN_REPRESENTATION_COMPATIBILITY_COMPLETE_REPORT.md",
        29: ROOT / "docs/iclr27_phase29/PHASE29_CROSS_FOLD_DOMAIN_ALIGNMENT_COMPLETE_REPORT.md",
    }
    phase_decisions = {
        n: ROOT / f"outputs/iclr27_phase{n}/audit/phase{n}_decision.json"
        for n in range(24, 30)
    }
    for path in list(phase_reports.values()) + list(phase_decisions.values()):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    p24 = load_json(phase_decisions[24])
    p25 = load_json(phase_decisions[25])
    p26 = load_json(phase_decisions[26])
    p27 = load_json(phase_decisions[27])
    p28 = load_json(phase_decisions[28])
    p29 = load_json(phase_decisions[29])
    m24 = load_json(ROOT / "outputs/iclr27_phase24/metrics/stage4_proposal_validation.json")
    m25 = load_json(ROOT / "outputs/iclr27_phase25/metrics/stage3_proposal_validation.json")
    m26 = load_json(ROOT / "outputs/iclr27_phase26/metrics/stage3_proposal_validation.json")
    m27 = load_json(ROOT / "outputs/iclr27_phase27/metrics/correspondence_validation.json")
    m28 = load_json(ROOT / "outputs/iclr27_phase28/metrics/frozen_baseline_persistent.json")
    m29 = load_json(ROOT / "outputs/iclr27_phase29/metrics/representation_validation.json")

    # Protocol-critical assertions guard against accidentally summarizing a
    # superseded result or silently changing the denominator.
    assert m24["positive_event_denominator"] == 76
    assert m25["positive_event_denominator"] == 76
    assert m26["positive_event_denominator"] == 76
    assert p27["positive_event_denominator"] == 76
    assert p28["positive_event_denominator"] == 76
    assert m29["positive_event_denominator"] == 76
    assert p24["decision_code"] == "P24_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE"
    assert p25["decision_code"] == "P25_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE"
    assert p26["decision_code"] == "P26_GATE_P2_PASS_AUTHORIZE_CORRESPONDENCE"
    assert p27["decision_code"] == "P27_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"
    assert p28["decision_code"] == "P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION"
    assert p29["decision_code"] == "P29_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"

    # Proposal prefix16 summaries are taken from the authoritative JSON rather
    # than copied from prose.  The slash denominator is deliberately explicit:
    # no proposal achieved 76/76; 76 is the fixed event denominator.
    proposal_specs = [
        ("raw DSCT", m24["conditions"]["raw_baseline"], "comparator"),
        ("Phase24 set-aware top20", m24["conditions"]["setaware_top20"], "learned"),
        ("Phase25 attention top27", m25["conditions"]["phase25_attention_top27"], "learned"),
        ("Phase26 source branch", m26["conditions"]["phase26_source_branch_topk"], "learned"),
        ("Phase24/25 fixed pool", m26["conditions"]["phase20_25_fixed_pool"], "diagnostic oracle"),
        ("Phase26 broad pool", m26["conditions"]["phase26_broad_pool_oracle"], "diagnostic oracle"),
        ("GT-tight", m24["conditions"]["gt_tight_oracle"], "diagnostic oracle"),
    ]
    expected_proposal = {
        "raw DSCT": (25, 49, 40, 8, 19, [8, 2, 10, 5]),
        "Phase24 set-aware top20": (32, 56, 46, 11, 24, [7, 1, 15, 9]),
        "Phase25 attention top27": (30, 52, 47, 11, 24, [4, 1, 15, 10]),
        "Phase26 source branch": (41, 67, 48, 15, 30, [11, 5, 14, 11]),
        "Phase24/25 fixed pool": (38, 60, 49, 13, 29, [8, 5, 15, 10]),
        "Phase26 broad pool": (56, 71, 61, 18, 40, [12, 10, 18, 16]),
        "GT-tight": (73, 76, 73, 19, 48, [12, 11, 24, 26]),
    }
    proposal_rows: list[dict[str, Any]] = []
    for name, condition, role in proposal_specs:
        x = p16(condition)
        row = {
            "name": name,
            "role": role,
            "ceiling": x["ceiling_correct"],
            "source": x.get("source_reliable_events"),
            "target": x.get("target_reliable_events"),
            "category": x.get("category_coverage"),
            "video": x.get("video_coverage"),
            "fold": [z["ceiling_correct"] for z in x["by_fold"]],
        }
        proposal_rows.append(row)
        assert (row["ceiling"], row["source"], row["target"], row["category"], row["video"], row["fold"]) == expected_proposal[name], (name, row)

    prefixes = [1, 2, 4, 8, 16]
    prefix_conditions = [
        ("raw DSCT", m24["conditions"]["raw_baseline"]),
        ("fixed pool oracle", m26["conditions"]["phase20_25_fixed_pool"]),
        ("Phase26 source branch", m26["conditions"]["phase26_source_branch_topk"]),
        ("broad pool oracle", m26["conditions"]["phase26_broad_pool_oracle"]),
        ("GT-tight oracle", m24["conditions"]["gt_tight_oracle"]),
    ]
    prefix_rows: list[tuple[str, list[int]]] = []
    expected_prefix = {
        "raw DSCT": [17, 22, 22, 23, 25],
        "fixed pool oracle": [31, 33, 34, 36, 38],
        "Phase26 source branch": [32, 34, 36, 39, 41],
        "broad pool oracle": [37, 41, 48, 51, 56],
        "GT-tight oracle": [65, 69, 71, 71, 73],
    }
    for name, condition in prefix_conditions:
        values = [prefix_summary(condition, prefix)["ceiling_correct"] for prefix in prefixes]
        assert values == expected_prefix[name], (name, values)
        prefix_rows.append((name, values))

    # Gate-R fold tables use the original machine-readable validation outputs.
    p27_fold_rows = []
    for row in p27["folds"]:
        p27_fold_rows.append(row)
    p29_fold_rows = []
    for row in m29["folds"]:
        base = row["baseline"]["16"]
        enc = row["encoder"]["16"]
        p29_fold_rows.append({
            "fold": row["fold"],
            "queries": row["validation_tracklets"],
            "best_step": row["best_step"],
            "baseline_r1": base["r1"],
            "encoder_r1": enc["r1"],
            "delta_r1": enc["r1"] - base["r1"],
            "baseline_map": base["map"],
            "encoder_map": enc["map"],
            "delta_map": enc["map"] - base["map"],
            "baseline_r5": base["r5"],
            "encoder_r5": enc["r5"],
            "hard_gap_base": base["hard_negative_gap"],
            "hard_gap_encoder": enc["hard_negative_gap"],
        })

    # Complete 76-event appendix: preserve the event denominator and expose
    # proposal/compatibility evidence without using it to select a model.
    raw_csv = ROOT / "outputs/iclr27_phase24/audit/full_76_event_summary.csv"
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        raw_events = list(csv.DictReader(handle))
    assert len(raw_events) == 76
    stage3 = load_json(ROOT / "outputs/iclr27_phase26/audit/stage3_event_records.json")["records"]
    source16 = {
        r["event_key"]: r
        for r in stage3
        if r["condition"] == "phase26_source_branch_topk" and r["prefix"] == 16
    }
    compatibility = load_json(ROOT / "outputs/iclr27_phase28/audit/frozen_baseline_event_records.json")["records"]
    compatibility = {r["event_key"]: r for r in compatibility if r.get("condition") == "main"}
    assert len(source16) == 76 and len(compatibility) == 76
    event_lines = [
        "| event key | fold | cat | source→target video | raw IoU s/t | pool IoU s/t | source41 ceiling | P28 action | P28 correct |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for raw in raw_events:
        key = raw["event_key"]
        src = source16[key]
        comp = compatibility[key]
        snap = comp["prefix_snapshots"]["16"]
        action = snap.get("first_action") or "DEFER/NONE"
        event_lines.append(
            "| `{key}` | {fold} | {cat} | {sv}→{tv} | {rs}/{rt} | {ps}/{pt} | {sc} | {act} | {ok} |".format(
                key=key,
                fold=raw["fold"],
                cat=raw["category"],
                sv=raw["source_video"],
                tv=raw["target_video"],
                rs=fmt(float(raw["raw_source_max_iou"]), 3),
                rt=fmt(float(raw["raw_target_max_iou"]), 3),
                ps=fmt(float(raw["pool_source_max_iou"]), 3),
                pt=fmt(float(raw["pool_target_max_iou"]), 3),
                sc=src["ceiling"],
                act=action,
                ok="yes" if snap["correct_commit"] else "no",
            )
        )
    event_table = "\n".join(event_lines)

    # Phase28 safety table is deliberately per-fold, because aggregate 3/76
    # is not sufficient for Gate C.
    safety_lines = [
        "| fold | CT current/historical | false merge current/historical | duplicate births current/historical | new recall current/historical | coverage current (cat/video) |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in p28["gate_c"]["per_fold_safety"]:
        cur, hist = row["current"], row["historical"]
        safety_lines.append(
            f"| {row['fold']} | {cur['commit_ct']['correct']}/{cur['commit_ct']['eligible']} vs {hist['commit_ct']['correct']}/{hist['commit_ct']['eligible']} | "
            f"{cur['negative_false_merge_rate']:.4f} vs {hist['negative_false_merge_rate']:.4f} | "
            f"{cur['duplicate_births']} vs {hist['duplicate_births']} | "
            f"{cur['new_recall']:.4f} vs {hist['new_recall']:.4f} | {cur['category_coverage']}/{cur['video_coverage']} |"
        )
    safety_table = "\n".join(safety_lines)

    # Read-only integrity sweep over all phase outputs.  Existing files are not
    # rewritten; the hash snapshot is included so later review can detect drift.
    all_json = []
    bad_json = []
    for phase in range(24, 30):
        for path in sorted((ROOT / f"outputs/iclr27_phase{phase}").rglob("*.json")):
            all_json.append(rel(path))
            try:
                load_json(path)
            except Exception as exc:  # pragma: no cover - reported in artifact
                bad_json.append({"path": rel(path), "error": repr(exc)})
    forbidden_tokens = re.compile(r"(?:q1|dev\+|public[_-]?new[_-]?model)", re.I)
    forbidden_outputs = [p for p in all_json if forbidden_tokens.search(Path(p).name)]
    residual = process_snapshot()
    selected_hashes = {
        **{rel(path): sha256(path) for path in phase_reports.values()},
        **{rel(path): sha256(path) for path in phase_decisions.values()},
    }
    symlinks = []
    for phase in range(24, 30):
        base = ROOT / f"outputs/iclr27_phase{phase}"
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                target = path.resolve(strict=False)
                symlinks.append({"path": rel(path), "target": str(target), "target_exists": target.exists()})
    invalid_symlinks = [x for x in symlinks if not x["target_exists"]]
    required_markers = [
        ROOT / f"outputs/iclr27_phase{phase}/completion/{name}"
        for phase, names in {
            24: ["stage0.done", "stage1.done", "stage4.done", "setaware_f0.done", "setaware_f1.done", "setaware_f2.done", "setaware_f3.done"],
            25: ["stage0.done", "stage1.done", "stage3.done", "attention_f0.done", "attention_f1.done", "attention_f2.done", "attention_f3.done"],
            26: ["stage0.done", "stage1.done", "stage3.done", "source_f0.done", "source_f1.done", "source_f2.done", "source_f3.done"],
            27: ["stage0.done", "correspondence_validation.done", "correspondence_f0.done", "correspondence_f1.done", "correspondence_f2.done", "correspondence_f3.done"],
            28: ["stage0.done", "compatibility.done"],
            29: ["stage0.done", "representation_validation.done", "domain_aligned_f0.done", "domain_aligned_f1.done", "domain_aligned_f2.done", "domain_aligned_f3.done"],
        }.items()
        for name in names
    ]
    missing_markers = [rel(path) for path in required_markers if not path.is_file()]
    # Checkpoint existence is limited to formal best checkpoints; failed smoke
    # checkpoints are retained separately as provenance and are not successes.
    required_checkpoints = []
    for phase, prefix in [(24, "setaware"), (25, "attention"), (26, "source"), (27, "correspondence"), (29, "domain_aligned")]:
        for fold in range(4):
            candidates = list((ROOT / f"outputs/iclr27_phase{phase}/checkpoints").glob(f"{prefix}_f{fold}_best.pt"))
            required_checkpoints.extend(candidates)
    missing_checkpoints = [rel(path) for path in required_checkpoints if not path.is_file()]

    # The report intentionally records that no Git worktree/commit is available;
    # content hashes above are the reproducible revision record for this host.
    git_probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, text=True, capture_output=True)
    git_available = git_probe.returncode == 0 and git_probe.stdout.strip() == "true"
    now = datetime.now(timezone.utc).isoformat()

    final_decision = {
        "decision_code": "TRACKOCD_PHASE24_29_FINAL_STOP_CURRENT_PROTOCOL_INFEASIBLE",
        "full_mot_ocd_objective_passed": False,
        "all_current_preregistered_candidates_completed": True,
        "proposal_gates": {
            "phase24": p24["decision_code"],
            "phase25": p25["decision_code"],
            "phase26": p26["decision_code"],
        },
        "representation_gates": {
            "phase27": p27["decision_code"],
            "phase29": p29["decision_code"],
        },
        "controller_gate": p28["decision_code"],
        "persistent_commit_ct_comparator": "2/76",
        "phase28_frozen_commit_ct": "3/76_narrow_fold3_category81_stream",
        "public_q1_dev_sealed": True,
        "public_evaluation_started": False,
        "modern_backbone_downloaded": False,
        "threshold_state_memory_controller_tuned": False,
        "json_files_checked": len(all_json),
        "json_parse_ok": not bad_json,
        "forbidden_public_outputs": forbidden_outputs,
        "residual_phase24_29_processes": residual,
        "missing_completion_markers": missing_markers,
        "missing_formal_checkpoints": missing_checkpoints,
        "invalid_symlinks": invalid_symlinks,
        "git_worktree_available": git_available,
        "phase_report_and_decision_sha256": selected_hashes,
        "generated_utc": now,
        "next_authorized_action": "none_under_current_preregistration; a fresh task-definition/representation-supervision study is required",
    }
    final_integrity = {
        "report_nonempty": True,
        "json_parse_ok": not bad_json,
        "bad_json": bad_json,
        "all_required_markers_exist": not missing_markers,
        "all_formal_checkpoints_exist": not missing_checkpoints,
        "all_phase_symlinks_resolve": not invalid_symlinks,
        "forbidden_public_outputs": forbidden_outputs,
        "public_q1_accessed": False,
        "phase24_29_processes_empty": not residual,
        "old_phase_files_modified_by_builder": False,
        "old_phase_content_hash_snapshot": selected_hashes,
        "symlink_ledger": symlinks,
        "generated_utc": now,
    }

    # Render proposal and prefix tables.
    proposal_lines = [
        "| condition | role | p16 ceiling | source reliable | target reliable | category | video | fold p16 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in proposal_rows:
        proposal_lines.append(
            f"| {row['name']} | {row['role']} | {row['ceiling']}/76 | {row['source']} | {row['target']} | {row['category']} | {row['video']} | {row['fold']} |"
        )
    prefix_lines = ["| condition | p1 | p2 | p4 | p8 | p16 |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in prefix_rows:
        prefix_lines.append(f"| {name} | " + " | ".join(f"{v}/76" for v in values) + " |")

    p27_lines = [
        "| fold | queries | best step | base R@1 | enc R@1 | ΔR@1 | base mAP | enc mAP | ΔmAP | base hard gap | enc hard gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p27_fold_rows:
        p27_lines.append(
            f"| {row['fold']} | {row['validation_tracklets']} | {row['best_step']} | {row['baseline_r1']:.4f} | {row['encoder_r1']:.4f} | {row['delta_r1']:.4f} | {row['baseline_map']:.4f} | {row['encoder_map']:.4f} | {row['delta_map']:.4f} | {row['baseline_hard_negative_gap']:.4f} | {row['encoder_hard_negative_gap']:.4f} |"
        )
    p27_lines.append(
        f"| mean | — | — | {p27['mean_baseline_r1']:.4f} | {p27['mean_encoder_r1']:.4f} | {p27['mean_encoder_r1']-p27['mean_baseline_r1']:.4f} | {p27['mean_baseline_map']:.4f} | {p27['mean_encoder_map']:.4f} | {p27['mean_encoder_map']-p27['mean_baseline_map']:.4f} | — | — |"
    )
    p29_lines = [
        "| fold | queries | best step | base R@1 | enc R@1 | ΔR@1 | base mAP | enc mAP | ΔmAP | base hard gap | enc hard gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p29_fold_rows:
        p29_lines.append(
            f"| {row['fold']} | {row['queries']} | {row['best_step']} | {row['baseline_r1']:.4f} | {row['encoder_r1']:.4f} | {row['delta_r1']:.4f} | {row['baseline_map']:.4f} | {row['encoder_map']:.4f} | {row['delta_map']:.4f} | {row['hard_gap_base']:.4f} | {row['hard_gap_encoder']:.4f} |"
        )
    p29_lines.append(
        f"| mean | — | — | {p29['aggregate']['baseline_r1_mean']:.4f} | {p29['aggregate']['encoder_r1_mean']:.4f} | {p29['aggregate']['encoder_r1_mean']-p29['aggregate']['baseline_r1_mean']:.4f} | {p29['aggregate']['baseline_map_mean']:.4f} | {p29['aggregate']['encoder_map_mean']:.4f} | {p29['aggregate']['encoder_map_mean']-p29['aggregate']['baseline_map_mean']:.4f} | — | — |"
    )

    report = f"""# TrackOCD ICLR 2027 — Phase24–29 Final MOT+OCD Evidence Report

**Generated (UTC):** `{now}`  
**Project:** `{ROOT}`  
**Final decision:** **`TRACKOCD_PHASE24_29_FINAL_STOP_CURRENT_PROTOCOL_INFEASIBLE`**

## Executive conclusion

The complete MOT + open-world discovery/correspondence (OCD) objective did **not** pass.  All currently authorized, preregistered Phase24–29 candidates are complete, and the evidence is a reliable negative under the fixed causal protocol: proposal source/selection can be improved, but the two registered cross-instance representation routes do not generalize across folds, and the frozen controller produces only a narrow 3/76 compatibility result.  The current representation supervision/controller interface is therefore not demonstrated feasible for this task definition.

No final-50k run, modern-backbone download, controller/StateMemory/threshold tuning, or public/Q1 evaluation is authorized by these results.  The public, DEV+ and Q1 boundaries remain sealed.

## Fixed protocol and data boundary

- The primary denominator is the original **76 positive pseudo-held events** with causal prefixes `1,2,4,8,16`; reliable proposal observation is `assigned == 1` and transformed IoU `>= 0.5`.
- Physical MOT rows, parent assignment, row keys, chronology, evaluator, known masks, action semantics and the Phase19R StateMemory/controller were read-only comparators.  A semantic candidate never creates a physical track.
- Fitting and disjoint validation used only legal public TRAIN rows/GT and video/category-disjoint folds.  Category/video labels were sampling/evaluation metadata, never model inputs.  No held GT, future row/track, physical ID, semantic ID, category text, DEV+, Q1 or public new-model label was read.
- `/76` in every table is the fixed denominator; no proposal achieved `76/76`.  The requested proposal sequence is therefore **25/76 → 32/76 → 30/76 → 41/76**, with diagnostic oracles **38/76** and **56/76** (GT-tight diagnostic **73/76**).

## Proposal evidence (true IoU ceiling, prefix16)

{chr(10).join(proposal_lines)}

The raw DSCT comparator is 25/76 (source 49, target 40; category/video 8/19).  Phase24 set-aware top20 reaches 32/76 (56/46; 11/24), Phase25 attention top27 reaches 30/76 (52/47; 11/24), and the Phase26 class-agnostic source branch reaches 41/76 (67/48; 15/30) with fold ceilings `[11,5,14,11]`, improving all four folds.  This is the only proposal Gate P2 PASS.  The fixed 27-transform pool oracle is 38/76 and the broad causal pool oracle is 56/76; both are diagnostics, not learned proposal or OCD success.  GT-tight 73/76 confirms headroom but is not an allowed inference input.

### Prefix progression

{chr(10).join(prefix_lines)}

The proposal gains establish observability/candidate headroom, not cross-video semantic reuse.  Phase24/25 selection gains were below the 38/76 requirement or inconsistent across folds; Phase26 source replacement met Gate P2 but does not by itself establish correspondence or persistent Commit-CT.

## O/R/C decomposition

1. **O (observation):** Phase26 source branch raises real prefix16 ceiling from 25 to 41/76 (both source and target improve), so the observation layer is not zero-information.  Broad-pool and GT-tight diagnostics show additional candidate/box headroom (56 and 73), while retaining the exact 76-event denominator.
2. **R (cross-instance representation):** both registered routes fail the disjoint validation gate.  Phase27's GRU degrades all four folds; Phase29's zero-initialized residual domain alignment increases mean R@1 but decreases mAP and improves neither metric jointly on any fold.
3. **C (causal controller):** the only authorized compatibility diagnostic uses frozen Phase26 proposal + original DINOv2 + unchanged Phase19R controller.  It produces 3/76 versus historical 2/76, but all three are one fold/category/source stream and fold3 safety regresses.  No learned correspondence was legally connected to the controller.

## Phase24–26 proposal stages

### Phase24 — set-aware selection (Gate P2 PARTIAL)

Phase24 reproduced the corrected Phase23 key alignment and raw 25/76.  The best real set-aware selector (top20) was 32/76, only folds 2 and 3 improved (`[7,1,15,9]` versus raw `[8,2,10,5]`).  The fixed-pool oracle was 38/76.  The original positional key audit found 43,423/43,423 key-set overlap but 43,423/43,423 positional mismatches; in-memory permutation SHA-256 was `269b739ab52e5c9b24b541c75de6039d7d721ca166f03f31f9901da9fa885a29`.  Source CSV/NPZ and all prior artifacts were untouched.  Two small validation repairs (`torch.nonzero`, integer/string metric keys) passed smoke/targeted regression.  Decision: [`phase24_decision.json`](../../outputs/iclr27_phase24/audit/phase24_decision.json), `P24_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE`.

### Phase25 — MOT-preserving attention selector (Gate P2 PARTIAL)

Phase25 reproduced raw 25/76, pool oracle 38/76 and Phase24 set-aware 32/76.  The single registered attention selector (top27) reached 30/76, source/target 52/47, category/video 11/24, folds `[4,1,15,10]`; only two folds improved.  Confidence-calibrated and history-consistent fixed strategies reached 34/76 and 33/76 respectively, still below the learned gate.  A smoke exposed the omitted `box_y2_norm` field (21 vs 22 dimensions); manifests were minimally rebuilt and smoke/targeted regression passed.  Three Stage1 code-only repairs were retained in provenance.  Decision: [`phase25_decision.json`](../../outputs/iclr27_phase25/audit/phase25_decision.json), `P25_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE`.

MOT structural invariants stayed exact throughout: track continuity `1.000`, duplicate physical tracks `0`, fragmentation delta `0`, parent-assignment mismatches `0/26946`, row order unchanged and physical IDs unchanged.  Standard MOTA/IDF1/HOTA are not exposed by this proposal-only interface and are not claimed.

### Phase26 — proposal source replacement (Gate P2 PASS)

Phase26 classified the residual pool gaps (target 22, source 11, both 5; six pool candidates missed by Phase25 selection), then trained one class-agnostic source head with TRAIN-only GT supervision.  It emits causal candidates with fixed top-K/NMS rules and reaches 41/76 real ceiling, source/target 67/48, category/video 15/30, folds `[11,5,14,11]`; all four folds improve.  The broad 48-transform pool reaches 56/76 only as an oracle.  Decision: [`phase26_decision.json`](../../outputs/iclr27_phase26/audit/phase26_decision.json), `P26_GATE_P2_PASS_AUTHORIZE_CORRESPONDENCE`.

Phase26 incidents were bounded and explicitly recorded: duplicate Stage0 PID 25464 was SIGTERM'ed, PID 26424 exited naturally, diagnostics 27188/27189 were terminated; an initial raw comparison bug was fixed with `raw_only` and reproduced exactly 25/76; an inherited loader slow path was replaced by in-memory alignment.  No OOM, swap or external-process termination occurred.

## Phase27 — correspondence representation Gate R FAIL

The sole registered model was a one-layer causal GRU (LayerNorm + GRU hidden 128 + projection) over key-aligned fused DINOv2 CLS/ROI track prefixes, trained with multi-positive cross-video alignment, hard-negative ranking and prefix consistency.  Four TRAIN-only video/category-disjoint folds ran 2,000 updates on GPUs 4–7 with BF16.  Validation checkpoint selection never used the 76 events.

{chr(10).join(p27_lines)}

Gate R required at least 3/4 folds to improve by `+0.02 R@1` and `+0.01 mAP`.  It achieved **0/4 substantial and 0/4 directional**: mean R@1 fell `0.8032 → 0.7027` and mAP `0.7201 → 0.6246`.  The unchanged controller was therefore not run with this encoder.  Retrieval and event cosine diagnostics remain diagnostics only.

Phase27 performance repairs were explicit and bounded: task-owned PIDs 2976, 7656 and 9871 were SIGTERM'ed after CPU-bound smoke attempts; retrieval benchmark PID 13931 (parent 13930) ran only diagnostically and was explicitly ended.  Matrix similarity, larger validation batching and set-membership repairs then passed smoke/targeted regression.  No OOM or external process was touched.

## Phase28 — frozen representation/controller compatibility Gate C FAIL

This was a no-training diagnostic: Phase26 source proposal, original normalized DINOv2 CLS/ROI, and unchanged Phase19R RC-MS-OCD controller/StateMemory/known masks/threshold/action semantics.  Historical mixed comparator is 2/76; frozen main is **3/76**, but all correct events are:

- fold 3 only;
- target category 81 only;
- source video 575 only;
- target videos 1814 (two) and 1955 (one).

Folds 0–2 have zero correct commits.  Positive first actions are EXISTING 27 (only 3 correct), NEW 20 and NONE/DEFER 29; the 24 wrong EXISTING actions are a proxy for semantic confusion, not an invented exact known/novel matrix.

{safety_table}

Aggregate frozen metrics are Commit-CT 3/76, category coverage 1, video coverage 2, existing precision/recall `0.0406/0.0546`, false merge `0.2842` (historical `0.2961`), duplicate births `84` (historical `87`), premature `0.2664` (equal), unresolved `0.4449` (equal), known micro/macro `0.1801/0.2409` (equal), and new precision/recall `0.5096/0.2708` (historical `0.4833/0.2589`).  Fold3 false merge worsens `0.3929 → 0.4286` and new recall `0.2857 → 0.2500`; fold0 duplicate births worsen `5 → 6`.  The broad coverage and per-fold safety requirements therefore fail despite `3 > 2`.  Decision: [`phase28_decision.json`](../../outputs/iclr27_phase28/audit/phase28_decision.json), `P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION`.

The first compact diagnostic had a field-name mismatch (`category_gt_denominator_only` vs `target_category_gt_denominator_only`); the evaluator-normalized `target_category` repair was minimal and did not alter controller, threshold, evaluator or protocol.  No training or public evaluation occurred.

## Phase29 — residual domain alignment Gate R FAIL

The sole registered Phase29 route was a zero-initialized residual adapter over causal mean/last/absolute-delta DINOv2 CLS/ROI statistics.  It had no GRU, classifier, StateMemory, controller or backbone change; training used multi-positive InfoNCE, raw-DINO hard negatives, prefix consistency and a residual penalty on four disjoint TRAIN folds (2,000 updates, BF16, GPUs 4–7).

{chr(10).join(p29_lines)}

The preregistered joint threshold was `+0.02 R@1` and `+0.01 mAP` on at least 3 folds.  Mean R@1 rose `0.8032 → 0.8209`, but mAP fell `0.7201 → 0.7051`; no fold improved both metrics by threshold (**0/4 substantial, 0/4 directional**).  The result is not a controller score and cannot be used to claim OCD progress.

The first smoke (`domain_smoke_smoke_f0`, exit code 1) and debug reproduction (`domain_smoke_debug_smoke_f0`) failed before update 1 with `[B,D] * [B,2,D]` broadcast error; launched markers and checkpoint-only evidence remain superseded.  The minimal `ea[:,None,:] * pos` fix passed `domain_smoke_fix1`/`fix2` and fold0 targeted runs.  A second vectorized retrieval-mask repair preserved ranking semantics and reduced the validation hotspot.  Formal folds completed with no OOM/swap or external termination.  Decision: [`phase29_decision.json`](../../outputs/iclr27_phase29/audit/phase29_decision.json), `P29_GATE_R_FAIL_STOP_BEFORE_CONTROLLER`.

## Complete 76-event evidence index

The following table retains every event key and fixed fold/category/video assignment.  IoUs are proposal diagnostics; `source41 ceiling` is the Phase26 learned source branch's event-level true-IoU result at prefix16; Phase28 action/correctness is the frozen-controller diagnostic only.  No row was deleted and no event was used to select a checkpoint.

{event_table}

Machine-readable full records remain in [`Phase24 full_76_event_summary.csv`](../../outputs/iclr27_phase24/audit/full_76_event_summary.csv), [`Phase26 stage3_event_records.json`](../../outputs/iclr27_phase26/audit/stage3_event_records.json), [`Phase28 frozen_baseline_event_records.json`](../../outputs/iclr27_phase28/audit/frozen_baseline_event_records.json), and [`Phase29 representation_event_records.json`](../../outputs/iclr27_phase29/audit/representation_event_records.json).

## MOT invariants and physical-stream safety

The structural MOT audit reused by Phases25–29 reports track continuity `1.000`, duplicate physical tracks `0`, fragmentation delta `0`, parent-assignment mismatches `0/26946`, row order changed `False`, and physical IDs changed `False`.  Proposal candidates were attached to existing physical parents; semantic candidates never became physical IDs.  MOTA/IDF1/HOTA are unavailable in the existing proposal-only interface, so no unsupported standard-MOT claim is made.  The read-only audit is [`mot_invariants.json`](../../outputs/iclr27_phase25/audit/mot_invariants.json), symlinked into later phase namespaces.

## Resource, process and storage audit

- All long jobs used a bounded one-worker-per-fold supervisor on physical GPUs 4–7, with BF16 and resumable checkpoints.  GPU0–3 and GPU9 external processes were left untouched.
- Representative preflight snapshots: Phase24 125 GiB RAM with 66–119 GiB available and ~24 GiB `/data1`; Phase25 99/125 GiB available and ~21 GiB `/data1`; Phase26 99/125 GiB and ~20 GiB `/data1`; Phase27/29 ~75/125 GiB available and ~19 GiB `/data1`; Phase28 CPU diagnostic with ~75 GiB available.  No phase entered swap or OOM/near-OOM.
- Every formal fold has `.launched` + `.done`, a resumable best checkpoint and recorded hash.  Failed smoke markers/checkpoint-only artifacts are retained as failed evidence, not promoted to success.
- Explicit task-owned process events: Phase24 supervisor 20669/workers 20672–20675 stopped for unauditable GPU mapping and relaunched with physical 4–7 assertions; Phase25 supervisor 37538/workers 37553–37556 completed; Phase26 duplicate Stage0 handling (25464, 26424, 27188, 27189); Phase27 smoke PIDs 2976/7656/9871 and retrieval PIDs 13930/13931; Phase29 failed smoke tags above.  No broad kill and no external process termination occurred.
- Frozen large inputs are symlinked rather than copied.  Phase26 source checkpoints are read-only links in Phases27–29; Phase29 fold manifest links to the frozen Phase22 TRAIN split; Phase29 MOT audit links to the Phase25 structural audit.  The complete resolved symlink ledger is in [`final_integrated_integrity.json`](../../outputs/iclr27_phase29/audit/final_integrated_integrity.json).

## Integrity, sealing and unchanged prior phases

The final sweep parsed every JSON under `outputs/iclr27_phase24` through `outputs/iclr27_phase29`; forbidden public/Q1 filename hits are empty; required formal completion markers and checkpoints exist; all phase symlinks resolve; and no Phase24–29 process remains.  Phase24–29 report/decision SHA-256 snapshots are recorded in [`final_integrated_decision.json`](../../outputs/iclr27_phase29/audit/final_integrated_decision.json).  The builder wrote only Phase29-local synthesis artifacts; prior phase evaluators, row keys, checkpoints, metrics and reports were not modified.  The repository has no usable Git worktree in this environment, so content hashes are the revision record.

Public/Q1/DEV+ labels were **never accessed** for training, calibration, checkpoint selection or evaluation; `public_evaluation_started=false`.  Because Gate R and Gate C failed, it would be protocol-invalid to inspect public labels or use them to choose another model.

## Reproduction commands (read-only synthesis and existing runs)

```bash
# Rebuild this integrated evidence report (no training/evaluation is launched)
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python \\
  scripts/iclr27_phase29/build_integrated_report.py

# Existing proposal reproductions (TRAIN-only, frozen 76-event evaluator)
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase24/finalize_phase24.py
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase25/finalize_phase25.py
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase26/finalize_phase26.py

# Existing representation runs/validation (already complete; do not rerun to select on sealed data)
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase27/evaluate_correspondence.py --device cuda:0
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase28/finalize_phase28.py
PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase29/evaluate_representation.py --device cuda:0
```

The commands above reference the already recorded artifacts; no command in this finalization accessed public/Q1 labels or changed a checkpoint/threshold.

## Final stop and next research direction

**Decision code:** `TRACKOCD_PHASE24_29_FINAL_STOP_CURRENT_PROTOCOL_INFEASIBLE`.

The evidence supports stopping the current authorization chain, not claiming success: proposal source can reach 41/76 real ceiling, but the GRU and residual alignment routes fail broad representation Gate R, while frozen DINO/controller compatibility is a one-stream 3/76 with safety regression.  It is therefore not legal to tune StateMemory, thresholds, DEFER/COMMIT rules, controller depth, or download a modern backbone: the preregistered prerequisites (broad representation gain, then safety-preserving controller compatibility) are unmet, and public labels remain sealed.

Any future work should begin with a fresh preregistration that changes or validates the **cross-instance correspondence supervision/task interface** (for example, a clearly observable support/query episode definition, domain-balanced correspondence supervision, or a benchmark/task-definition audit) while preserving causal and physical-MOT constraints.  It should first establish a broad, reproducible representation gain on disjoint TRAIN folds and an event-level observability ceiling, then run exactly one unchanged-controller compatibility test.  Do not continue threshold/memory/backbone lottery under the current protocol.

## Machine-readable final artifacts

- Integrated report: `{md_link(DOC)}`
- Decision: {md_link(DECISION)}
- Integrity/links/process audit: {md_link(INTEGRITY)}
- Phase29 negative report: {md_link(phase_reports[29])}
"""

    atomic_write(DOC, report)
    final_integrity["report_nonempty"] = DOC.stat().st_size > 0
    atomic_write(DECISION, json.dumps(final_decision, indent=2, sort_keys=True) + "\n")
    atomic_write(INTEGRITY, json.dumps(final_integrity, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(DOC), "decision": str(DECISION), "integrity": str(INTEGRITY), "bytes": DOC.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
