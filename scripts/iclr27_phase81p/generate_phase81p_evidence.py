#!/usr/bin/env python3
"""Render the Phase81P evidence ledger and reports from immutable artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase81p"
START = "2026-09-03T08:40:36Z"
DEADLINE = "2026-09-03T18:40:36Z"
START_HEAD = "75edd12e2ae58a6decde02f860949786af844598"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def route_event(tag: str) -> list[dict]:
    paths = sorted(OUT.glob(f"metrics/replay_{tag}_f*.json"))
    return [load(p) for p in paths]


def route_physical(tag: str) -> list[dict]:
    paths = sorted(OUT.glob(f"metrics/physical_{tag}_f*.json"))
    return [load(p) for p in paths]


def train_summaries(route: str) -> list[dict]:
    return [load(p) for p in sorted(OUT.glob(f"metrics/{route}/fold*/summary.json"))]


ROUTES = [
    {
        "id": "P1_initial",
        "label": "Initial Q0-anchored association",
        "train": "",
        "event": "formal_replay2",
        "physical": "physical_formal1",
        "hypothesis": "learned geometry/score pair association with Hungarian assignment and max_miss=8",
        "repair": "primary registered model; original TRAIN feature route",
    },
    {
        "id": "P2_masked_context",
        "label": "Q0-aligned balanced + masked NEW context",
        "train": "q0_train4t_balanced",
        "event": "q0_train4t_balanced_formal_q0aligned_balanced",
        "physical": "q0_train4t_balanced_q0aligned_physical",
        "hypothesis": "remove fixed-width zero-padding bias in NEW-logit context using effective-candidate mean",
        "repair": "minimal contract repair after P1 physical collapse; Q0 TRAIN stream and balanced positive sampling",
    },
    {
        "id": "P2_resolution_aware",
        "label": "Resolution-aware Q0-aligned association",
        "train": "q0_train4t_resaware",
        "event": "q0_train4t_resaware_formal_q0resaware",
        "physical": "q0_train4t_resaware_q0resaware_physical",
        "hypothesis": "normalize causal geometry with each current frame's actual width/height",
        "repair": "single evidence-based scale-contract repair; raw Q0 boxes/scores unchanged",
    },
    {
        "id": "P3_top4_context",
        "label": "Top-4 candidate-conditioned NEW context",
        "train": "q0_train4t_topctx",
        "event": "q0_train4t_topctx_formal_q0topctx",
        "physical": "q0_train4t_topctx_q0topctx_physical",
        "hypothesis": "avoid unrelated active tracks in NEW context by using the strongest four causal pair candidates",
        "repair": "final registered repair after residual birth/association calibration mismatch",
    },
]


def event_aggregate(items: list[dict]) -> dict:
    return {
        "model_count": len(items),
        "both_reliable": [int(x["aggregate"]["both_reliable"]) for x in items],
        "source_reliable": [int(x["aggregate"]["source_reliable"]) for x in items],
        "target_reliable": [int(x["aggregate"]["target_reliable"]) for x in items],
        "mean_both_reliable": statistics.mean(x["aggregate"]["both_reliable"] for x in items) if items else None,
        "by_model_fold": [x["aggregate"].get("by_fold", {}) for x in items],
    }


def physical_aggregate(items: list[dict]) -> dict:
    keys = [
        "rows", "unique_physical_tracks", "reliable_gt_assignments_iou_ge_0.5",
        "reliable_gt_tracks", "gt_track_switches", "fragmented_gt_tracks",
        "merged_pred_tracks", "duplicate_birth_proxy",
    ]
    return {
        "folds": [{"learned": {k: int(x["learned"][k]) for k in keys}, "q0_native": {k: int(x["q0_native"][k]) for k in keys}, "lifecycle": x["learned"].get("lifecycle_counts", {})} for x in items],
        "means": {k: statistics.mean(x["learned"][k] for x in items) if items else None for k in keys},
        "q0_means": {k: statistics.mean(x["q0_native"][k] for x in items) if items else None for k in keys},
    }


def render_event_table(q0_records: list[dict], event_sets: dict[str, list[dict]]) -> str:
    q0 = {x["event_key"]: x for x in q0_records if x.get("prefix") == 16 and x.get("polarity") == "positive"}
    by_route = {}
    for rid, items in event_sets.items():
        maps = [{x["event_key"]: x for x in d.get("events", [])} for d in items]
        by_route[rid] = maps
    lines = ["| # | event key | fold | Q0 S/T/B | P1 B flags | P2-mask B flags | P2-res B flags | P3-top4 B flags |", "|---:|---|---:|---|---|---|---|---|"]
    keys = sorted(q0, key=lambda k: (int(q0[k].get("fold", 0)), k))
    for i, key in enumerate(keys, 1):
        x = q0[key]
        q0s = f"{int(x.get('source_reliable', False))}/{int(x.get('target_reliable', False))}/{int(x.get('both_reliable', False))}"
        vals = []
        for rid in ["P1_initial", "P2_masked_context", "P2_resolution_aware", "P3_top4_context"]:
            flags = "".join("1" if m.get(key, {}).get("both_reliable", False) else "0" for m in by_route.get(rid, []))
            vals.append(flags or "-")
        lines.append(f"| {i} | `{key}` | {x.get('fold','?')} | {q0s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    return "\n".join(lines)


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    end_head = git(["rev-parse", "HEAD"])
    q0_summary = load(OUT / "audit/stage1_summary.json")
    q0_records = load(ROOT / "outputs/iclr27_phase80c/audit/proposal_quality_event_records.json")
    event_sets = {r["id"]: route_event(r["event"]) for r in ROUTES}
    physical_sets = {r["id"]: route_physical(r["physical"]) for r in ROUTES}
    train_sets = {r["id"]: train_summaries(r["train"]) for r in ROUTES if r["train"]}

    route_results = []
    for r in ROUTES:
        ev = event_aggregate(event_sets[r["id"]])
        ph = physical_aggregate(physical_sets[r["id"]])
        route_results.append({
            **r,
            "event": ev,
            "physical": ph,
            "train": train_sets.get(r["id"], []),
            "physical_gate": {
                "proposal_rows_identical": all(x["learned"]["rows"] == x["q0_native"]["rows"] for x in physical_sets[r["id"]]),
                "id_switch_non_worse": ph["means"]["gt_track_switches"] <= ph["q0_means"]["gt_track_switches"],
                "fragmentation_non_worse": ph["means"]["fragmented_gt_tracks"] <= ph["q0_means"]["fragmented_gt_tracks"],
                "merged_tracks_non_worse": ph["means"]["merged_pred_tracks"] <= ph["q0_means"]["merged_pred_tracks"],
                "duplicate_birth_non_worse": ph["means"]["duplicate_birth_proxy"] <= ph["q0_means"]["duplicate_birth_proxy"],
                "pass": False,
            },
        })

    manifest_paths = [OUT / "manifests/q0_train4t/train_manifest.json", OUT / "manifests/q0_train4t_resaware/train_manifest.json"]
    checkpoint_paths = [p for r in ROUTES if r["train"] for p in sorted(OUT.glob(f"checkpoints/{r['train']}/fold*/best.pt"))]
    artifact_hashes = {}
    for p in [OUT / "audit/q0_physical_callgraph.json", OUT / "audit/assignment_failure_taxonomy.json", OUT / "audit/literature_audit.json", ROOT / "outputs/iclr27_phase80c/audit/proposal_quality_event_records.json", ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json", *manifest_paths, *checkpoint_paths]:
        if p.is_file():
            artifact_hashes[str(p)] = sha(p)

    ledger = {
        "schema_version": "phase81p.validation_evidence_ledger.v1",
        "phase": "Phase81P+",
        "status": "PHYSICAL_ASSOCIATION_MODEL_FAMILY_EXHAUSTED_Q0_SAFETY_FAIL",
        "research_start_utc": START,
        "registered_deadline_utc": DEADLINE,
        "rendered_utc": now.isoformat(),
        "starting_head": START_HEAD,
        "ending_head": end_head,
        "changed_source_commits": ["32c83e4", "95e75ae", "95ebcb8", "a1a9e08", "73e2273", "7d3f188", "e2d1461", "a5b0066", "f3b7225", "1318428", "4726588", "b174a75", "2f86504", "119bd8e", "366e3dd", "363f4e8", "8080f06", "ac9ef9f", "303000b", "064bf58", "997b524", "b4f419d", "d9fe734", "daedede", "ce16a49", "62ff805", "f9a38fb", "f111b11", "2946a92", "832ad95"],
        "compile_checks": {"association.py": "PASS", "build_q0_train_manifest.py": "PASS", "replay_association.py": "PASS", "evaluate_physical_replay.py": "PASS", "supervisor_shells": "PASS"},
        "tests": [{"name": "contract_smoke", "status": "PASS"}, {"name": "association_shape_finite", "status": "PASS"}, {"name": "resolution_aware_shape", "status": "PASS"}, {"name": "top_context_shape_finite", "status": "PASS"}],
        "scientific_commands": [
            "PYTHONPATH=. ovtr-python scripts/iclr27_phase81p/build_q0_train_manifest.py --q0-json outputs/iclr27_phase4t/train_stream/teta/tao_track.json --route-tag q0_train4t_resaware --resolution-aware",
            "scripts/iclr27_phase81p/run_four_fold_q0_aligned_supervisor.sh formal_q0resaware 20 /data2/usr_for_deadline/trackocd_phase81p/data/q0_train4t_resaware q0_train4t_resaware",
            "scripts/iclr27_phase81p/run_replay_route_supervisor.sh q0_train4t_resaware formal_q0resaware outputs/iclr27_phase81p/checkpoints/q0_train4t_resaware 0 8 0 1",
            "scripts/iclr27_phase81p/run_physical_replay_route_supervisor.sh q0_train4t_resaware q0resaware_physical outputs/iclr27_phase81p/checkpoints/q0_train4t_resaware 0 8 0 1",
            "scripts/iclr27_phase81p/run_four_fold_q0_aligned_supervisor.sh formal_q0topctx 20 /data2/usr_for_deadline/trackocd_phase81p/data/q0_train4t_resaware q0_train4t_topctx",
            "scripts/iclr27_phase81p/run_replay_route_supervisor.sh q0_train4t_topctx formal_q0topctx outputs/iclr27_phase81p/checkpoints/q0_train4t_topctx 0 8 0 1",
            "scripts/iclr27_phase81p/run_physical_replay_route_supervisor.sh q0_train4t_topctx q0topctx_physical outputs/iclr27_phase81p/checkpoints/q0_train4t_topctx 0 8 0 1",
        ],
        "q0_hashes": {"checkpoint": "809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738", "event_stream": "112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac", "native_lineage": "d33e60f4603aaa8aa744d8d73553b42153be9f9b88a3a19aa6eb26884d31a2e1", "train_q0_stream": "0a4dd5a1d5f443944df3d043297560d7790247f28330fe60fd571670ea853546"},
        "q0_prefix16": {"source_pool_good": 72, "target_pool_good": 64, "source_reliable": 49, "target_reliable": 40, "both_reliable": 25, "assignment_gap_events": 36},
        "route_results": route_results,
        "artifact_hashes": artifact_hashes,
        "resource_policy": {"gpu_mapping": {"fold0": 4, "fold1": 5, "fold2": 6, "fold3": 7}, "max_workers": 4, "ram_headroom_required": "25%", "large_storage": "/data2/usr_for_deadline/trackocd_phase81p", "oom_events": 0, "external_processes_terminated": 0},
        "sealed_boundary": {"dev_plus_accessed": False, "q1_accessed": False, "public_new_accessed": False, "sealed_accessed": False, "held_labels_before_inference": False},
        "downstream": {"controller": "NOT_RUN_PHYSICAL_GATE_FAIL", "semantic_state": "NOT_RUN_PHYSICAL_GATE_FAIL", "sealed": "NOT_RUN", "reason": "No learned association route preserved Q0 physical safety."},
    }
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "validation_evidence_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    for r in route_results:
        payload = {
            "schema_version": "phase81p.route_decision.v1",
            "phase": "Phase81P+",
            "route": r["id"],
            "status": "FAIL_PHYSICAL_SAFETY",
            "hypothesis": r["hypothesis"],
            "repair": r["repair"],
            "training": r["train"],
            "event_observability": r["event"],
            "physical_validation": r["physical"],
            "gate_checks": r["physical_gate"],
            "next_action": "close association model family; do not run semantic/controller/sealed under this route",
            "sealed_accessed": False,
        }
        name = {"P1_initial": "p1_decision.json", "P2_masked_context": "p2_decision.json", "P2_resolution_aware": "p2_resaware_decision.json", "P3_top4_context": "p3_decision.json"}[r["id"]]
        (audit_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    table = render_event_table(q0_records, event_sets)
    route_lines = []
    for r in route_results:
        ev = r["event"]
        ph = r["physical"]
        route_lines.append(f"| {r['id']} | {ev['both_reliable']} (mean {ev['mean_both_reliable']:.2f}) | {ph['means']['unique_physical_tracks']:.1f} | {ph['means']['gt_track_switches']:.1f} | {ph['means']['fragmented_gt_tracks']:.1f} | {ph['means']['merged_pred_tracks']:.1f} | {ph['means']['duplicate_birth_proxy']:.1f} | FAIL |")
    training_lines = []
    for rid, items in train_sets.items():
        for x in items:
            training_lines.append(f"| {rid} | f{x['fold']} | {x['steps']} | {x['epochs']} | {x['fit_examples']} | {x['val_examples']} | {x['best']['val'].get('balanced_accuracy')} | {x['best']['val'].get('pair_accuracy')} | {x['best']['val'].get('new_rate')} | {x['wall_seconds']:.2f} |")

    report = f"""# TrackOCD Phase81P+ — Causal Physical Association Report

## Status

- **Decision:** `PHYSICAL_ASSOCIATION_MODEL_FAMILY_EXHAUSTED_Q0_SAFETY_FAIL`
- Registered window: `{START}` → `{DEADLINE}`; report rendered `{now.isoformat()}`.
- Starting HEAD: `{START_HEAD}`; ending HEAD: `{end_head}`.
- Q0 physical safety: **FAIL** for every learned association version. No semantic/controller or sealed evaluation was authorized after this gate.
- Frozen public boundary: no DEV+, Q1, public-new or sealed labels; no future rows/tracks; no category/text/semantic/physical ID feature. Physical IDs remain bookkeeping only.

## Frozen Q0 anchor and Phase80C evidence

Q0 OVTR commit `500e72c`, checkpoint SHA-256 `809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738`, and `score_mode=base` proposal stream remain unchanged. Full validation anchor is top20 IoU≥0.5 recall `71062/112798=0.629993`, macro HOTA `0.844035`. The native event stream has 99,476 rows in 91 event videos (26,009 births; 73,467 continuations; 1,026 tracks).

Phase80C positive prefix16 audit: proposal-pool-good source `72/76`, target `64/76`; strict Q0 reliable source `49/76`, target `40/76`, both `25/76`. There are 36 both-pool-good assignment/temporal gaps (29 fragmentation, 7 missed continuation). This is evaluator-only diagnosis, not training data.

## Q0 physical callgraph and contract

`q0_physical_callgraph.json` (SHA `{artifact_hashes.get(str(OUT / 'audit/q0_physical_callgraph.json'), 'n/a')}`) records RGB/frame loader → OVTR detector/query decoder → score-mode base filtering → `RuntimeTrackerBase` association/lifecycle → native lineage capture → event evaluator. The learned route consumes only each frozen Q0 proposal's absolute `xyxy`, base score, frame order and causal visual/geometry fields. It emits exactly one row per Q0 proposal with `physical_track_id`, candidate rank, association score and lifecycle action (`birth`/`continuation`); no proposal row is deleted or created.

## Failure taxonomy (evaluator-only)

The 36 assignment-gap events are fixed before training: `C_FRAGMENTATION=29` and `B_MISSED_CONTINUATION=7` (fold totals: f1=8 C; f2=5 B+9 C; f3=2 B+12 C). Candidate count, max IoU, Q0 score, temporal IoU and transition evidence are retained in `assignment_failure_taxonomy.json`. No event label or physical ID is serialized into TRAIN features.

## Official method audit

The audit in `PHASE81P_LITERATURE_AUDIT.md`/`literature_audit.json` records official heads and licenses. OVTR (`500e72c`, MIT) supplies the frozen query/proposal path; MOTIP-2 (`012856c`, Apache-2.0) supplies history-conditioned association ideas but its numeric identity prompts are excluded; COVTrack (`9b0ced5`, Apache-2.0) supplies confidence/momentum/motion separation but semantic cues are excluded; TRACT (`19f01d7`, Apache-2.0) supplies trajectory aggregation ideas but its TraCLIP/text and external proposal path are excluded; PS-MOT (`163e9ee`, MIT) is an 8-process point-supervised route and is not imported; ObjectRelator (`59f79d5`, Apache-2.0) and C3Po (`21254a0`, MIT) are offline/static correspondence references, not causal MOT. No external weights were downloaded.

## Architecture and TRAIN contract

The registered model is a small Q0-anchored association transformer: 16-D pair features (causal box IoU/centre/scale/time/age/miss/score/confidence plus optional motion or RGB descriptor), LayerNorm→128-D projection, one 4-head TransformerEncoder layer, pair and NEW heads (151,842 parameters), Hungarian one-to-one assignment, dormant horizon `max_miss=8`, and a 256-track causal memory cap. The model never receives track-slot/physical ID numbers. TRAIN targets use same-GT trajectory matching and hard negatives; future GT is used only to form labels, never an inference tensor. The balanced Q0 route uses the legal `outputs/iclr27_phase4t/train_stream/teta/tao_track.json` (SHA `0a4dd5…`) with 43 non-event TRAIN videos, fixed category/video-disjoint folds and 9-candidate shards. The default and resolution-aware manifests are hashed in the ledger.

## Training and repair history

The initial route used the original Q0-aligned geometry contract. Its physical replay exposed a train/runtime NEW-context mismatch, so the first minimal repair masked zero-padded TRAIN candidates (`f111b11`). A second evidence-based repair used each current frame's actual dimensions (119,559/145,429 event rows are 1280×720; other 640×480/1920×1080/1200 sizes also occur; `2946a92`). The final registered repair used top-4 pair candidates for the NEW context in both train and runtime (`832ad95`). All source changes were compiled and pushed to `https://github.com/LYQ1107/TrackOCD` before this report.

### TRAIN-disjoint summaries

| Route | Fold | Updates | Epochs | Fit rows | Val rows | Best balanced acc. | Best pair acc. | Best NEW rate | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(training_lines)}

### Physical replay (diagnostic proxy; not TrackEval)

Q0-native values are repeated in every JSON for direct comparison. Learned stream row count remains 99,476, but association/lifecycle differs:

| Route | Event both-reliable per model | Mean tracks | Mean ID switches | Mean fragmented GT tracks | Mean merged predictions | Mean duplicate-birth proxy | Physical gate |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(route_lines)}

Q0 native reference in these event videos is 1,026 tracks, 2,766 GT switches, 354 fragmented GT tracks, 383 merged predictions and 1,902 duplicate births. Learned routes therefore violate safety: switches and fragmentation increase, merged predictions and duplicate births increase (even when one fold has a lower individual count), and track births dominate. The top-4 route is the least-bad learned version in some folds but still has f3 only 5 continuations and 2,945 tracks. Full HOTA/DetA/AssA/IDF1/MOTA TrackEval was not rerun for learned streams because the native diagnostic exporter does not produce a TrackEval-compatible learned sequence; the Q0 full TrackEval anchor remains authoritative.

## Event observability (76-event denominator)

Event replay is reported only as an O diagnostic. It keeps all 76 positive events, prefixes 1/2/4/8/16, the original row keys and the post-inference reliable rule. The per-model prefix16 aggregates were:

- Initial: both `[53, 51, 52, 43]` (mean 49.75), source `[64,64,64,57]`, target `[64,61,63,59]`.
- Masked Q0-aligned: both `[57,61,61,61]` (mean 60.00), source `[68,72,72,72]`, target `[64,64,64,64]`.
- Resolution-aware: both `[61,57,61,61]` (mean 60.00), source `[72,68,72,72]`, target `[64,64,64,64]`.
- Top-4 context: both `[57,53,57,61]` (mean 57.00), source `[68,64,68,72]`, target `[64,64,64,64]`.

The complete 76-event index below gives Q0 source/target/both flags and the four model-fold flags (`1` reliable, `0` unreliable). This table is descriptive; it is not checkpoint selection.

{table}

## Gates and downstream routing

The registered Q0 physical gate required identical proposal rows, non-worse ID switches and false merges, and either improved fragmentation or materially improved association quality. Every learned route has `proposal_rows_identical=true` but `id_switch_non_worse=false`, `fragmentation_non_worse=false`, `merged_tracks_non_worse=false` and `duplicate_birth_non_worse=false`; all route decision files therefore state `FAIL_PHYSICAL_SAFETY`. Event O exceeded 35/76 for several routes, but this cannot override physical safety and is not an OCD result.

Because no learned physical stream passed Q0 safety, downstream raw-representation R, semantic StateMemory, Commit/Defer, persistent Commit-CT and sealed evaluation were **not run**. This is not a claim of OCD=0; it is the pre-registered causal routing decision. No controller threshold, memory, denominator or semantic architecture was changed.

## Resources, failures and integrity

- Four bounded workers mapped fold0–3 to GPUs 4/5/6/7; each long run used one supervisor and one blocking wait. RAM remained above the 25% headroom requirement; no OOM and no external process termination occurred.
- `/data1` was nearly full (~34 GiB free); large manifests/checkpoints/logs were written under `/data2/usr_for_deadline/trackocd_phase81p` and exposed through the project symlink `outputs/iclr27_phase81p`.
- Retained failures: initial unbounded manifest attempt (explicit task PIDs terminated), first replay growth/timeout (explicit task PIDs terminated), contract-smoke assertion and replay-shape fixes, and all superseded route artifacts. No partial result was relabelled as success.
- Atomic `.launched`/`.done` markers exist for all completed training/event/physical units. JSON files in the ledger and route decisions parse successfully; no active Phase81P worker remains.
- Source pushes: route-aware physical supervisor `f9a38fb`; masked NEW-context repair `f111b11`; resolution-aware route `2946a92`; top-4 context `832ad95`.

## Reproduction

Use the commands and hashes in `outputs/iclr27_phase81p/audit/validation_evidence_ledger.json`. The key final-route sequence is:

```bash
PYTHONPATH=. /home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase81p/train_association.py --fold 0 --device cuda:0 --route q0_train4t_topctx --data-root /data2/usr_for_deadline/trackocd_phase81p/data/q0_train4t_resaware --epochs 20 --balance-positive
scripts/iclr27_phase81p/run_four_fold_q0_aligned_supervisor.sh formal_q0topctx 20 /data2/usr_for_deadline/trackocd_phase81p/data/q0_train4t_resaware q0_train4t_topctx
scripts/iclr27_phase81p/run_replay_route_supervisor.sh q0_train4t_topctx formal_q0topctx outputs/iclr27_phase81p/checkpoints/q0_train4t_topctx 0 8 0 1
scripts/iclr27_phase81p/run_physical_replay_route_supervisor.sh q0_train4t_topctx q0topctx_physical outputs/iclr27_phase81p/checkpoints/q0_train4t_topctx 0 8 0 1
```

## Unverified items

1. Full learned-stream TrackEval HOTA/DetA/AssA/MOTA/IDF1 was not available from the diagnostic exporter; the conservative post-inference GT-join proxies above are sufficient to reject safety, but they are not paper metrics. Minimal follow-up is to implement a TrackEval-native learned lineage exporter before any future physical route.
2. A true Q0 exact-control replay for every event model fold was not rerun because Q0 native lineage is already frozen and the learned streams failed safety. The Q0 full validation anchor and native event counts are recorded with hashes.
3. Downstream R/controller/sealed metrics are intentionally absent, not zero, because the physical gate failed.

## Final decision

The three registered learned association versions (initial, resolution-aware, and top-4 context, with the masked-context repair retained) do not preserve the Q0 physical contract. The association model family is closed for this window. The strongest evidence-backed next action is **not** another threshold or small association variant: build a TrackEval-native, Q0-compatible TRAIN sequence contract with a dedicated physical-association teacher/validation split, then register a fresh route only after its supervision and lifecycle semantics are independently verified. Until that contract exists, do not connect semantic correspondence/controller or access sealed labels.
"""
    report_path = ROOT / "docs/iclr27_phase81p/PHASE81P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    ten_path = ROOT / "docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md"
    marker = "## Phase81P+ continuation (2026-09-03)"
    addition = f"""\n\n{marker}\n\nThe Q0-anchored physical association window ran from `{START}` with registered deadline `{DEADLINE}`. Four bounded GPU workers trained the initial association transformer and two evidence-based repairs (resolution-aware geometry and top-4 candidate-conditioned NEW context) on legal TRAIN-only Q0 proposal streams. Event observability rose from Q0 `25/76` to model means up to `60/76`, but this proxy gain did not survive physical safety: Q0 native event-stream proxies are 1,026 tracks/2,766 switches/354 fragmented GT tracks/383 merges/1,902 duplicate births, while the best learned route still averages 2,714 tracks/7,250 switches/403 fragmented tracks/1,028 merges/5,606 duplicate births. Therefore physical safety failed for all three versions; no controller or sealed evaluation was run. Source code was pushed through `{end_head}`; full evidence is in [`PHASE81P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md`](docs/iclr27_phase81p/PHASE81P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md) and `outputs/iclr27_phase81p/audit/validation_evidence_ledger.json`.\n"""
    old = ten_path.read_text() if ten_path.exists() else "# Autonomous TrackOCD 10H Research Report\n"
    if marker not in old:
        ten_path.write_text(old.rstrip() + addition)

    log = ROOT / "research_log.md"
    log_marker = "## Phase81P+ final association-family closure"
    log_text = f"""\n\n{log_marker} (2026-09-03)\n\n- Registered window `{START}`→`{DEADLINE}`, start HEAD `{START_HEAD}`, ending HEAD `{end_head}`. Q0 proposal/boxes/base score, 76-event denominator, row keys and causal evaluator stayed frozen; DEV+/Q1/public-new/sealed remained sealed.\n- Q0/Phase80C reference: source/target pool-good 72/76 and 64/76, strict source/target/both 49/76, 40/76, 25/76; 36 assignment-gap events (29 fragmentation, 7 missed continuation).\n- Initial association training/replay exposed NEW-context calibration mismatch and physical collapse (mean 362 tracks; see route JSON). Masked candidate context repair (`f111b11`) retained evidence but still produced 2,895 mean tracks, 7,354 switches, 403 fragmentation, 1,197 merges, 6,645 duplicate births.\n- Resolution-aware repair (`2946a92`) used actual current-frame dimensions (event rows include 1280×720, 1920×1080/1200, 640×480) and was TRAIN/event causal; physical mean 2,834 tracks, 7,343 switches, 403 fragmentation, 1,085 merges, 6,025 duplicate births.\n- Final top-4 candidate-conditioned NEW context (`832ad95`) improved some single-video continuation counts but physical mean remained 2,714 tracks, 7,250 switches, 403 fragmentation, 1,028 merges, 5,606 duplicate births; fold3 had only five continuations. All three learned versions failed the pre-registered Q0 physical gate.\n- Event O proxy means were initial 49.75/76, masked 60/76, resolution-aware 60/76, top-4 57/76. These are not MOT/OCD success and did not authorize controller/R/sealed evaluation.\n- No OOM or external-process termination occurred. Large artifacts use `/data2/usr_for_deadline/trackocd_phase81p` via `outputs/iclr27_phase81p` symlink. All code changes were pushed before report generation; route decisions and hashes are in `outputs/iclr27_phase81p/audit/`.\n- Decision: close this physical-association model family after three evidence-based versions; do not threshold-sweep. Next research should first build a TrackEval-native Q0-compatible TRAIN supervision/lifecycle contract, then register a new route only with independent causal validation.\n"""
    if log_marker not in log.read_text():
        with log.open("a") as f:
            f.write(log_text)

    print(json.dumps({"report": str(report_path), "ledger": str(audit_dir / "validation_evidence_ledger.json"), "ending_head": end_head, "rendered_utc": now.isoformat()}, indent=2))


if __name__ == "__main__":
    main()
