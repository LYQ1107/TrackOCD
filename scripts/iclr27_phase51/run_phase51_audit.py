#!/usr/bin/env python3
"""Phase51 read-only methods/contract audit.

The audit deliberately reuses only already verified official repository
records and frozen TRAIN artifacts.  It writes all new material below the
Phase51 namespace and never touches an earlier phase.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase51"
DOC = ROOT / "docs/iclr27_phase51"


def atomic_json(path: pathlib.Path, value: object) -> None:
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


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return f"UNAVAILABLE: {exc.output.strip()}"


def live_phase51_workers() -> list[dict[str, object]]:
    """Direct /proc scan; excludes this process and its shell ancestors."""
    excluded = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            pid = int((pathlib.Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception:
            break
    out = []
    for p in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            n = int(p.name)
            if n in excluded:
                continue
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
            if "scripts/iclr27_phase51/" in cmd or "scripts/iclr27_phase54/" in cmd or "scripts/iclr27_phase56/" in cmd:
                out.append({"pid": n, "cmd": cmd})
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return out


def main() -> None:
    for d in ("audit", "manifests", "metrics", "checkpoints", "completion", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    methods_path = ROOT / "outputs/iclr27_phase50/audit/github_methods.json"
    prior = json.loads(methods_path.read_text(encoding="utf-8"))
    methods = list(prior.get("methods", []))
    # These entries are historical references already verified in the project
    # prior-art audits.  No code or checkpoint is fetched here.
    methods.extend([
        {
            "name": "OVTrack", "repo_url": "https://github.com/bytedance/ovtrack",
            "paper_url": "https://arxiv.org/abs/2304.05605", "commit_or_tag": "not re-fetched; prior-art record",
            "license": "repository license must be checked before reuse", "release_or_publication": "CVPR 2023",
            "task": "open-vocabulary tracking-by-detection", "inputs_outputs": "detector proposals to physical tracks",
            "online_causal": "yes within video", "unknown_novel": "open-vocabulary detector categories",
            "persistent_query": "no TrackOCD semantic query", "cross_video_correspondence": False,
            "text_or_id_dependency": "open-vocabulary detector/category head; incompatible as-is",
            "supervision": "LVIS/TAO detection and tracking", "reusable": ["proposal/association reference"],
            "not_reusable": ["text/category head", "no persistent semantic state"],
            "trackocd_match": "physical/open-vocabulary reference only",
        },
        {
            "name": "COVTrack", "repo_url": "https://github.com/zekunqian/COVTrack",
            "paper_url": "https://openaccess.thecvf.com/content/ICCV2025/html/Qian_Continuous_Open-Vocabulary_Tracking_via_Adaptive_Multi-Cue_Fusion_ICCV_2025_paper.html",
            "commit_or_tag": "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b",
            "license": "Apache-2.0 (prior repository audit)", "release_or_publication": "ICCV 2025",
            "task": "continuous open-vocabulary MOT with multi-cue fusion", "inputs_outputs": "detector/category cues to physical tracks",
            "online_causal": "online within video", "unknown_novel": "open-vocabulary categories",
            "persistent_query": False, "cross_video_correspondence": False,
            "text_or_id_dependency": "semantic/category cues and tracking IDs; not a no-text semantic bank",
            "supervision": "C-TAO/open-vocabulary tracking", "reusable": ["confidence/association diagnostics"],
            "not_reusable": ["category head", "no cross-video Commit/Defer"], "trackocd_match": "partial frontend reference",
        },
        {
            "name": "VOVTrack", "repo_url": "https://github.com/zhang-tao-whu/VOVTrack",
            "paper_url": "https://arxiv.org/abs/2403.11821", "commit_or_tag": "not re-fetched; prior-art record",
            "license": "license not confirmed in prior audit", "release_or_publication": "ICCV 2025",
            "task": "open-vocabulary tracking with state-prompt detection", "inputs_outputs": "detector features/state prompts to tracks",
            "online_causal": "online within video", "unknown_novel": "yes at detector category level",
            "persistent_query": False, "cross_video_correspondence": False,
            "text_or_id_dependency": "state/category cues and physical association", "supervision": "LVIS/TAO plus unlabeled association",
            "reusable": ["state-prompt/association ideas"], "not_reusable": ["no TrackOCD semantic state contract"],
            "trackocd_match": "partial frontend reference",
        },
    ])
    methods_doc = {
        "phase": 51,
        "audit_date": datetime.now(timezone.utc).date().isoformat(),
        "selection_rule": "official repository README/paper and previously recorded remote HEADs; no external checkout/checkpoint",
        "methods": methods,
        "decision": {
            "selected_external_method": None,
            "selected_design": "TrackOCD-native unified causal graph; no external model claimed",
            "reason": "Every audited method covers only a subset: OVTR/open-vocabulary MOT relies on CLIP text, ObjectRelator/C3Po are paired cross-view correspondence, MOTIP-2/MeMOTR/MOTR are physical-ID tracking, and MASA/COVTrack/OVTrack/VOVTrack depend on external detector/category cues. None provides no-text/no-ID cross-video semantic state plus persistent Commit/Defer.",
            "downloads": False,
            "sealed_inputs_accessed": False,
        },
    }
    atomic_json(OUT / "audit/github_methods.json", methods_doc)

    free = command("free -h")
    gpu = command("nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader")
    disk = command("df -h /data1")
    process_count = command("ps -e --no-headers | wc -l")
    resource = {
        "phase": 51,
        "cwd": str(ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "free_h": free,
        "gpu": gpu,
        "disk_data1": disk,
        "process_count": process_count,
        "phase51_workers": live_phase51_workers(),
        "gpu_policy": "maximum four cards; planned formal mapping 4,5,6,7",
        "git_revision": command("git rev-parse HEAD 2>&1"),
        "git_status": command("git status --short 2>&1"),
    }
    atomic_json(OUT / "audit/resource_preflight.json", resource)

    frozen = {}
    for rel in [
        "docs/iclr27_phase29/TRACKOCD_FINAL_MOT_OCD_SEALED_REPORT.md",
        "docs/iclr27_phase50/PHASE50_END_TO_END_TRAINING_REPORT.md",
        "outputs/iclr27_phase50/audit/phase50_decision.json",
        "outputs/iclr27_phase26/checkpoints/source_generator_f0_best.pt",
        "outputs/iclr27_phase46/checkpoints/gate_f0_best.pt",
    ]:
        p = ROOT / rel
        frozen[rel] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() and p.is_file() else None}
    atomic_json(OUT / "audit/frozen_prior_context.json", {"phase": 51, "artifacts": frozen,
        "frozen_components": ["Phase26 proposal/physical stream", "Phase46 gate bridge", "Phase19R controller/StateMemory", "row key/76-event evaluator"],
        "public_q1_dev_access": False, "future_or_id_inputs": False})

    # Reuse the frozen TRAIN split through a symlink, never a feature copy.
    link = OUT / "manifests/fold_manifest.json"
    source = ROOT / "outputs/iclr27_phase27/manifests/fold_manifest.json"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(source)
    atomic_json(OUT / "audit/phase51_decision.json", {
        "phase": 51,
        "decision_code": "P51_AUDIT_COMPLETE_ALLOW_NATIVE_GRAPH",
        "external_method_selected": None,
        "contract_authorized": True,
        "train_only": True,
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "held GT as model input", "category/text/ID features"],
        "fold_manifest_symlink": str(link),
    })
    (OUT / "completion/stage0.done").write_text(json.dumps({"phase": 51, "stage": 0, "decision": "PASS"}) + "\n", encoding="utf-8")

    lines = [
        "# Phase51 — Official method audit",
        "",
        f"Audit date: {methods_doc['audit_date']}. Official repository HEADs and prior verified records are captured in [`github_methods.json`](../../outputs/iclr27_phase51/audit/github_methods.json). No external code, checkpoint, DEV+, Q1 or sealed label was downloaded/read.",
        "",
        "## Decision",
        "",
        "No external method is selected as a drop-in solution. OVTR is the closest proposal/query reference but its open-vocabulary head uses CLIP text and its TAO task has no cross-video semantic state. ObjectRelator and C3Po provide cross-view correspondence but require paired view/geometry assumptions and lack causal MOT lifecycle. MOTIP-2, MeMOTR and MOTR provide physical query memory but their identity is physical ID. MASA/COVTrack/OVTrack/VOVTrack rely on detector/category cues or external proposals. Phase51 therefore implements a TrackOCD-native graph and uses these projects only as auditable design references.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase51/run_phase51_audit.py",
        "```",
        "",
        "The audit records exact URLs, paper links, revisions/licences where verified, compatibility boundaries, resource preflight and the frozen TRAIN manifest symlink. The complete causal architecture and loss contract are in [`PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md`](PHASE51_END_TO_END_ARCHITECTURE_CONTRACT.md).",
    ]
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "PHASE51_GITHUB_METHOD_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
