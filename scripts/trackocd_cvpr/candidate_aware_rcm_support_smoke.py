#!/usr/bin/env python3
"""Bounded candidate-aware lifecycle smoke with external M-support audit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import torch

from scripts.trackocd_cvpr.candidate_aware_lifecycle_smoke import SimilarityLifecycleScorer
from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos, policy_current_frames
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.reliability_training_loss import (
    prediction_owned_support_targets,
    support_agreement_plan,
)
from src.trackocd_v2.io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_aware_rcm_support_smoke_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_aware_rcm_support_smoke_v1/result.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.candidate-aware-rcm-support-smoke.v1":
        raise ValueError("unexpected candidate-aware RCM support configuration")
    if config.get("status") != "FROZEN_CATEGORY_FREE_CANDIDATE_AWARE_RCM_SUPPORT_SMOKE_NO_TRAINING":
        raise ValueError("candidate-aware RCM support smoke is not frozen")
    if config.get("fixed_similarity_reuse_threshold") != 0.55:
        raise ValueError("only the fixed registered similarity margin is allowed")
    if config.get("pilot_policy_relative_ordinals") != [0, 1] or config.get("frames_per_video") != 2:
        raise ValueError("fixed two-video/two-frame scope required")
    return config


def _install_input_barrier() -> None:
    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("candidate-aware RCM support smoke forbids external activity")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            forbidden = (
                "/TAO-Amodal/", "/frames/val/", "/frames/test/", "/roles.json",
                ".pth", ".pt", "evaluator_only_geometry_join.json",
            )
            if any(token in path for token in forbidden):
                raise PermissionError("candidate-aware RCM support smoke forbids prohibited input")

    import sys

    sys.addaudithook(audit)


def run(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing candidate-aware RCM support receipt")
    started = time.monotonic()
    _install_input_barrier()
    rcm_config = json.loads((ROOT / config["joint_rcm_config"]).read_text(encoding="utf-8"))
    videos, projection, _policy_proof = load_completed_policy_videos(ROOT, rcm_config, lambda: None)
    fit_tracks, _fit_proof = load_completed_fit_tracks(ROOT, rcm_config, lambda: None)
    support = support_groups(fit_tracks, -1, 5)
    frozen_evidence = RawPreservingEvidence().eval().requires_grad_(False)
    with torch.inference_mode():
        raw, _, _, _ = reencode_batched_supports(
            frozen_evidence, support, torch.ones(5), torch.ones(5)
        )
    known = {index: row.numpy().copy() for index, row in enumerate(raw)}
    controller = JointRCMAtomicMemberControl(
        RawPreservingEvidence().eval(),
        CandidateScoreMLP(feature_count=14).eval(),
        MemberReliabilityMLP().eval(),
        known,
        support,
        quarantine=True,
        K=8,
        bank_limits={
            "max_members": rcm_config["physical_tracks_bound"],
            "max_tokens": rcm_config["observations_bound"],
            "max_commits": rcm_config["observations_bound"],
            "max_payload_bytes": rcm_config["payload_bytes_bound"],
        },
    )
    controller.scorer = SimilarityLifecycleScorer(config["fixed_similarity_reuse_threshold"])
    external_targets = {
        (video, str(track)): target
        for (video, track), target in projection.targets.items()
    }
    fixed_videos = videos[:2]
    epoch_to_video = {epoch: int(video) for epoch, (video, _arrays) in enumerate(fixed_videos)}
    reasons: Counter[str] = Counter()
    event_count = m_support_steps = positive_other_video = 0
    token_epochs: dict[str, set[int]] = {}
    frames_consumed = 0
    for epoch, (_video, arrays) in enumerate(fixed_videos):
        if epoch:
            controller.start_next_video()
        for frame, observations in policy_current_frames(arrays):
            prior_bank, _ = controller.state_snapshot()
            result, _ = controller.ingest(frame, observations, track_grad=False)
            for token in result.routed_predictions():
                event = token[1]
                if event.token is not None:
                    token_epochs.setdefault(event.token, set()).add(epoch)
            for token, _reliability in controller.preframe_training_evidence().items():
                labels = prediction_owned_support_targets(
                    prior_bank, token, external_targets, epoch_to_video
                )
                plan = support_agreement_plan(labels)
                reasons[plan.reason] += 1
                if any(value is not None for value in plan.targets):
                    m_support_steps += 1
                if plan.reason == "EXTERNAL_FULL_OTHER_GROUP_AGREEMENT":
                    positive_other_video += 1
            event_count += len(result.events)
            frames_consumed += 1
            if frames_consumed % config["frames_per_video"] == 0:
                break
    cross_video_token_count = sum(len(epochs) >= 2 for epochs in token_epochs.values())
    m_steps = m_support_steps > 0 and positive_other_video > 0
    result = {
        "schema_version": "trackocd.cvpr.candidate-aware-rcm-support-smoke-result.v1",
        "status": "PASS_CANDIDATE_AWARE_RCM_M_SUPPORT_SMOKE" if m_steps else "FAIL_CANDIDATE_AWARE_RCM_M_SUPPORT_ZERO",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()},
        "lifecycle": {
            "videos": len(fixed_videos),
            "frames": frames_consumed,
            "events": event_count,
            "cross_video_token_count": int(cross_video_token_count),
            "memory_versions": len(controller.memory_versions()),
            "harness_forced_cross_video_reuse": False,
        },
        "m_support": {
            "m_support_steps": int(m_support_steps),
            "positive_other_video_groups": int(positive_other_video),
            "plan_reasons": dict(sorted(reasons.items())),
        },
        "boundaries": config["boundaries"],
        "resources": {
            "device": "CPU",
            "workers": 1,
            "optimizer_steps": 0,
            "training_gpu_hours": 0.0,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "wall_seconds": time.monotonic() - started,
        },
        "interpretation": config["interpretation"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "cross_video_token_count": result["lifecycle"]["cross_video_token_count"],
        "m_support_steps": result["m_support"]["m_support_steps"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
