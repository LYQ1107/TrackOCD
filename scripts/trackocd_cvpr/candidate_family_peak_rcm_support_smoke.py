#!/usr/bin/env python3
"""Bounded family-relative candidate rule smoke with external M-support audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.candidate_aware_rcm_support_smoke import _install_input_barrier
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
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_family_peak_rcm_support_smoke_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_rcm_support_smoke_v1/result.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.candidate-family-peak-rcm-support-smoke.v1":
        raise ValueError("unexpected family-peak RCM configuration")
    if config.get("status") != "FROZEN_CATEGORY_FREE_FAMILY_PEAK_RCM_SUPPORT_SMOKE_NO_TRAINING":
        raise ValueError("family-peak RCM smoke is not frozen")
    if config.get("mechanism") != "family_peak_standardization":
        raise ValueError("unexpected family-peak mechanism")
    if config.get("pilot_policy_relative_ordinals") != [0, 1] or config.get("frames_per_video") != 2:
        raise ValueError("fixed two-video/two-frame scope required")
    if config.get("limits") != {
        "optimizer_steps": 0,
        "training_gpu_hours": 0,
        "device": "CPU",
        "workers": 1,
    }:
        raise ValueError("family-peak smoke limits are not closed")
    if config.get("boundaries") != {
        "labels_external_loss_only": True,
        "labels_or_gt_in_numeric_forward": False,
        "category_or_vocabulary_in_numeric_forward": False,
        "val_or_test_accessed": False,
        "optimizer_started": False,
        "formal_metrics": False,
        "model_selection": False,
        "predicted_frontend_qualification": False,
        "harness_forced_cross_video_reuse": False,
        "scientific_acceptance": False,
    }:
        raise ValueError("family-peak boundary is not closed")
    joint_config = ROOT / config["joint_rcm_config"]
    if _sha256(joint_config) != config["joint_rcm_config_sha256"]:
        raise ValueError("registered joint RCM configuration changed")
    return config


class FamilyPeakScorer:
    """Compare candidate families after deterministic within-family scaling."""

    def __call__(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 14 or context.shape != (6,):
            raise ValueError("unexpected candidate bridge shape")
        if features.dtype != torch.float32 or context.dtype != torch.float32:
            raise ValueError("registered float32 candidate bridge required")
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError("finite candidate bridge required")
        scores = torch.full((len(features) + 2,), -100.0, dtype=torch.float32)
        existing = torch.where(features[:, 1] < 0.5)[0]
        known = torch.where(features[:, 1] >= 0.5)[0]

        def peak(indices: torch.Tensor) -> tuple[float, int | None]:
            if not len(indices):
                return float("-inf"), None
            values = features[indices, 0]
            scale = values.std(unbiased=False) + torch.finfo(values.dtype).eps
            standardized = (values - values.mean()) / scale
            position = int(torch.argmax(standardized))
            return float(standardized[position]), int(indices[position])

        existing_peak, existing_index = peak(existing)
        known_peak, _ = peak(known)
        if existing_index is not None and existing_peak >= known_peak:
            scores[existing_index] = 10.0 + features[existing_index, 0]
        else:
            scores[len(features)] = 10.0
        return scores


def run(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing family-peak receipt")
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
    controller.scorer = FamilyPeakScorer()
    external_targets = {
        (video, str(track)): target
        for (video, track), target in projection.targets.items()
    }
    fixed_videos = videos[:2]
    epoch_to_video = {epoch: int(video) for epoch, (video, _arrays) in enumerate(fixed_videos)}
    reasons: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    event_count = m_support_steps = positive_other_video = 0
    token_epochs: dict[str, set[int]] = {}
    frames_consumed = 0
    for epoch, (_video, arrays) in enumerate(fixed_videos):
        if epoch:
            controller.start_next_video()
        for frame, observations in policy_current_frames(arrays):
            prior_bank, _ = controller.state_snapshot()
            result, _ = controller.ingest(frame, observations, track_grad=False)
            for event in result.events:
                actions[event.action] += 1
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
    support_sizes = Counter(
        len(row.payloads)
        for row in controller.state_snapshot()[0].raw_support_snapshot()
    )
    positive = m_support_steps > 0 and positive_other_video > 0
    result = {
        "schema_version": "trackocd.cvpr.candidate-family-peak-rcm-support-smoke-result.v1",
        "status": "PASS_FAMILY_PEAK_RCM_M_SUPPORT_SMOKE" if positive else "FAIL_FAMILY_PEAK_RCM_M_SUPPORT_ZERO",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "mechanism": {
            "name": config["mechanism"],
            "hand_selected_similarity_threshold": False,
            "score_sweep": False,
            "tie_break": "EXISTING",
        },
        "lifecycle": {
            "videos": len(fixed_videos),
            "frames": frames_consumed,
            "events": event_count,
            "actions": dict(sorted(actions.items())),
            "cross_video_token_count": int(cross_video_token_count),
            "memory_versions": len(controller.memory_versions()),
            "harness_forced_cross_video_reuse": False,
        },
        "m_support": {
            "m_support_steps": int(m_support_steps),
            "positive_other_video_groups": int(positive_other_video),
            "plan_reasons": dict(sorted(reasons.items())),
        },
        "support_group_size_histogram": {
            str(size): int(count) for size, count in sorted(support_sizes.items())
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
        "positive_other_video_groups": result["m_support"]["positive_other_video_groups"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
