#!/usr/bin/env python3
"""Bounded one-step M-branch smoke after family-relative anonymous routing."""

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
from scripts.trackocd_cvpr.candidate_family_peak_rcm_support_smoke import FamilyPeakScorer
from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos, policy_current_frames
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.reliability_training_loss import (
    prediction_owned_support_targets,
    support_agreement_losses,
    support_agreement_plan,
)
from src.trackocd_v2.io import atomic_json

DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_family_peak_m_training_smoke_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_m_training_smoke_v1/result.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.candidate-family-peak-m-training-smoke.v1":
        raise ValueError("unexpected family-peak M-training configuration")
    if config.get("status") != "FROZEN_POLICY_TRAIN_FAMILY_PEAK_M_BRANCH_SMOKE_NO_PRED_OR_GOAL_ACCEPTANCE":
        raise ValueError("family-peak M-training smoke is not frozen")
    if config.get("pilot_policy_relative_ordinals") != [0, 1] or config.get("frames_per_video") != 2:
        raise ValueError("fixed two-video/two-frame scope required")
    if config.get("max_optimizer_steps") != 1 or config.get("limits") != {
        "optimizer_steps": 1,
        "training_gpu_hours": 0,
        "device": "CPU",
        "workers": 1,
    }:
        raise ValueError("exact one-step CPU M scope required")
    if config.get("boundaries") != {
        "labels_external_loss_only_after_decision": True,
        "labels_or_gt_in_numeric_forward": False,
        "category_or_vocabulary_in_numeric_forward": False,
        "val_or_test_accessed": False,
        "predicted_track_training": False,
        "model_selection": False,
        "formal_metrics": False,
        "goal_acceptance": False,
        "weights_uploaded": False,
        "external_processes_touched": False,
    }:
        raise ValueError("M-training boundary is not closed")
    if _sha256(ROOT / config["joint_rcm_config"]) != config["joint_rcm_config_sha256"]:
        raise ValueError("registered joint RCM configuration changed")
    return config


def run(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing M-training receipt")
    started = time.monotonic()
    _install_input_barrier()
    rcm_config = json.loads((ROOT / config["joint_rcm_config"]).read_text(encoding="utf-8"))
    videos, projection, _ = load_completed_policy_videos(ROOT, rcm_config, lambda: None)
    fit_tracks, _ = load_completed_fit_tracks(ROOT, rcm_config, lambda: None)
    support = support_groups(fit_tracks, -1, 5)
    frozen_evidence = RawPreservingEvidence().eval().requires_grad_(False)
    with torch.inference_mode():
        raw, _, _, _ = reencode_batched_supports(
            frozen_evidence, support, torch.ones(5), torch.ones(5)
        )
    known = {index: row.numpy().copy() for index, row in enumerate(raw)}
    member = MemberReliabilityMLP().train()
    controller = JointRCMAtomicMemberControl(
        frozen_evidence,
        CandidateScoreMLP(feature_count=14).eval(),
        member,
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
    optimizer = torch.optim.AdamW(
        member.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    before_digest = _parameter_digest(member)
    external_targets = {
        (video, str(track)): target
        for (video, track), target in projection.targets.items()
    }
    fixed_videos = videos[:2]
    epoch_to_video = {epoch: int(video) for epoch, (video, _arrays) in enumerate(fixed_videos)}
    reasons: Counter[str] = Counter()
    frames = events = positive_groups = support_plans = 0
    losses: list[float] = []
    gradient_norms: list[float] = []
    optimizer_steps = 0
    for epoch, (_video, arrays) in enumerate(fixed_videos):
        if epoch:
            controller.start_next_video()
        video_frames = 0
        for frame, observations in policy_current_frames(arrays):
            prior_bank, _ = controller.state_snapshot()
            result, _ = controller.ingest(frame, observations, track_grad=False)
            events += len(result.events)
            m_terms = []
            for token, reliability in controller.preframe_training_evidence().items():
                labels = prediction_owned_support_targets(
                    prior_bank, token, external_targets, epoch_to_video
                )
                plan = support_agreement_plan(labels)
                reasons[plan.reason] += 1
                support_plans += 1
                if plan.reason == "EXTERNAL_FULL_OTHER_GROUP_AGREEMENT":
                    positive_groups += 1
                m_terms.extend(loss for _, loss in support_agreement_losses(reliability.logits, plan))
            if m_terms and optimizer_steps < config["max_optimizer_steps"]:
                loss = torch.stack(m_terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(member.parameters(), config["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                    raise ValueError("finite nonzero M gradient required")
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
                gradient_norms.append(float(norm))
            frames += 1
            video_frames += 1
            if video_frames >= config["frames_per_video"]:
                break
    after_digest = _parameter_digest(member)
    result = {
        "schema_version": "trackocd.cvpr.candidate-family-peak-m-training-smoke-result.v1",
        "status": "PASS_FAMILY_PEAK_M_BRANCH_TRAINING_SMOKE" if optimizer_steps == 1 else "FAIL_FAMILY_PEAK_M_BRANCH_TRAINING_SMOKE",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "scope": {
            "videos": len(fixed_videos),
            "frames": frames,
            "events": events,
            "support_plans": support_plans,
            "positive_other_video_groups": positive_groups,
            "plan_reasons": dict(sorted(reasons.items())),
        },
        "training": {
            "branch": "M_member_reliability_only",
            "optimizer": config["optimizer"],
            "optimizer_steps": optimizer_steps,
            "losses": losses,
            "gradient_norms": gradient_norms,
            "parameter_digest_before": before_digest,
            "parameter_digest_after": after_digest,
            "parameters_changed": before_digest != after_digest,
            "predicted_track_training": False,
        },
        "boundaries": config["boundaries"],
        "resources": {
            "device": "CPU",
            "workers": 1,
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
        "optimizer_steps": result["training"]["optimizer_steps"],
        "positive_other_video_groups": result["scope"]["positive_other_video_groups"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
