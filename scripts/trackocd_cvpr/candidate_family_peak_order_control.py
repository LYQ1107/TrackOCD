#!/usr/bin/env python3
"""Fixed-order control for the family-relative candidate mechanism."""

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
    support_agreement_plan,
)
from src.trackocd_v2.io import atomic_json

DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_family_peak_order_control_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_order_control_v1/result.json"


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
    if config.get("schema_version") != "trackocd.cvpr.candidate-family-peak-order-control.v1":
        raise ValueError("unexpected family-peak order-control configuration")
    if config.get("status") != "FROZEN_CATEGORY_FREE_FAMILY_PEAK_ORDER_CONTROL_NO_TRAINING":
        raise ValueError("family-peak order control is not frozen")
    if config.get("orders") != [[0, 1], [1, 0]] or config.get("frames_per_video") != 2:
        raise ValueError("exact two-order/two-frame scope required")
    if config.get("limits") != {
        "optimizer_steps": 0,
        "training_gpu_hours": 0,
        "device": "CPU",
        "workers": 1,
    }:
        raise ValueError("order-control limits are not closed")
    expected_boundaries = {
        "labels_external_loss_only": True,
        "labels_or_gt_in_numeric_forward": False,
        "category_or_vocabulary_in_numeric_forward": False,
        "val_or_test_accessed": False,
        "optimizer_started": False,
        "formal_metrics": False,
        "model_selection": False,
        "order_winner_selected": False,
        "predicted_frontend_qualification": False,
        "harness_forced_cross_video_reuse": False,
        "scientific_acceptance": False,
    }
    if config.get("boundaries") != expected_boundaries:
        raise ValueError("order-control boundary is not closed")
    joint = ROOT / config["joint_rcm_config"]
    if _sha256(joint) != config["joint_rcm_config_sha256"]:
        raise ValueError("registered joint RCM configuration changed")
    return config


def _controller(rcm_config: dict, known: dict[int, object], support: list[tuple]) -> JointRCMAtomicMemberControl:
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
    return controller


def _run_order(order, videos, projection, rcm_config, support, known, frame_limit):
    controller = _controller(rcm_config, known, support)
    external_targets = {
        (video, str(track)): target
        for (video, track), target in projection.targets.items()
    }
    selected = [videos[index] for index in order]
    epoch_to_video = {epoch: int(video) for epoch, (video, _arrays) in enumerate(selected)}
    reasons: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    token_epochs: dict[str, set[int]] = {}
    events = m_support_steps = positive_other_video = frames = 0
    for epoch, (_video, arrays) in enumerate(selected):
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
                m_support_steps += int(any(value is not None for value in plan.targets))
                positive_other_video += int(plan.reason == "EXTERNAL_FULL_OTHER_GROUP_AGREEMENT")
            events += len(result.events)
            frames += 1
            if frames % frame_limit == 0:
                break
    return {
        "order": list(order),
        "videos": len(selected),
        "frames": frames,
        "events": events,
        "actions": dict(sorted(actions.items())),
        "cross_video_token_count": sum(len(epochs) >= 2 for epochs in token_epochs.values()),
        "m_support_steps": m_support_steps,
        "positive_other_video_groups": positive_other_video,
        "plan_reasons": dict(sorted(reasons.items())),
        "memory_versions": len(controller.memory_versions()),
    }


def run(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing order-control receipt")
    started = time.monotonic()
    _install_input_barrier()
    rcm_config = json.loads((ROOT / config["joint_rcm_config"]).read_text(encoding="utf-8"))
    videos, projection, _ = load_completed_policy_videos(ROOT, rcm_config, lambda: None)
    fit_tracks, _ = load_completed_fit_tracks(ROOT, rcm_config, lambda: None)
    support = support_groups(fit_tracks, -1, 5)
    frozen = RawPreservingEvidence().eval().requires_grad_(False)
    with torch.inference_mode():
        raw, _, _, _ = reencode_batched_supports(frozen, support, torch.ones(5), torch.ones(5))
    known = {index: row.numpy().copy() for index, row in enumerate(raw)}
    per_order = [
        _run_order(order, videos, projection, rcm_config, support, known, config["frames_per_video"])
        for order in config["orders"]
    ]
    result = {
        "schema_version": "trackocd.cvpr.candidate-family-peak-order-control-result.v1",
        "status": "PASS_FAMILY_PEAK_ORDER_CONTROL_SMOKE",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "mechanism": {
            "name": config["mechanism"],
            "orders_preregistered": config["orders"],
            "order_winner_selected": False,
            "score_sweep": False,
        },
        "per_order": per_order,
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
        "orders": [row["order"] for row in result["per_order"]],
        "m_support_steps": [row["m_support_steps"] for row in result["per_order"]],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
