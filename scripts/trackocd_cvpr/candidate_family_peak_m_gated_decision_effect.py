#!/usr/bin/env python3
"""Paired short-clock control for an M-adjusted candidate decision score."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
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
    support_agreement_losses,
    support_agreement_plan,
)
from src.trackocd_v2.io import atomic_json

DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_family_peak_m_gated_decision_effect_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_m_gated_decision_effect_v1/result.json"


class MAdjustedFamilyPeakScorer:
    """Fixed numeric scorer using adapted-vs-raw change as the M-sensitive term."""

    def __call__(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 14 or context.shape != (6,):
            raise ValueError("unexpected M-adjusted candidate bridge shape")
        if features.dtype != torch.float32 or context.dtype != torch.float32:
            raise ValueError("registered float32 candidate bridge required")
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError("finite M-adjusted candidate bridge required")
        scores = torch.full((len(features) + 2,), -100.0, dtype=torch.float32)
        existing = torch.where(features[:, 1] < 0.5)[0]
        known = torch.where(features[:, 1] >= 0.5)[0]
        adjusted = features[:, 0] + features[:, 12]

        def peak(indices: torch.Tensor) -> tuple[float, int | None]:
            if not len(indices):
                return float("-inf"), None
            values = adjusted[indices]
            scale = values.std(unbiased=False) + torch.finfo(values.dtype).eps
            standardized = (values - values.mean()) / scale
            position = int(torch.argmax(standardized))
            return float(standardized[position]), int(indices[position])

        existing_peak, existing_index = peak(existing)
        known_peak, _ = peak(known)
        if existing_index is not None and existing_peak >= known_peak:
            scores[existing_index] = 10.0 + adjusted[existing_index]
        else:
            scores[len(features)] = 10.0
        return scores


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


def _event_digest(frame: int, aliases, events) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame).encode("ascii"))
    for group, event in zip(aliases, events):
        digest.update(b"|")
        digest.update("\x1f".join(group).encode("utf-8"))
        digest.update(event.action.encode("ascii"))
        digest.update(str(event.token).encode("utf-8"))
        digest.update(event.candidate_feature_bytes)
        digest.update(event.query_context_bytes)
        digest.update(event.score_bytes)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.candidate-family-peak-m-gated-decision-effect.v1":
        raise ValueError("unexpected M-gated decision configuration")
    if config.get("status") != "FROZEN_POLICY_TRAIN_M_GATED_DECISION_EFFECT_NO_PRED_OR_GOAL_ACCEPTANCE":
        raise ValueError("M-gated decision scope is not frozen")
    if config.get("pilot_policy_relative_ordinals") != [0, 1] or config.get("frames_per_video") != 8:
        raise ValueError("fixed two-video/eight-frame scope required")
    if config.get("online_optimizer_steps") != 1:
        raise ValueError("exact one-step online M scope required")
    if config.get("limits") != {
        "frozen_optimizer_steps": 0,
        "online_optimizer_steps": 1,
        "training_gpu_hours": 0,
        "device": "CPU",
        "workers": 1,
    }:
        raise ValueError("M-gated limits are not closed")
    if config.get("scorer", {}).get("feature_indices") != {"adapted_cos": 0, "adapted_minus_raw": 12}:
        raise ValueError("registered M-sensitive feature bridge changed")
    if config.get("boundaries", {}).get("val_or_test_accessed") is not False:
        raise ValueError("Val/Test boundary is not closed")
    joint_config = ROOT / config["joint_rcm_config"]
    if _sha256(joint_config) != config["joint_rcm_config_sha256"]:
        raise ValueError("registered joint RCM configuration changed")
    return config


def _make_controller(rcm_config, known, support, member):
    evidence = RawPreservingEvidence().eval().requires_grad_(False)
    controller = JointRCMAtomicMemberControl(
        evidence,
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
    controller.scorer = MAdjustedFamilyPeakScorer()
    return controller


def _run_arm(config, rcm_config, fixed_videos, known, support, *, online):
    torch.manual_seed(int(config.get("seed", 1027)))
    member = MemberReliabilityMLP().train(online)
    controller = _make_controller(rcm_config, known, support, member)
    optimizer = torch.optim.AdamW(
        member.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    ) if online else None
    before_digest = _parameter_digest(member)
    videos_to_epoch = {epoch: int(video) for epoch, (video, _arrays) in enumerate(fixed_videos)}
    external_targets = {}
    # The caller fills this prediction-owned projection after construction.
    actions: Counter[str] = Counter()
    event_digests = []
    action_digests = []
    update_frames = []
    losses = []
    gradients = []
    plan_reasons: Counter[str] = Counter()
    frames = events = support_plans = positive_groups = optimizer_steps = 0
    for epoch, (_video, arrays) in enumerate(fixed_videos):
        if epoch:
            controller.start_next_video()
        consumed = 0
        for frame, observations in policy_current_frames(arrays):
            prior_bank, _ = controller.state_snapshot()
            result, _ = controller.ingest(frame, observations, track_grad=False)
            actions.update(event.action for event in result.events)
            event_digests.append(_event_digest(frame, result.routing_groups, result.events))
            action_digest = hashlib.sha256()
            action_digest.update(str(frame).encode("ascii"))
            for event in result.events:
                action_digest.update(b"|")
                action_digest.update(event.action.encode("ascii"))
                action_digest.update(str(event.token).encode("utf-8"))
            action_digests.append(action_digest.hexdigest())
            terms = []
            for token, reliability in controller.preframe_training_evidence().items():
                labels = prediction_owned_support_targets(prior_bank, token, external_targets, videos_to_epoch)
                plan = support_agreement_plan(labels)
                plan_reasons[plan.reason] += 1
                support_plans += 1
                if plan.reason == "EXTERNAL_FULL_OTHER_GROUP_AGREEMENT":
                    positive_groups += 1
                if online and optimizer_steps < config["online_optimizer_steps"]:
                    terms.extend(loss for _, loss in support_agreement_losses(reliability.logits, plan))
            if terms and online and optimizer_steps < config["online_optimizer_steps"]:
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(member.parameters(), config["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                    raise ValueError("finite nonzero online M gradient required")
                optimizer.step()
                optimizer_steps += 1
                update_frames.append({"epoch": epoch, "frame": int(frame)})
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
            frames += 1
            events += len(result.events)
            consumed += 1
            if consumed >= config["frames_per_video"]:
                break
    after_digest = _parameter_digest(member)
    return {
        "optimizer_steps": optimizer_steps,
        "frames": frames,
        "events": events,
        "actions": dict(sorted(actions.items())),
        "support_plans": support_plans,
        "positive_other_video_groups": positive_groups,
        "plan_reasons": dict(sorted(plan_reasons.items())),
        "event_digests": event_digests,
        "action_digests": action_digests,
        "update_frames": update_frames,
        "losses": losses,
        "gradient_norms": gradients,
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "parameters_changed": before_digest != after_digest,
        "memory_versions": len(controller.memory_versions()),
        "memory_lifecycle_counts": dict(sorted(Counter(v.lifecycle for v in controller.memory_versions()).items())),
    }


def run(config_path: Path = DEFAULT_CONFIG, output_path: Path = DEFAULT_OUTPUT) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing M-gated decision receipt")
    started = time.monotonic()
    _install_input_barrier()
    rcm_config = json.loads((ROOT / config["joint_rcm_config"]).read_text(encoding="utf-8"))
    videos, projection, _ = load_completed_policy_videos(ROOT, rcm_config, lambda: None)
    fit_tracks, _ = load_completed_fit_tracks(ROOT, rcm_config, lambda: None)
    support = support_groups(fit_tracks, -1, 5)
    frozen_evidence = RawPreservingEvidence().eval().requires_grad_(False)
    with torch.inference_mode():
        raw, _, _, _ = reencode_batched_supports(frozen_evidence, support, torch.ones(5), torch.ones(5))
    known = {index: row.numpy().copy() for index, row in enumerate(raw)}
    fixed_videos = videos[:2]
    external_targets = {(video, str(track)): target for (video, track), target in projection.targets.items()}
    # Bind the lawful post-decision projection through the local arm helper.
    # It is deliberately kept outside the numeric controller forward.
    original = _run_arm

    def run_bound_arm(*args, **kwargs):
        # The helper's compact signature is retained; bind the projection by
        # temporarily using the module-level closure only after input loading.
        return _run_arm_with_targets(*args, external_targets=external_targets, **kwargs)

    frozen = run_bound_arm(config, rcm_config, fixed_videos, known, support, online=False)
    online = run_bound_arm(config, rcm_config, fixed_videos, known, support, online=True)
    changed = [left != right for left, right in zip(frozen["event_digests"], online["event_digests"])]
    action_changed = [left != right for left, right in zip(frozen["action_digests"], online["action_digests"])]
    result = {
        "schema_version": "trackocd.cvpr.candidate-family-peak-m-gated-decision-effect-result.v1",
        "status": "PASS_M_GATED_DECISION_EFFECT_CONTROL"
        if online["optimizer_steps"] == config["online_optimizer_steps"]
        else "FAIL_M_GATED_DECISION_EFFECT_CONTROL",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "scope": {
            "videos": len(fixed_videos),
            "frames": frozen["frames"],
            "events": frozen["events"],
            "complete_fixed_clock": False,
            "frames_per_video": config["frames_per_video"],
        },
        "arms": {
            "frozen_m": {key: value for key, value in frozen.items() if key not in ("event_digests", "action_digests")},
            "online_m": {key: value for key, value in online.items() if key not in ("event_digests", "action_digests")},
        },
        "paired_effect": {
            "same_initial_seed_and_scope": True,
            "changed_frame_count": int(sum(changed)),
            "action_changed_frame_count": int(sum(action_changed)),
            "policy_effect_observed": bool(any(changed)),
            "decision_action_effect_observed": bool(any(action_changed)),
            "action_counts_equal": frozen["actions"] == online["actions"],
            "arm_selected": False,
            "frozen_event_digest": hashlib.sha256("".join(frozen["event_digests"]).encode()).hexdigest(),
            "online_event_digest": hashlib.sha256("".join(online["event_digests"]).encode()).hexdigest(),
            "frozen_action_digest": hashlib.sha256("".join(frozen["action_digests"]).encode()).hexdigest(),
            "online_action_digest": hashlib.sha256("".join(online["action_digests"]).encode()).hexdigest(),
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


def _run_arm_with_targets(config, rcm_config, fixed_videos, known, support, *, online, external_targets):
    # This wrapper mirrors _run_arm while keeping the target projection explicit
    # and post-decision.  It is separate so no target can enter controller.ingest.
    torch.manual_seed(int(config.get("seed", 1027)))
    member = MemberReliabilityMLP().train(online)
    controller = _make_controller(rcm_config, known, support, member)
    optimizer = torch.optim.AdamW(member.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) if online else None
    before_digest = _parameter_digest(member)
    videos_to_epoch = {epoch: int(video) for epoch, (video, _arrays) in enumerate(fixed_videos)}
    actions: Counter[str] = Counter(); event_digests = []; action_digests = []
    update_frames = []; losses = []; gradients = []; plan_reasons: Counter[str] = Counter()
    frames = events = support_plans = positive_groups = optimizer_steps = 0
    for epoch, (_video, arrays) in enumerate(fixed_videos):
        if epoch: controller.start_next_video()
        consumed = 0
        for frame, observations in policy_current_frames(arrays):
            prior_bank, _ = controller.state_snapshot()
            result, _ = controller.ingest(frame, observations, track_grad=False)
            actions.update(event.action for event in result.events)
            event_digests.append(_event_digest(frame, result.routing_groups, result.events))
            action_digest = hashlib.sha256(str(frame).encode("ascii"))
            for event in result.events:
                action_digest.update(b"|"); action_digest.update(event.action.encode("ascii")); action_digest.update(str(event.token).encode("utf-8"))
            action_digests.append(action_digest.hexdigest())
            terms = []
            for token, reliability in controller.preframe_training_evidence().items():
                labels = prediction_owned_support_targets(prior_bank, token, external_targets, videos_to_epoch)
                plan = support_agreement_plan(labels)
                plan_reasons[plan.reason] += 1; support_plans += 1
                if plan.reason == "EXTERNAL_FULL_OTHER_GROUP_AGREEMENT": positive_groups += 1
                if online and optimizer_steps < config["online_optimizer_steps"]:
                    terms.extend(loss for _, loss in support_agreement_losses(reliability.logits, plan))
            if terms and online and optimizer_steps < config["online_optimizer_steps"]:
                loss = torch.stack(terms).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(member.parameters(), config["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0: raise ValueError("finite nonzero online M gradient required")
                optimizer.step(); optimizer_steps += 1; update_frames.append({"epoch": epoch, "frame": int(frame)})
                losses.append(float(loss.detach())); gradients.append(float(norm))
            frames += 1; events += len(result.events); consumed += 1
            if consumed >= config["frames_per_video"]: break
    after_digest = _parameter_digest(member)
    return {"optimizer_steps": optimizer_steps, "frames": frames, "events": events, "actions": dict(sorted(actions.items())),
            "support_plans": support_plans, "positive_other_video_groups": positive_groups,
            "plan_reasons": dict(sorted(plan_reasons.items())), "event_digests": event_digests,
            "action_digests": action_digests, "update_frames": update_frames, "losses": losses,
            "gradient_norms": gradients, "parameter_digest_before": before_digest,
            "parameter_digest_after": after_digest, "parameters_changed": before_digest != after_digest,
            "memory_versions": len(controller.memory_versions()),
            "memory_lifecycle_counts": dict(sorted(Counter(v.lifecycle for v in controller.memory_versions()).items()))}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "action_changed_frame_count": result["paired_effect"]["action_changed_frame_count"], "online_optimizer_steps": result["arms"]["online_m"]["optimizer_steps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
