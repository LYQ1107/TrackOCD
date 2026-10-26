#!/usr/bin/env python3
"""Registered bounded family-peak residual joint R/C/M Train pilot.

The family-relative route is frozen before any external loss is read.  A tiny
numeric residual is kept in the scorer so the candidate loss still has a real
C gradient; the residual is deliberately too small to replace the registered
family-peak action during this interface pilot.  This is not a Pred endpoint,
development evaluation, or Goal A/B/C acceptance run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.candidate_aware_rcm_support_smoke import _install_input_barrier
from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
from src.trackocd_cvpr.joint_rcm_policy_training import train_joint_rcm_policy_pilot
from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_v2.io import atomic_json, sha256_file


CONFIG = "configs/trackocd_cvpr/family_peak_joint_rcm_training_pilot_v1.json"
SOURCES = (
    CONFIG,
    "scripts/trackocd_cvpr/pilot_family_peak_joint_rcm.py",
    "src/trackocd_cvpr/joint_rcm_policy_training.py",
    "src/trackocd_cvpr/joint_rc_policy_training.py",
    "src/trackocd_cvpr/raw_preserving_evidence.py",
    "src/trackocd_cvpr/candidate_score_contract.py",
    "src/trackocd_cvpr/differentiable_candidate_features.py",
    "src/trackocd_cvpr/differentiable_support_bank.py",
    "src/trackocd_cvpr/atomic_member_control.py",
    "src/trackocd_cvpr/causal_observation_router.py",
    "src/trackocd_cvpr/prefix_member_bank.py",
    "src/trackocd_cvpr/candidate_training_loss.py",
    "src/trackocd_cvpr/policy_training_data.py",
    "src/trackocd_cvpr/fit_R_training.py",
    "src/trackocd_cvpr/fit_loss_supervision.py",
    "src/trackocd_cvpr/GT_video_cache.py",
    "src/trackocd_cvpr/member_reliability.py",
    "src/trackocd_cvpr/reliability_training_loss.py",
    "src/trackocd_cvpr/reliability_support_bridge.py",
    "src/trackocd_cvpr/committed_support_bridge.py",
    "src/trackocd_cvpr/versioned_reliability_memory.py",
    "src/trackocd_cvpr/version_lineage_audit.py",
    "src/trackocd_cvpr/scoped_train_census.py",
    "src/trackocd_cvpr/remote_registration.py",
    "src/trackocd_v2/io.py",
    "scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py",
    "configs/trackocd_cvpr/formal_research_scope_approval_v1.json",
    "configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json",
    "configs/trackocd_cvpr/autonomous_research.json",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _digest_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


class FamilyPeakResidualScorer(torch.nn.Module):
    """Fixed family-peak action plus a differentiable numeric residual.

    The route uses only candidate features and query context.  The residual is
    scaled by a registered constant and therefore cannot replace the fixed
    family-peak baseline within this bounded pilot.  It is retained in the
    logits, rather than detached, so the lawful external C loss reaches the
    scorer and then the R evidence path.
    """

    def __init__(self, residual_scale: float = 1e-3):
        super().__init__()
        if residual_scale != 1e-3:
            raise ValueError("Only the registered residual scale is allowed")
        self.residual_scale = residual_scale
        self.residual = CandidateScoreMLP(feature_count=14)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 14 or context.shape != (6,):
            raise ValueError("unexpected family-peak residual bridge shape")
        if features.dtype != torch.float32 or context.dtype != torch.float32:
            raise ValueError("registered float32 family-peak residual bridge required")
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError("finite family-peak residual bridge required")
        learned = self.residual(features, context)
        scores = self.residual_scale * learned - 100.0
        existing = torch.where(features[:, 1] < 0.5)[0]
        known = torch.where(features[:, 1] >= 0.5)[0]

        def peak(indices: torch.Tensor):
            if not len(indices):
                return features.new_tensor(float("-inf")), None
            values = features[indices, 0]
            standardized = (values - values.mean()) / (values.std(unbiased=False) + torch.finfo(values.dtype).eps)
            position = int(torch.argmax(standardized).detach())
            return standardized[position].detach(), int(indices[position].detach())

        existing_peak, existing_index = peak(existing)
        known_peak, _ = peak(known)
        if existing_index is not None and float(existing_peak) >= float(known_peak):
            baseline = features.new_tensor(110.0 + float(existing_peak))
            selector = torch.nn.functional.one_hot(
                torch.tensor(existing_index, device=features.device), len(scores)
            ).to(features.dtype)
        else:
            baseline = features.new_tensor(110.0)
            selector = torch.nn.functional.one_hot(
                torch.tensor(len(features), device=features.device), len(scores)
            ).to(features.dtype)
        return scores + selector * baseline


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.family-peak-residual-joint-rcm-pilot.v1":
        raise ValueError("unexpected family-peak residual joint configuration")
    if config.get("status") != "REGISTERED_FAMILY_PEAK_RESIDUAL_JOINT_RCM_PILOT_NOT_SCIENTIFIC_ACCEPTANCE":
        raise ValueError("family-peak residual joint pilot is not frozen")
    if config.get("pilot_policy_relative_ordinals") != [0, 1]:
        raise ValueError("fixed two-video policy scope required")
    if config.get("updates_per_episode") != 2 or config.get("episodes") != 1:
        raise ValueError("exact two-update one-episode scope required")
    if config.get("scorer") != {
        "family_rule": "within_family_population_standardized_adapted_similarity",
        "tie_break": "EXISTING",
        "residual_scale": 1e-3,
        "residual_parameters": 804,
    }:
        raise ValueError("registered family-peak residual scorer changed")
    if config.get("limits") != {
        "optimizer_steps": 2,
        "training_gpu_hours": 0,
        "device": "CPU",
        "workers": 1,
    }:
        raise ValueError("family-peak residual limits are not closed")
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
        raise ValueError("family-peak residual boundary is not closed")
    base = ROOT / config["joint_rcm_config"]
    if sha256_file(base) != config["joint_rcm_config_sha256"]:
        raise ValueError("registered joint RCM configuration changed")
    return config


def _source_gate(config: dict, preregistration_commit: str, git_proxy: str) -> dict:
    proof = verify_registration(
        ROOT, config["branch"], preregistration_commit, git_proxy, allow_public_api=True
    )
    for path in SOURCES:
        registered = __import__("subprocess").check_output(
            ["git", "show", preregistration_commit + ":" + path], cwd=ROOT
        )
        if registered != (ROOT / path).read_bytes():
            raise ValueError("Registered source differs: " + path)
    return proof


def _install_restricted_barrier(output: Path) -> None:
    _install_input_barrier()

    def barrier(event: str, args: tuple) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix in (".pt", ".pth") and path.parent != output:
                raise PermissionError("joint family-peak pilot forbids external checkpoints")
            forbidden = ("/TAO-Amodal/", "/frames/val/", "/frames/test/", "/annotations/")
            if any(token in str(path) for token in forbidden):
                raise PermissionError("joint family-peak pilot forbids raw Val/Test/annotation input")

    sys.addaudithook(barrier)


def run(config_path: Path, output_path: Path, preregistration_commit: str, git_proxy: str) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing family-peak joint receipt")
    output_path.mkdir(parents=True)
    started = time.monotonic()
    proof = _source_gate(config, preregistration_commit, git_proxy)
    _install_restricted_barrier(output_path)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base_config = json.loads((ROOT / config["joint_rcm_config"]).read_text(encoding="utf-8"))
    videos, projection, policy_proof = load_completed_policy_videos(ROOT, base_config, lambda: None)
    fit_tracks, fit_proof = load_completed_fit_tracks(ROOT, base_config, lambda: None)
    known_supports = support_groups(fit_tracks, -1, 5)
    frozen_evidence = RawPreservingEvidence().eval().requires_grad_(False)
    with torch.inference_mode():
        raw, _, _, _ = reencode_batched_supports(
            frozen_evidence, known_supports, torch.ones(5), torch.ones(5)
        )
    known = {index: row.numpy().copy() for index, row in enumerate(raw)}
    indices = config["pilot_policy_relative_ordinals"]
    pilot_videos = [videos[index] for index in indices]
    if len(pilot_videos) != 2:
        raise ValueError("fixed two-video pilot payload missing")

    peak_before = _digest_model(FamilyPeakResidualScorer())

    def guard() -> None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if rss > config["rss_bytes_ceiling"]:
            raise RuntimeError("family-peak joint RSS ceiling exceeded")
        bytes_used = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
        if bytes_used > config["output_bytes_ceiling"]:
            raise RuntimeError("family-peak joint output ceiling exceeded")

    def save_checkpoint(evidence, scorer, member, optimizer, step, episode, episode_step, rng_state):
        path = output_path / ("private_family_peak_joint_step" + str(step) + "_checkpoint.pt")
        temporary = output_path / (path.name + ".partial")
        with temporary.open("xb") as handle:
            torch.save({
                "evidence_state": evidence.state_dict(),
                "scorer_state": scorer.state_dict(),
                "member_state": member.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "episode": episode,
                "episode_step": episode_step,
                "numpy_rng_state": rng_state,
            }, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def progress(row, curve):
        atomic_json(output_path / "progress.json", row)
        atomic_json(output_path / "private_loss_curve.json", curve)
        guard()

    evidence, scorer, member, optimizer, summary = train_joint_rcm_policy_pilot(
        pilot_videos, projection, known, known_supports, base_config, guard, progress,
        save_checkpoint, scorer_factory=FamilyPeakResidualScorer,
    )
    guard()
    endpoint = output_path / "private_family_peak_joint_step2_checkpoint.pt"
    if not endpoint.is_file():
        raise ValueError("fixed family-peak joint endpoint missing")
    reopened = torch.load(endpoint, map_location="cpu", weights_only=True)
    evidence_check = RawPreservingEvidence()
    scorer_check = FamilyPeakResidualScorer()
    from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
    member_check = MemberReliabilityMLP()
    evidence_check.load_state_dict(reopened["evidence_state"], strict=True)
    scorer_check.load_state_dict(reopened["scorer_state"], strict=True)
    member_check.load_state_dict(reopened["member_state"], strict=True)
    if (_digest_model(evidence_check) != _digest_model(evidence)
            or _digest_model(scorer_check) != _digest_model(scorer)
            or _digest_model(member_check) != _digest_model(member)):
        raise ValueError("family-peak joint endpoint did not reopen exactly")
    result = {
        "status": "COMPLETED_FAMILY_PEAK_RESIDUAL_JOINT_RCM_PILOT_NOT_DEV_PRED_OR_GOAL_ACCEPTANCE",
        "generated_utc": _now(),
        "live_remote_registration_proof": proof,
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "policy_scope": {key: value for key, value in policy_proof.items()
                          if key not in ("private_label_bindings",)},
        "offline_fit_prior_scope": fit_proof,
        "pilot_scope": {
            "policy_relative_ordinals": indices,
            "videos": len(pilot_videos),
            "selection_frozen_before_private_payload_access": True,
            "complete_fixed_clock_consumed": True,
            "not_development_or_final": True,
        },
        "scorer": {
            "family_rule": config["scorer"]["family_rule"],
            "residual_scale": config["scorer"]["residual_scale"],
            "residual_parameters": config["scorer"]["residual_parameters"],
            "initial_digest": peak_before,
            "final_digest": _digest_model(scorer),
            "parameters_changed": peak_before != _digest_model(scorer),
        },
        "summary": summary,
        "joint_parameter_counts": {
            "evidence": sum(p.numel() for p in evidence.parameters()),
            "scorer": sum(p.numel() for p in scorer.parameters()),
            "member": sum(p.numel() for p in member.parameters()),
        },
        "private_endpoint": {
            "bytes": endpoint.stat().st_size,
            "sha256": sha256_file(endpoint),
            "content_uploaded": False,
            "reopened_exact": True,
        },
        "boundaries": config["boundaries"],
        "resources": {
            "device": "CPU",
            "workers": 1,
            "training_GPU_hours": 0.0,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "wall_seconds": time.monotonic() - started,
            "CUDA_initialized": torch.cuda.is_initialized(),
        },
        "interpretation": config["interpretation"],
    }
    atomic_json(output_path / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--git-proxy", default="")
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    output = args.output.resolve() if args.output else ROOT / config["output_directory"]
    result = run(args.config.resolve(), output, args.preregistration_commit, args.git_proxy)
    print(json.dumps({
        "status": result["status"],
        "optimizer_steps": result["summary"]["optimizer_steps"],
        "metadata_frames": result["summary"]["episodes"][0]["metadata_frames"],
        "M_loss_rows": sum(row["M_loss"] is not None for row in result["summary"]["loss_curve"]),
        "C_loss_rows": sum(row["C_loss"] is not None for row in result["summary"]["loss_curve"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
