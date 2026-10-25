#!/usr/bin/env python3
"""Category-free cross-video lifecycle smoke for the predicted R/C/M path."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json  # noqa: E402


CONFIG = ROOT / "configs/trackocd_cvpr/predicted_train_rcm_cross_video_smoke_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def install_input_barrier() -> None:
    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError(f"cross-video smoke forbids external activity: {event}")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            forbidden = (
                "/TAO-Amodal/",
                "/data/raw/tao/",
                "/frames/val/",
                "/frames/test/",
                "/roles.json",
                "/train_labels.parquet",
                "/evaluator_only_geometry_join.json",
                ".pth",
                ".pt",
            )
            if any(token in path for token in forbidden):
                raise PermissionError(f"cross-video smoke forbids prohibited input: {path}")

    sys.addaudithook(audit)


def load_video(spec: dict) -> tuple[dict, dict, np.ndarray]:
    payload = ROOT / spec["payload"]
    index_path = ROOT / spec["index"]
    if sha256_file(payload) != spec["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {payload}")
    if sha256_file(index_path) != spec["index_sha256"]:
        raise ValueError(f"index hash mismatch: {index_path}")
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(payload, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=False) for key in loaded.files}
    required = {"visual", "geometry_normalized_xyxy", "quality", "frame_ids", "track_offsets"}
    if not required <= set(arrays):
        raise ValueError("predicted feature payload missing required arrays")
    if len(rows) + 1 != arrays["track_offsets"].shape[0]:
        raise ValueError("track index/offset count mismatch")
    video_id = int(spec["video_id"])
    if len(rows) < 4 or any(int(row["video_id"]) != video_id for row in rows):
        raise ValueError("fixed category-free video index mismatch")
    return spec, {"rows": rows, "arrays": arrays}, np.asarray([], dtype=np.int64)


def selected_observations(data: dict, tracks_per_video: int, frames_per_video: int):
    rows = data["rows"]
    arrays = data["arrays"]
    all_frames = sorted({int(frame) for row in rows for frame in row["frame_ids"]})
    frame_ids = tuple(all_frames[:frames_per_video])
    selected = [
        index for index, row in enumerate(rows)
        if all(frame in set(int(value) for value in row["frame_ids"]) for frame in frame_ids)
    ][:tracks_per_video]
    if len(selected) != tracks_per_video or len(frame_ids) != frames_per_video:
        raise ValueError("fixed stream-order cross-video selection is not available")
    from src.trackocd_cvpr.causal_observation_router import CurrentObservation

    observations = []
    frame_array = arrays["frame_ids"]
    offsets = arrays["track_offsets"]
    for frame in frame_ids:
        current = []
        for index in selected:
            begin, end = int(offsets[index]), int(offsets[index + 1])
            positions = np.flatnonzero(frame_array[begin:end] == frame)
            if len(positions) != 1:
                raise ValueError("fixed track/frame mapping is not one-to-one")
            position = begin + int(positions[0])
            visual = np.asarray(arrays["visual"][position], dtype=np.float32)
            box = np.asarray(arrays["geometry_normalized_xyxy"][position], dtype=np.float32)
            quality = float(arrays["quality"][position])
            if visual.shape != (768,) or box.shape != (4,) or not np.isfinite(visual).all() or not np.isfinite(box).all():
                raise ValueError("non-finite predicted numeric input")
            if np.any(box[2:] <= box[:2]) or np.any(box < 0) or np.any(box > 1) or not 0 <= quality <= 1:
                raise ValueError("invalid predicted geometry/quality")
            current.append(CurrentObservation(f"anon-{index:04d}", visual, box, quality))
        observations.append(tuple(current))
    return tuple(frame_ids), tuple(selected), tuple(observations)


class LifecycleHarnessScorer:
    """Non-learned scorer used only to force a cross-video plumbing path."""

    def __init__(self) -> None:
        self.force_new = True

    def set_phase(self, *, force_new: bool) -> None:
        self.force_new = bool(force_new)

    def __call__(self, features, context):
        import torch

        if features.ndim != 2 or context.shape != (6,):
            raise ValueError("unexpected candidate bridge shape")
        scores = torch.full((len(features) + 2,), -10.0, dtype=torch.float32)
        if self.force_new or len(features) <= 1:
            scores[len(features)] = 10.0
        else:
            scores[1] = 10.0
        return scores


def build_control(anchor_visual: np.ndarray, anchor_box: np.ndarray, anchor_quality: float):
    import torch

    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
    from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix
    from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl
    from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence

    torch.manual_seed(1027)
    evidence = RawPreservingEvidence().eval()
    scorer_model = CandidateScoreMLP(feature_count=14).eval()
    member = MemberReliabilityMLP().eval()
    anchor = VisiblePrefix(
        torch.from_numpy(anchor_visual[None].astype(np.float32)),
        torch.from_numpy(anchor_box[None].astype(np.float32)),
        torch.tensor([anchor_quality], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int64),
    )
    with torch.inference_mode():
        prototype = evidence(anchor.visual, anchor.boxes, anchor.quality, anchor.elapsed).raw.numpy()
    control = JointRCMAtomicMemberControl(
        evidence,
        scorer_model,
        member,
        {0: prototype},
        ((anchor,),),
        quarantine=True,
        K=8,
        bank_limits={"max_members": 64, "max_tokens": 128, "max_commits": 1024, "max_payload_bytes": 1 << 20},
    )
    scorer = LifecycleHarnessScorer()
    control.scorer = scorer
    return control, scorer


def main() -> int:
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["output"]
    if output.exists():
        raise RuntimeError("cross-video smoke output already exists; refusing overwrite")
    install_input_barrier()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    loaded = []
    for spec in cfg["videos"]:
        loaded.append(load_video(spec))
    selections = [selected_observations(data, cfg["selection"]["tracks_per_video"], cfg["selection"]["frames_per_video"]) for _, data, _ in loaded]
    first_arrays = loaded[0][1]["arrays"]
    first_index = selections[0][1][0]
    first_begin = int(first_arrays["track_offsets"][first_index])
    control, scorer = build_control(
        first_arrays["visual"][first_begin],
        first_arrays["geometry_normalized_xyxy"][first_begin],
        float(first_arrays["quality"][first_begin]),
    )
    event_count = 0
    frame_records = []
    for video_position, ((spec, data, _), selection) in enumerate(zip(loaded, selections)):
        if video_position:
            control.start_next_video()
        _, _, frames = selection
        for frame_position, observations in enumerate(frames):
            scorer.set_phase(force_new=(video_position == 0 and frame_position == 0))
            result, _ = control.ingest(frame_position, observations, track_grad=False)
            event_count += len(result.events)
            frame_records.append({"video_id": int(spec["video_id"]), "frame_position": frame_position, "events": len(result.events)})
    bank, _ = control.state_snapshot()
    by_token = defaultdict(set)
    for commit in bank.commit_history():
        by_token[commit.token].add(commit.video_epoch)
    cross_video_tokens = sorted(token for token, epochs in by_token.items() if len(epochs) >= 2)
    if not cross_video_tokens:
        raise AssertionError("lifecycle harness did not create a cross-video token")
    supports = bank.raw_support_snapshot()
    support_payload_counts = {support.token: len(support.payloads) for support in supports}
    result = {
        "schema_version": "trackocd.cvpr.predicted-train-rcm-cross-video-smoke-result.v1",
        "status": "PASS_CATEGORY_FREE_CROSS_VIDEO_RCM_LIFECYCLE_SMOKE_NO_TRAINING",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "selection": [
            {"video_id": int(spec["video_id"]), "frame_ids": list(selection[0]), "track_indices": list(selection[1])}
            for (spec, _, _), selection in zip(loaded, selections)
        ],
        "lifecycle": {
            "videos": len(loaded),
            "frames": sum(len(selection[2]) for selection in selections),
            "events": event_count,
            "optimizer_steps": 0,
            "memory_versions": len(control.memory_versions()),
            "cross_video_tokens": cross_video_tokens,
            "cross_video_token_count": len(cross_video_tokens),
            "support_payload_counts": support_payload_counts,
            "append_only_commit_history": True,
            "harness_forced_cross_video_reuse": True,
        },
        "boundaries": {
            "labels_or_GT_read": False,
            "category_or_vocabulary_read": False,
            "future_suffix_used": False,
            "val_or_test_accessed": False,
            "training_started": False,
            "formal_metrics": False,
            "model_selection": False,
            "scientific_acceptance": False,
        },
        "resources": {
            "device": "CPU",
            "workers": 1,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "training_gpu_hours": 0.0,
        },
        "interpretation": (
            "This proves only that the anonymous append-only R/C/M lifecycle can retain a support token across video epochs when a non-learned harness forces reuse. "
            "It does not provide M supervision, lawful predicted-track training, OCD/MOT metrics, or model-selection evidence."
        ),
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "cross_video_token_count": len(cross_video_tokens)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
