#!/usr/bin/env python3
"""Run a fixed no-training R/C/M interface smoke on one predicted Train video.

The scorer used here is an explicit deterministic harness: it drives NEW on
the first frame and EXISTING afterwards only to exercise the real R/C/M tensor
and versioned-memory path.  It is not a learned scorer, benchmark, or metric.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, sha256_file  # noqa: E402


CONFIG = ROOT / "configs/trackocd_cvpr/predicted_train_rcm_interface_smoke_v1.json"


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def install_input_barrier() -> None:
    def audit(event: str, args: Tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError(f"R/C/M smoke forbids external activity: {event}")
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
                raise PermissionError(f"R/C/M smoke forbids prohibited input: {path}")

    sys.addaudithook(audit)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_rows(cfg: Dict[str, Any], arrays: Dict[str, np.ndarray], rows: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    selected = [int(value) for value in cfg["fixed_selection"]["track_indices"]]
    frame_ids = [int(value) for value in cfg["fixed_selection"]["frame_ids"]]
    if selected != [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]:
        raise ValueError("Fixed R/C/M smoke selection changed")
    if len(rows) != 1016 or arrays["track_offsets"].shape != (1017,):
        raise ValueError("Pilot input index universe changed")
    if arrays["visual"].shape != (29670, 768):
        raise ValueError("Pilot feature universe changed")
    for index in selected:
        row = rows[index]
        if int(row["video_id"]) != 3 or row["source_split"] != "train_predicted":
            raise ValueError("Non-Train or wrong fixed video row")
        if not set(frame_ids).issubset(set(int(value) for value in row["frame_ids"])):
            raise ValueError("Fixed track lacks the eight registered frames")
    return selected, frame_ids


def make_observations(cfg: Dict[str, Any], arrays: Dict[str, np.ndarray], rows: List[Dict[str, Any]], selected: List[int], frame_ids: List[int]):
    from src.trackocd_cvpr.causal_observation_router import CurrentObservation

    frames = arrays["frame_ids"]
    offsets = arrays["track_offsets"]
    visual = arrays["visual"]
    boxes = arrays["geometry_normalized_xyxy"]
    quality = arrays["quality"]
    by_frame = []
    for frame_id in frame_ids:
        current = []
        for index in selected:
            begin, end = int(offsets[index]), int(offsets[index + 1])
            positions = np.flatnonzero(frames[begin:end] == frame_id)
            if len(positions) != 1:
                raise ValueError("Fixed track/frame mapping is not one-to-one")
            position = begin + int(positions[0])
            box = np.asarray(boxes[position], dtype=np.float32)
            value = np.asarray(visual[position], dtype=np.float32)
            score = float(quality[position])
            if value.shape != (768,) or box.shape != (4,) or not np.isfinite(value).all() or not np.isfinite(box).all():
                raise ValueError("Non-finite predicted numeric input")
            if np.any(box[2:] <= box[:2]) or np.any(box < 0) or np.any(box > 1) or not 0 <= score <= 1:
                raise ValueError("Invalid normalized predicted geometry/quality")
            current.append(CurrentObservation(f"anon-{index:04d}", value, box, score))
        by_frame.append(tuple(current))
    return by_frame


class DeterministicSmokeScorer:
    """Explicit non-learned lifecycle driver, still accepted by the C contract."""

    def __init__(self, scorer, first_frame: bool = True):
        self.scorer = scorer
        self.first_frame = first_frame

    def set_phase(self, first_frame: bool) -> None:
        self.first_frame = bool(first_frame)

    def __call__(self, features, context):
        import torch

        if features.ndim != 2 or context.shape != (6,):
            raise ValueError("Smoke scorer received wrong C tensor shape")
        scores = torch.full((len(features) + 2,), -10.0, dtype=torch.float32)
        if self.first_frame:
            scores[len(features)] = 10.0  # NEW
        elif len(features) > 1:
            scores[1] = 10.0  # first EXISTING group after the one anchor
        else:
            scores[len(features)] = 10.0
        # The production R/C/M bridge keeps the scorer graph type even when
        # this harness disables gradients; return a Torch tensor, not a
        # detached NumPy callback result.
        return scores


def new_control(anchor_visual, anchor_box, anchor_quality):
    import torch

    from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix
    from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl
    from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP

    torch.manual_seed(1027)
    evidence = RawPreservingEvidence().eval()
    learned_scorer = CandidateScoreMLP(feature_count=14).eval()
    scorer = DeterministicSmokeScorer(learned_scorer)
    member = MemberReliabilityMLP().eval()
    anchor = VisiblePrefix(
        torch.from_numpy(anchor_visual[None].astype(np.float32)),
        torch.from_numpy(anchor_box[None].astype(np.float32)),
        torch.from_numpy(np.asarray([anchor_quality], dtype=np.float32)),
        torch.tensor([0], dtype=torch.int64),
    )
    with torch.inference_mode():
        prototype = evidence(anchor.visual, anchor.boxes, anchor.quality, anchor.elapsed).raw.numpy()
    control = JointRCMAtomicMemberControl(
        evidence,
        learned_scorer,
        member,
        {0: prototype},
        ((anchor,),),
        quarantine=True,
        K=8,
        bank_limits={"max_members": 64, "max_tokens": 128, "max_commits": 1024, "max_payload_bytes": 1 << 20},
    )
    # The production class owns the scorer call.  Replace only its numeric
    # callback for this explicit harness; R and M remain the real modules.
    control.scorer = scorer
    return control, scorer


def run_sequence(by_frame, selected, arrays, frame_count: int, route_prefix: str):
    import torch

    from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl

    offsets = arrays["track_offsets"]
    visual = arrays["visual"]
    boxes = arrays["geometry_normalized_xyxy"]
    quality = arrays["quality"]
    anchor_position = int(offsets[selected[0]])
    control, scorer = new_control(visual[anchor_position], boxes[anchor_position], float(quality[anchor_position]))
    frame_records = []
    with torch.inference_mode():
        for frame_index, observations in enumerate(by_frame[:frame_count]):
            if route_prefix != "anon-":
                observations = tuple(
                    type(item)(route_prefix + item.routing_id.removeprefix("anon-"), item.visual, item.box_normalized_xyxy, item.quality)
                    for item in observations
                )
            scorer.set_phase(frame_index == 0)
            result, _ = control.ingest(frame_index, observations, track_grad=False)
            frame_records.append([
                {
                    "visible_input_sha256": event.visible_input_sha256,
                    "action": event.action,
                    "ambiguity_reason": event.ambiguity_reason,
                }
                for event in result.events
            ])
    return control, frame_records


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result_path = ROOT / cfg["result_path"]
    if result_path.exists():
        raise RuntimeError("R/C/M smoke result already exists; refusing overwrite")
    payload_path = ROOT / cfg["input_payload"]
    index_path = ROOT / cfg["input_index"]
    if sha256_file(payload_path) != cfg["input_payload_sha256"] or sha256_file(index_path) != cfg["input_index_sha256"]:
        raise ValueError("Pilot input hash mismatch")
    rows = load_jsonl(index_path)
    install_input_barrier()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with np.load(payload_path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=False) for key in loaded.files}
    selected, frame_ids = selected_rows(cfg, arrays, rows)
    by_frame = make_observations(cfg, arrays, rows, selected, frame_ids)
    if sum(len(item) for item in by_frame) != int(cfg["fixed_selection"]["observations"]):
        raise ValueError("Fixed smoke observation count changed")
    full_control, full_records = run_sequence(by_frame, selected, arrays, 8, "anon-")
    renamed_control, renamed_records = run_sequence(by_frame, selected, arrays, 8, "renamed-")
    prefix_control, prefix_records = run_sequence(by_frame, selected, arrays, 4, "anon-")
    if full_records != renamed_records:
        raise AssertionError("Opaque routing-ID rename changed numeric R/C/M outputs")
    if full_records[:4] != prefix_records:
        raise AssertionError("Future suffix changed the first four R/C/M frames")
    actions = Counter(item["action"] for frame in full_records for item in frame)
    if actions["NEW"] == 0 or actions["EXISTING"] == 0:
        raise AssertionError("Smoke harness did not exercise NEW and EXISTING lifecycle")
    if len(full_control.memory_versions()) == 0:
        raise AssertionError("R/C/M memory did not retain a version")
    result = {
        "schema_version": "trackocd.cvpr.predicted-train-rcm-interface-smoke-result.v1",
        "status": "PASS_FIXED_PREDICTED_TRAIN_RCM_INTERFACE_SMOKE_NO_TRAINING",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": {
            "payload": str(payload_path.relative_to(ROOT)),
            "payload_sha256": cfg["input_payload_sha256"],
            "index": str(index_path.relative_to(ROOT)),
            "index_sha256": cfg["input_index_sha256"],
            "video_id": 3,
            "tracks": len(selected),
            "frames": len(frame_ids),
            "observations": sum(len(item) for item in by_frame),
            "events": sum(len(item) for item in full_records),
        },
        "interface": {
            "R": "RawPreservingEvidence",
            "C": "CandidateScoreMLP tensor shape and candidate bridge; deterministic harness callback",
            "M": "JointRCMAtomicMemberControl versioned reliability memory",
            "actions": dict(sorted(actions.items())),
            "memory_versions": len(full_control.memory_versions()),
            "first_frame_records": len(full_records[0]),
            "last_frame_records": len(full_records[-1]),
            "opaque_route_rename_invariant": True,
            "future_suffix_prefix_invariant": True,
            "learned_scorer_optimized": False,
            "synthetic_numeric_anchor_not_semantic_label": True,
        },
        "resources": {
            "device": "CPU",
            "workers": 1,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "optimizer_steps": 0,
            "training_gpu_hours": 0.0,
        },
        "boundaries": cfg["boundaries"],
        "interpretation": "Interface smoke only; it validates causal R/C/M tensor and append-only-memory plumbing on category-free predicted features. It is not trained, not a qualified frontend, and reports no OCD/MOT or discovery metric.",
    }
    atomic_json(result_path, result)
    print(json.dumps({"status": result["status"], "actions": result["interface"]["actions"], "memory_versions": result["interface"]["memory_versions"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
