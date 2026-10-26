#!/usr/bin/env python3
"""Run a bounded, label-free candidate-aware lifecycle smoke."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from scripts.trackocd_cvpr import predicted_train_rcm_cross_video_smoke as base
from src.trackocd_v2.io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/candidate_aware_lifecycle_smoke_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_aware_lifecycle_smoke_v1/result.json"


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
    if config.get("schema_version") != "trackocd.cvpr.candidate-aware-lifecycle-smoke.v1":
        raise ValueError("unexpected candidate-aware lifecycle configuration")
    if config.get("status") != "FROZEN_CATEGORY_FREE_CANDIDATE_AWARE_LIFECYCLE_SMOKE_NO_TRAINING":
        raise ValueError("candidate-aware lifecycle smoke is not frozen")
    if config.get("fixed_similarity_reuse_threshold") != 0.55:
        raise ValueError("only the registered fixed similarity margin is allowed")
    if _sha256(ROOT / config["base_input_config"]) != config["base_input_config_sha256"]:
        raise ValueError("base input configuration changed")
    if config["boundaries"] != {
        "labels_or_gt_read": False,
        "category_or_vocabulary_read": False,
        "val_or_test_accessed": False,
        "training_started": False,
        "formal_metrics": False,
        "model_selection": False,
        "harness_forced_cross_video_reuse": False,
        "scientific_acceptance": False,
        "raw_prediction_persisted": False,
    }:
        raise ValueError("candidate-aware lifecycle boundary is not closed")
    return config


class SimilarityLifecycleScorer:
    """Fixed numeric rule; candidate bindings never enter this scorer."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def __call__(self, features, context):
        if features.ndim != 2 or features.shape[1] != 14 or context.shape != (6,):
            raise ValueError("unexpected candidate bridge shape")
        scores = torch.full((len(features) + 2,), -100.0, dtype=torch.float32)
        if len(features):
            existing = torch.where(features[:, 1] < 0.5)[0]
            if len(existing):
                values = features[existing, 0]
                best_position = int(torch.argmax(values))
                if float(values[best_position]) >= self.threshold:
                    scores[existing[best_position]] = 10.0 + values[best_position]
                    return scores
        scores[len(features)] = 10.0
        return scores


def run(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    if output_path.exists():
        raise ValueError("preserve an existing candidate-aware lifecycle receipt")
    started = time.monotonic()
    base.install_input_barrier()
    base_config = json.loads((ROOT / config["base_input_config"]).read_text(encoding="utf-8"))
    loaded = [base.load_video(spec) for spec in base_config["videos"]]
    selections = [
        base.selected_observations(data, config["limits"]["tracks_per_video"], config["limits"]["frames_per_video"])
        for _, data, _ in loaded
    ]
    first_arrays = loaded[0][1]["arrays"]
    first_index = selections[0][1][0]
    first_begin = int(first_arrays["track_offsets"][first_index])
    control, _ = base.build_control(
        first_arrays["visual"][first_begin],
        first_arrays["geometry_normalized_xyxy"][first_begin],
        float(first_arrays["quality"][first_begin]),
    )
    control.scorer = SimilarityLifecycleScorer(config["fixed_similarity_reuse_threshold"])
    event_count = 0
    for video_position, ((_spec, _data, _), selection) in enumerate(zip(loaded, selections)):
        if video_position:
            control.start_next_video()
        for frame_position, observations in enumerate(selection[2]):
            result, _ = control.ingest(frame_position, observations, track_grad=False)
            event_count += len(result.events)
    bank, _ = control.state_snapshot()
    by_token = defaultdict(set)
    for commit in bank.commit_history():
        by_token[commit.token].add(commit.video_epoch)
    cross_video_tokens = [token for token, epochs in by_token.items() if len(epochs) >= 2]
    result = {
        "schema_version": "trackocd.cvpr.candidate-aware-lifecycle-smoke-result.v1",
        "status": "PASS_UNTRAINED_CANDIDATE_AWARE_LIFECYCLE_SMOKE" if cross_video_tokens else "FAIL_NO_NATURAL_CROSS_VIDEO_REUSE",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "lifecycle": {
            "videos": len(loaded),
            "frames": sum(len(selection[2]) for selection in selections),
            "events": event_count,
            "optimizer_steps": 0,
            "memory_versions": len(control.memory_versions()),
            "cross_video_token_count": len(cross_video_tokens),
            "harness_forced_cross_video_reuse": False,
            "append_only_commit_history": True,
        },
        "boundaries": config["boundaries"],
        "resources": {
            "device": "CPU",
            "workers": 1,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "wall_seconds": time.monotonic() - started,
            "training_gpu_hours": 0.0,
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
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
