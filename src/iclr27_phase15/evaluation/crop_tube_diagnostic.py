"""One bounded frame/tube diagnostic for Phase15 Branch D.

This is an offline localization check only.  It does not train, alter the
DSCT stream, or authorize a Q1 replay.  The max-frame-pair score intentionally
uses the complete public track tube and is therefore marked non-causal; it is
not a candidate TrackOCD method.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.iclr27_phase15.representation.phase15a_probe import (
    interval,
    retrieval_from_matrix,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    z = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz",
                allow_pickle=False)
    manifest = json.loads((ROOT /
                           "outputs/iclr27_phase15/manifests/phase15_preregistration.json").read_text())
    ids = np.asarray(manifest["split"]["meta_validation"]["track_indices"], dtype=np.int64)
    labels = z["labels"][ids].astype(np.int64)
    videos = z["video_ids"][ids].astype(np.int64)
    ff = z["frame_feats"][ids].astype(np.float32)
    fm = z["frame_mask"][ids].astype(bool)
    frames = []
    for i in range(len(ids)):
        x = ff[i][fm[i]]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        frames.append(x)
    mean = np.asarray([x.mean(axis=0) for x in frames], dtype=np.float32)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    last = np.asarray([x[-1] for x in frames], dtype=np.float32)
    last /= np.maximum(np.linalg.norm(last, axis=1, keepdims=True), 1e-12)
    n = len(ids)
    tube = np.full((n, n), -1e9, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j and int(videos[i]) != int(videos[j]):
                tube[i, j] = float(np.max(frames[i] @ frames[j].T))
    mean_m = mean @ mean.T
    last_m = last @ last.T
    metrics = {
        "mean_prefix8": retrieval_from_matrix(mean, labels, videos, mean_m),
        "last_frame": retrieval_from_matrix(last, labels, videos, last_m),
        "tube_max_frame_pair": retrieval_from_matrix(mean, labels, videos, tube),
    }
    raw = metrics["mean_prefix8"]
    tube_m = metrics["tube_max_frame_pair"]
    payload = {
        "protocol": "phase15d_crop_tube_diagnostic",
        "source": "outputs/iclr27_phase6d/assets/full_tao_tracks.npz",
        "meta_tracks": int(n), "metrics": metrics,
        "tube_gain_vs_mean_prefix8": {
            "r1": (tube_m["r1"] or 0.0) - (raw["r1"] or 0.0),
            "map": (tube_m["map"] or 0.0) - (raw["map"] or 0.0),
        },
        "future_frames_used_for_diagnostic": True,
        "q1_label_used": False, "devplus_used": False,
        "physical_id_used_as_feature": False,
        "causal_online_gate_evaluated": False,
        "same_frozen_audit_improvement": False,
        "decision": "stop_after_single_bounded_diagnostic",
    }
    atomic(ROOT / "outputs/iclr27_phase15/eval/phase15d_crop_tube_diagnostic.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
