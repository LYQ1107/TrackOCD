from __future__ import annotations

import numpy as np


class DiscoveryEncoder:
    """Frozen discovery branch: DINOv2 frame features -> L2 -> track mean ->
    L2. No trainable parameters, no gradients, no semantic information."""

    name = "dino_track_mean"

    def __init__(self, frame_features=None, mean_features=None):
        self.frame_features = frame_features
        self.mean_features = mean_features

    def __call__(self, sample_id):
        if self.mean_features is not None and sample_id in self.mean_features:
            v = np.asarray(self.mean_features[sample_id], dtype=np.float32)
        else:
            frames = np.asarray(self.frame_features[sample_id], dtype=np.float32)
            v = frames.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_track_mean(self, frames):
        frames = np.asarray(frames, dtype=np.float32)
        v = frames.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)
