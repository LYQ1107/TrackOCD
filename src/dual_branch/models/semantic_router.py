from __future__ import annotations

import numpy as np
import torch


class SemanticRouter:
    """Semantic branch: frozen-frame trajectory transformer embedding +
    nearest-known-prototype routing and known classification.

    Matches TrackOCD-v1 D1 (Btransformer_b2): routing/known prediction are
    made from the transformer semantic embedding with a proxy-calibrated
    threshold. The classifier head is used during training; at inference the
    router uses the same nearest-prototype rule as the bake-off.
    """

    def __init__(self, model, known_protos, threshold):
        self.model = model
        self.known_protos = known_protos  # dict sem_id -> normalized embedding
        self.threshold = threshold
        if self.model is not None:
            self.model.eval()

    def decide(self, emb):
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        best_id, best_s = None, -1.0
        for cid, p in self.known_protos.items():
            s = float(np.dot(emb, p))
            if s > best_s:
                best_s, best_id = s, cid
        is_known = best_s >= self.threshold
        return is_known, best_id, best_s
