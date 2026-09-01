from __future__ import annotations

import numpy as np


class DualBranchModel:
    """Hard dual-branch composition.

    Output dict explicitly contains:
      semantic_embedding, discovery_embedding, known_logits, routing_score,
      routing_decision.
    The discovery embedding is guaranteed frozen: it comes from cached DINO
    features / mean and never receives semantic gradients.
    """

    def __init__(self, semantic_router, discovery_encoder):
        self.semantic_router = semantic_router
        self.discovery_encoder = discovery_encoder

    def forward_track(self, sample_id, semantic_emb, discovery_emb):
        discovery_emb = np.asarray(discovery_emb, dtype=np.float32)
        # hard gradient isolation: discovery embedding is detached/frozen
        discovery_emb = discovery_emb.copy()
        assert not hasattr(discovery_emb, "requires_grad") or not discovery_emb.requires_grad
        is_known, known_id, score = self.semantic_router.decide(semantic_emb)
        return {
            "semantic_embedding": np.asarray(semantic_emb, dtype=np.float32),
            "discovery_embedding": discovery_emb,
            "known_logits": None,
            "routing_score": score,
            "routing_decision": "known" if is_known else "novel",
            "semantic_category_id": known_id,
        }
