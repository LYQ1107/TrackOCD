"""ORBIT-BC: birth-controlled causal novel memory evaluation."""
from __future__ import annotations

import numpy as np
import torch

from src.orbit.action_router import KNOWN, EXISTING_NOVEL, NEW_NOVEL
from src.orbit.bi_memory import BiMemory, stats_to_tensor
from src.orbit.evaluate import embed_track


def run_stream_bc(model, rows, features, known_protos, radii, device,
                  birth_threshold=0.0, novel_update_rate=0.2):
    mem = BiMemory(known_protos, radii, novel_update_rate=novel_update_rate)
    preds = []
    for i, r in enumerate(rows):
        z, rel = embed_track(model, features[r["sample_id"]], device)
        kid, ks = mem.known_id(z)
        nid, ns = mem.existing_novel(z)
        stats = mem.stats(z, rel, len(features[r["sample_id"]]), known_id=kid, novel_id=nid)
        st = stats_to_tensor(stats, device=device)
        with torch.no_grad():
            logits = model.action_net(st)
        action = int(logits.argmax(dim=1).item())
        if action == NEW_NOVEL and nid is not None and ns >= birth_threshold:
            action = EXISTING_NOVEL
        if action == KNOWN and kid is not None:
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "known", "semantic_category_id": int(kid),
            })
        elif action == EXISTING_NOVEL and nid is not None:
            mem.update_novel(nid, z)
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": int(nid),
            })
        else:
            vid = mem.create_novel(z)
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": int(vid),
            })
    return preds, mem
