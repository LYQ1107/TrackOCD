"""Batch ORBIT-D1 inference for large predicted-track sets."""
from __future__ import annotations

import numpy as np
import torch

from src.orbit.action_router import KNOWN, EXISTING_NOVEL, NEW_NOVEL
from src.orbit.bi_memory import BiMemory
from src.orbit.evaluate import load_model


def embed_batch(model, features, sids, device, batch=512):
    max_t = max(len(features[s]) for s in sids)
    zs = np.zeros((len(sids), 768), dtype=np.float32)
    rels = np.zeros(len(sids), dtype=np.float32)
    for start in range(0, len(sids), batch):
        chunk = sids[start:start + batch]
        x = np.zeros((len(chunk), max_t, 768), dtype=np.float32)
        m = np.zeros((len(chunk), max_t), dtype=bool)
        for i, s in enumerate(chunk):
            f = features[s]
            x[i, :len(f)] = f
            m[i, :len(f)] = True
        xt = torch.as_tensor(x, device=device)
        mt = torch.as_tensor(m, device=device)
        with torch.no_grad():
            out = model.aggregate(xt, mt)
        zs[start:start + len(chunk)] = out["z"].cpu().numpy()
        rels[start:start + len(chunk)] = out["cos"].mean(dim=1).cpu().numpy()
    return zs, rels


def run_batch_bc(model, rows, features, known_protos, radii, device,
                 birth_threshold=0.0, novel_update_rate=0.2, batch=512):
    sids = [r["sample_id"] for r in rows]
    zs, rels = embed_batch(model, features, sids, device, batch)
    mem = BiMemory(known_protos, radii, novel_update_rate=novel_update_rate)
    preds = []
    for i, r in enumerate(rows):
        z = zs[i]
        rel = float(rels[i])
        kid, ks = mem.known_id(z)
        nid, ns = mem.existing_novel(z)
        stats = mem.stats(z, rel, len(features[r["sample_id"]]), known_id=kid, novel_id=nid)
        st = torch.as_tensor([stats], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.action_net(st)
        action = int(logits.argmax(dim=1).item())
        if action == NEW_NOVEL and nid is not None and ns >= birth_threshold:
            action = EXISTING_NOVEL
        if action == KNOWN and kid is not None:
            preds.append({"sample_id": r["sample_id"], "stream_order": i,
                          "prediction_type": "known", "semantic_category_id": int(kid)})
        elif action == EXISTING_NOVEL and nid is not None:
            mem.update_novel(nid, z)
            preds.append({"sample_id": r["sample_id"], "stream_order": i,
                          "prediction_type": "novel", "virtual_category_id": int(nid)})
        else:
            vid = mem.create_novel(z)
            preds.append({"sample_id": r["sample_id"], "stream_order": i,
                          "prediction_type": "novel", "virtual_category_id": int(vid)})
    return preds, mem
