from __future__ import annotations

import numpy as np


class B2Memory:
    """TrackOCD-v1 corrected B2 online nearest-prototype memory.

    Rules (identical to the corrected baseline):
    - best known prototype sim >= threshold -> known:<semantic id>
    - else best novel center sim >= threshold -> attach + EMA update
    - else create new virtual category
    Optional `novel_only=True` skips the known check (used by D2 after the
    semantic router already decided novel). No future access, no post-hoc
    edits, no knowledge of the true novel category count.
    """

    def __init__(self, known_protos, threshold=0.45, ema=True, novel_only=False):
        self.known_protos = dict(known_protos)
        self.threshold = threshold
        self.ema = ema
        self.novel_only = novel_only
        self.novel = {}
        self.counts = {}
        self.next_id = 100000
        self.log = []

    def predict_one(self, emb, sample_id, stream_order):
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        if not self.novel_only:
            best_k, best_ks = None, -1.0
            for cid, p in self.known_protos.items():
                s = float(np.dot(emb, p))
                if s > best_ks:
                    best_ks, best_k = s, cid
            if best_ks >= self.threshold:
                self.log.append({"stream_order": stream_order, "sample_id": sample_id,
                                 "virtual_category_id": best_k})
                return best_k, "known"
        best_n, best_ns = None, -1.0
        for cid, c in self.novel.items():
            s = float(np.dot(emb, c))
            if s > best_ns:
                best_ns, best_n = s, cid
        if best_ns >= self.threshold:
            self.novel[best_n] = (self.novel[best_n] * self.counts[best_n] + emb) / (
                self.counts[best_n] + 1
            )
            self.novel[best_n] /= np.linalg.norm(self.novel[best_n]) + 1e-12
            self.counts[best_n] += 1
            vid = best_n
        else:
            vid = self.next_id
            self.next_id += 1
            self.novel[vid] = emb.copy()
            self.counts[vid] = 1
        self.log.append({"stream_order": stream_order, "sample_id": sample_id,
                         "virtual_category_id": vid})
        return vid, "novel"

    def memory_stats(self):
        return {"novel_categories": len(self.novel), "known_prototypes": len(self.known_protos)}
