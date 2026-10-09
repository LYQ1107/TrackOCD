"""Small adapter exposing the existing frozen PHE-Track checkpoint online."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


class PHETrackAdapter:
    """Run the historical PHE hash memory on one anonymous track vector.

    The adapter changes the old image-cache lookup into ``step(vector)``.  It
    does not retrain PHE, add novel text, or read evaluator labels.  The
    checkpoint's known class IDs are outputs only; unavailable known classes
    remain an explicit coverage limitation in the report.
    """

    def __init__(self, checkpoint: Path, *, radius: int = 2, device: str = "cpu") -> None:
        import torch

        from src.ocd.phe_track.phe_track_model import PPNetTrack

        payload = torch.load(str(checkpoint), map_location=device)
        class_ids = [int(value) for value in payload["class_ids"]]
        model = PPNetTrack(
            in_dim=768,
            prototype_dim=768,
            num_classes=len(class_ids),
            global_proto_per_class=10,
            hash_code_length=12,
        )
        model.load_state_dict(payload["ema"])
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.class_ids = class_ids
        self.radius = int(radius)
        self.known_centers: List[Tuple[int, np.ndarray]] = []
        with torch.no_grad():
            for class_index in range(len(class_ids)):
                begin = class_index * self.model.global_proto_per_class
                end = begin + self.model.global_proto_per_class
                prototype = self.model.prototype_vectors_global[begin:end].mean(0, keepdim=True)
                code = torch.tanh(self.model.hash_head(prototype) * 3).sign()[0].cpu().numpy()
                self.known_centers.append((class_index, (code > 0).astype(np.int8)))
        self.reset()

    def reset(self) -> None:
        self.novel_centers: List[Tuple[str, np.ndarray]] = []
        self.next_id = 0

    def _hash(self, vector: np.ndarray) -> np.ndarray:
        value = self.torch.as_tensor(np.asarray(vector, dtype=np.float32), device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            code = self.model(value)[0].detach().cpu().numpy()
        return (code > 0).astype(np.int8)

    def step(self, vector: np.ndarray) -> dict[str, Any]:
        code = self._hash(vector)
        best_distance = self.radius + 1
        best_kind = None
        best_token = None
        best_class_index = None
        for class_index, center in self.known_centers:
            distance = int(np.count_nonzero(code != center))
            if distance < best_distance:
                best_distance = distance
                best_kind = "KNOWN"
                best_token = "K:PHE:%d" % class_index
                best_class_index = class_index
        for token, center in self.novel_centers:
            distance = int(np.count_nonzero(code != center))
            if distance < best_distance:
                best_distance = distance
                best_kind = "EXISTING"
                best_token = token
                best_class_index = None
        if best_token is not None and best_distance <= self.radius:
            return {"kind": best_kind, "token": best_token, "hamming_distance": best_distance, "class_index": best_class_index}
        token = "A:PHE:%d" % self.next_id
        self.next_id += 1
        self.novel_centers.append((token, code.copy()))
        return {"kind": "NEW", "token": token, "hamming_distance": best_distance}
