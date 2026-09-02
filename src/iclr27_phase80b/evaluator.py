"""TRAIN-disjoint retrieval evaluation for the causal-memory scorer."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.iclr27_phase75d.retrieval_metrics import score_records

from .data import MemoryBank, PREFIXES, materialize_bank


def indices(bank: MemoryBank) -> tuple[list[int], list[int]]:
    pos = set(bank.positives); neg = set(bank.negatives)
    return ([i for i, key in enumerate(bank.candidates) if key in pos], [i for i, key in enumerate(bank.candidates) if key in neg])


@torch.no_grad()
def evaluate_banks(model: torch.nn.Module, banks: list[MemoryBank], table, device: torch.device, *, limit: int | None = None) -> dict[str, Any]:
    selected = banks if limit is None else banks[: int(limit)]
    prefix_rows: list[dict[str, Any]] = []
    for t, prefix in enumerate(PREFIXES):
        records: list[dict[str, Any]] = []
        interventions = 0; gate_values: list[float] = []; state_rows: list[dict[str, Any]] = []
        for bank in selected:
            seq = torch.as_tensor(materialize_bank(bank, table), dtype=torch.float32, device=device)
            out = model(seq)
            scores = out["scores"][t].detach().cpu().numpy().astype(np.float32)
            raw = out["raw"][t].detach().cpu().numpy().astype(np.float32)
            gate_values.append(float(out["gate"][t].mean().cpu()))
            interventions += int(np.argmax(scores) != np.argmax(raw))
            state_rows.append({"episode_id": bank.episode_id, "evidence_mean": float(out["evidence"][t].mean().cpu()), "evidence_std": float(out["evidence"][t].std(unbiased=False).cpu()), "gate_mean": gate_values[-1]})
            pos_idx, neg_idx = indices(bank)
            records.append({"query_key": bank.query_key, "category": bank.category, "video": bank.video, "candidates": list(bank.candidates), "positives": list(bank.positives), "negatives": list(bank.negatives), "scores": scores.tolist(), "raw_scores": raw.tolist()})
        metric = score_records(records)
        metric.update({"scope": "phase80b_train_memory_mimic_validation", "intervention_count": interventions, "intervention_rate": float(interventions / max(len(records), 1)), "mean_gate": float(np.mean(gate_values)) if gate_values else 0.0, "state_rows": state_rows})
        prefix_rows.append({"prefix": prefix, "learned": metric, "raw": {k: metric.get(f"raw_{k}") for k in ("r1", "r5", "map", "hard_negative_gap")}, "queries": len(records), "candidate_count": int(sum(len(r["candidates"]) for r in records))})
    return {"prefix_rows": prefix_rows, "queries": len(selected)}


def p16(result: dict[str, Any]) -> dict[str, Any]:
    m = dict(next(x for x in result["prefix_rows"] if x["prefix"] == 16)["learned"])
    for key in ("r1", "map", "hard_negative_gap"):
        m["delta_" + ("hard_gap" if key == "hard_negative_gap" else key)] = float(m[key] - m["raw_" + key])
    return m

