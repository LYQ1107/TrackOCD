from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from src.iclr27_phase75d.retrieval_metrics import score_records


def evaluate_examples(model, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Evaluate raw-preserving HELP-only routing on stored val examples."""
    by_prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
    details: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not rows:
        return {"prefix_rows": [], "examples": 0}
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray([r["features"] for r in rows], dtype=np.float32), device=device)
        pred = model(x).argmax(dim=-1).detach().cpu().numpy()
    for row, action in zip(rows, pred.tolist()):
        use_relation = int(action) == 0  # HELP only; HARM/NEUTRAL exact raw fallback
        scores = row["relation_scores"] if use_relation else row["raw_scores"]
        by_prefix[int(row["prefix"])].append({
            "query_key": row["query_key"], "category": row.get("category"), "video": row.get("video"),
            "candidates": row["candidates"], "positives": row["positives"], "negatives": row["negatives"],
            "scores": scores, "raw_scores": row["raw_scores"],
        })
        details[int(row["prefix"])].append({
            "query_key": row["query_key"], "episode_id": row["episode_id"], "action": int(action),
            "use_relation": use_relation, "teacher_label": int(row["label"]),
            "raw_correct": bool(row["raw_correct"]), "relation_correct": bool(row["relation_correct"]),
            "unsafe": bool(row["raw_correct"] and use_relation and not row["relation_correct"]),
            "category": row.get("category"), "video": row.get("video"),
        })
    prefix_rows = []
    for prefix in (1, 2, 4, 8, 16):
        metric = score_records(by_prefix[prefix])
        d = details[prefix]
        teacher_agreement = float(np.mean([int(x["action"] == x["teacher_label"]) for x in d])) if d else 0.0
        prefix_rows.append({"prefix": prefix, "metric": metric, "teacher_agreement": teacher_agreement, "teacher_use_rate": float(np.mean([int(x["teacher_label"] == 0) for x in d])) if d else 0.0, "router_help_rate": float(np.mean([int(x["action"] == 0) for x in d])) if d else 0.0, "unsafe_count": int(sum(int(x["unsafe"]) for x in d)), "details": d})
    return {"prefix_rows": prefix_rows, "examples": len(rows)}


def p16(result: dict[str, Any]) -> dict[str, Any]:
    row = next(x for x in result["prefix_rows"] if x["prefix"] == 16)
    m = row["metric"]
    return {"r1": m["r1"], "map": m["map"], "raw_r1": m["raw_r1"], "raw_map": m["raw_map"], "hard_negative_gap": m["hard_negative_gap"], "raw_hard_negative_gap": m["raw_hard_negative_gap"], "delta_r1": m["r1"] - m["raw_r1"], "delta_map": m["map"] - m["raw_map"], "delta_hard_gap": m["hard_negative_gap"] - m["raw_hard_negative_gap"], "unsafe_flip_count": m["unsafe_flip_count"], "router_help_rate": row["router_help_rate"], "teacher_agreement": row["teacher_agreement"], "teacher_use_rate": row["teacher_use_rate"], "queries": m["queries"]}
