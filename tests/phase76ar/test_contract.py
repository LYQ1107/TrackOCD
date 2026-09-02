from __future__ import annotations

import json
from pathlib import Path

import torch

from src.iclr27_phase76ar.data import LegalFitEpisode, MemoryMimicBank, load_stream_payload
from src.iclr27_phase76ar.relation_model import SelectiveAnchoredRelation


def test_raw_first_and_invalid_support_fallback():
    model = SelectiveAnchoredRelation().eval()
    raw = torch.tensor(0.37)
    empty = model(torch.zeros((0, 1536)), torch.zeros((0, 5)), torch.zeros(13), raw, torch.tensor([raw]))
    assert torch.isfinite(empty["final"])
    assert float(empty["final"]) == float(raw)
    assert float(empty["gate"]) == 0.0
    assert float(empty["delta_bounded"]) == 0.0


def test_per_match_quality_and_length_normalized_pool():
    torch.manual_seed(4)
    model = SelectiveAnchoredRelation().eval()
    tokens = torch.randn(2, 1536)
    quality = torch.tensor([[0.9, 0.1, 0.1, 0.5, 0.5], [0.1, 0.9, 0.9, 0.1, 0.1]])
    summary = torch.zeros(13)
    out = model(tokens, quality, summary, torch.tensor(0.2), torch.tensor([0.2, 0.1]))
    changed = model(tokens, quality.flip(0), summary, torch.tensor(0.2), torch.tensor([0.2, 0.1]))
    assert out["weights"].shape == (2,)
    assert not torch.allclose(out["weights"], changed["weights"])
    duplicate = model(torch.cat([tokens[:1], tokens[:1]], dim=0), torch.cat([quality[:1], quality[:1]], dim=0), summary, torch.tensor(0.2), torch.tensor([0.2, 0.1]))
    single = model(tokens[:1], quality[:1], summary, torch.tensor(0.2), torch.tensor([0.2, 0.1]))
    assert torch.allclose(duplicate["pooled"], single["pooled"], atol=1e-6)


def test_dual_stream_objects_are_distinct():
    paths = sorted(Path("outputs/iclr27_phase76ar/banks").glob("streams_f*.json"))
    assert len(paths) == 4
    memory, legal = load_stream_payload(paths[0])
    assert memory and legal
    assert isinstance(memory[0], MemoryMimicBank)
    assert isinstance(legal[0], LegalFitEpisode)
    assert memory[0].source != legal[0].source
    assert memory[0].episode_id != legal[0].episode_id or memory[0].query_key == legal[0].query_key


def test_preregistration_forbidden_inputs():
    cfg = json.loads(Path("configs/iclr27_phase76ar/preregistration.json").read_text())
    forbidden = set(cfg["forbidden_inference_inputs"])
    assert {"category", "semantic_id", "physical_id", "text", "future", "sealed labels"}.issubset(forbidden)
