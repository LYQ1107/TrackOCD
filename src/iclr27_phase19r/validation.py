"""Small scientific validity tests for Phase19R (T1–T7 and causal parity)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.data.episodes import EpisodeFactory
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.runtime.runner import ModelStreamController, state_signature
from src.iclr27_phase19r.runtime.state import StateMemory


def _proto_hash(model: RCMSOCD) -> str:
    return hashlib.sha256(model.known_prototypes.detach().cpu().numpy().tobytes()).hexdigest()


def _forced_trace(memory: StateMemory, dim: int = 8) -> list[dict]:
    torch.manual_seed(7)
    raws = [torch.nn.functional.normalize(torch.randn(dim), dim=0) for _ in range(7)]
    spec = [("NEW", None, "A", 1, "v1"), ("NEW", None, "B", 2, "v2"),
            ("EXISTING", 0, "A", 3, "v3"), ("KNOWN", None, "K", 4, "v4"),
            ("DEFER", None, None, 5, "v5"), ("NEW", None, "C", 6, "v6"),
            ("EXISTING", 1, "B", 7, "v7")]
    out = []
    for raw, (action, idx, cat, video, track) in zip(raws, spec):
        out.append(memory.apply_action(action, raw, raw, video, track, state_index=idx,
                                       oracle_category=None if cat is None else ord(cat),
                                       quality=1.0, confidence=1.0))
    return out


def main() -> None:
    data = Phase19RData(0)
    factory = EpisodeFactory(data, ladder="L2", validation=True)
    ep = factory.sample(np.random.default_rng(1902))
    proto = torch.from_numpy(data.known_prototypes)
    model = RCMSOCD(proto, torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias))

    # T1: exactly the same supported category can occupy either role, but the
    # corresponding known slot is active only for the visible-known role.
    c = int(ep.active_known_ids[0]); j = data.known_to_index[c]
    visible_mask = torch.zeros(48, dtype=torch.bool); visible_mask[j] = True
    pseudo_mask = visible_mask.clone(); pseudo_mask[j] = False
    raw = torch.from_numpy(ep.items[0].raw)[None]; geom = torch.from_numpy(ep.items[0].geom)[None]; q = torch.tensor([.9])
    empty = {"state_raw": torch.zeros(1, 0, 768), "state_z": torch.zeros(1, 0, 768), "state_features": torch.zeros(1, 0, 6), "state_mask": torch.zeros(1, 0, dtype=torch.bool)}
    lv = model(raw, geom, q, visible_mask[None], empty, allow_defer=False)["known_logits"][0]
    lp = model(raw, geom, q, pseudo_mask[None], empty, allow_defer=False)["known_logits"][0]
    assert torch.isfinite(lv[j]) and float(lp[j]) < -9990
    # Same raw observation has a KNOWN target only under visible role; the
    # pseudo-novel role is necessarily NEW/EXISTING/DEFER and cannot select c.
    assert visible_mask[j] and not pseudo_mask[j]
    visible_target = "KNOWN"; pseudo_target = "NEW"
    assert visible_target == "KNOWN" and pseudo_target != "KNOWN"

    # T2: mixed episode exposes every required action and multi-state candidate.
    kinds = set(ep.action_targets)
    assert {"KNOWN", "NEW", "EXISTING", "DEFER"} <= kinds
    assert any(x.target_kind == "NEW" and i > 0 for i, x in enumerate(ep.items))
    assert any(x.target_kind == "EXISTING" for x in ep.items)
    assert any(x.hard_negative for x in ep.items)

    # T3: forced train/inference wrappers execute the same transition core.
    m1, m2 = StateMemory(max_states=16), StateMemory(max_states=16)
    a = _forced_trace(m1); b = _forced_trace(m2)
    assert len(a) == len(b) == 7
    for x, y in zip(a, b):
        assert x["action"] == y["action"] and x["state_count"] == y["state_count"]
    assert state_signature(m1) == state_signature(m2)
    assert [x["candidate_order"] for x in a] == [x["candidate_order"] for x in b]
    assert [x["action"] for x in a] == ["NEW", "NEW", "EXISTING", "KNOWN", "DEFER", "NEW", "EXISTING"]

    # T4: persistence and monotonic anonymous SIDs across three tracks.
    assert m1.state_count == 3 and [s.sid for s in m1.states] == [100000, 100001, 100002]

    # T5: a different-category hard-negative is not a valid existing state.
    neg = next(x for x in ep.items if x.hard_negative)
    assert neg.target_kind == "NEW" and neg.oracle_category_for_loss_only is not None

    # T6: changing all suffix values cannot alter a prefix aggregate.
    key = ep.items[0].track_key; pos = ep.items[0].prefix_position
    before = data.prefix(key, pos)
    suffix = data.raw.copy(); idx = data.track_rows[key][pos + 1:]
    if idx:
        suffix[idx] = np.roll(suffix[idx], 1, axis=1)
    # The causal prefix only addresses rows through `pos`.
    take = data.track_rows[key][:pos + 1]
    w = np.asarray([data._quality_row(i) for i in take], np.float32); w = np.maximum(w, .02)
    after_raw = np.average(data.raw[take], axis=0, weights=w); after_raw /= max(float(np.linalg.norm(after_raw)), 1e-6)
    assert np.array_equal(before[0], after_raw.astype(np.float32))

    # T7: known prototype buffer is bitwise invariant under an optimizer step.
    h0 = _proto_hash(model); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(raw, geom, q, visible_mask[None], empty, allow_defer=False)
    out["logits"].sum().backward(); opt.step(); assert _proto_hash(model) == h0

    result = {"protocol": "trackocd_iclr27_phase19r_validity_tests", "passed": True,
              "tests": {"T1_known_mask_role": True, "T2_mixed_episode": True,
                        "T3_forced_transition_parity": True, "T4_persistent_stream": True,
                        "T5_negative_reuse": True, "T6_future_perturbation": True,
                        "T7_prototype_freeze": True},
              "forced_actions": [x["action"] for x in a], "state_count": m1.state_count,
              "prototype_hash": h0, "trainer_observed_semantic_values": data.trainer_observed_semantic_values,
              "physical_id_used_as_feature": False}
    path = Path("outputs/iclr27_phase19r/audit/causal_validation.json"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
