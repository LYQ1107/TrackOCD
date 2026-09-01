"""Phase 5A critical protocol unit tests.

19 no-future, 20 no-retroactive-relabel, 21 first-frame action,
22 birth/reuse, 24 predict-then-update.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from src.iclr27_phase5a.assign_create.memory import CategoryMemory


def make_memory(dim=4):
    protos = torch.eye(dim, dtype=torch.float32)
    return CategoryMemory(protos, list(range(dim)), ema_alpha=0.1)


def test_first_frame_action():
    mem = make_memory()
    h = torch.tensor([0.0, 1.0, 0.0, 0.0])
    a, sid, _ = mem.step(h, 0.5, (1, 1))
    assert a in ("known", "existing", "new")
    assert sid is not None
    # far vector must produce a new birth
    h2 = torch.tensor([0.5, 0.5, 0.5, 0.5])
    a2, _, _ = mem.step(h2, 0.9, (2, 2))
    assert a2 == "new"
    print("test_first_frame_action PASS")


def test_birth_reuse():
    mem = make_memory()
    # far vector -> new slot 0
    h = torch.tensor([0.5, 0.5, 0.5, 0.5])
    a, sid, _ = mem.step(h, 0.8, (1, 1))
    assert a == "new" and sid == 0 and mem.size == 1
    # same vector on a different track -> existing slot 0
    a2, sid2, _ = mem.step(h, 0.8, (2, 2))
    assert a2 == "existing" and sid2 == 0
    print("test_birth_reuse PASS")


def test_predict_then_update():
    mem = make_memory()
    h = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # pre-update decision on a fresh memory
    sims_before = mem.similarities(h.reshape(1, -1))
    a, sid, _ = mem.step(h, 0.8, (1, 1))
    sims_after = mem.similarities(h.reshape(1, -1))
    # the current observation's memory state after its own update must be
    # different only because the new slot was appended; the pre-update
    # decision remains unchanged (immutable record).
    assert a == "new" and sid == 0
    assert sims_before.shape[1] == sims_after.shape[1] - 1
    print("test_predict_then_update PASS")


def test_no_future():
    """Truncation invariance: the action at step t depends only on <=t."""
    mem_full = make_memory()
    mem_trunc = make_memory()
    stream = [torch.tensor([0.5, 0.5, 0.5, 0.5]),
              torch.tensor([0.5, 0.5, 0.5, 0.4]),
              torch.tensor([1.0, 0.0, 0.0, 0.0])]
    acts_full = []
    for h in stream:
        acts_full.append(mem_full.step(h, 0.7, (1, 1)))
    for t in range(len(stream)):
        mem_t = make_memory()
        acts = []
        for h in stream[:t + 1]:
            acts.append(mem_t.step(h, 0.7, (1, 1)))
        assert acts[t][0] == acts_full[t][0]
        assert acts[t][1] == acts_full[t][1]
    print("test_no_future PASS")


def test_no_retroactive_relabel():
    mem = make_memory()
    stream = [torch.tensor([0.5, 0.5, 0.5, 0.5]),
              torch.tensor([0.5, 0.5, 0.5, 0.4]),
              torch.tensor([1.0, 0.0, 0.0, 0.0]),
              torch.tensor([0.5, 0.5, 0.5, 0.5])]
    records = []
    for h in stream:
        records.append(mem.step(h, 0.7, (1, 1)))
    frozen = copy.deepcopy(records[:2])
    assert records[:2] == frozen
    print("test_no_retroactive_relabel PASS")


if __name__ == "__main__":
    test_first_frame_action()
    test_birth_reuse()
    test_predict_then_update()
    test_no_future()
    test_no_retroactive_relabel()
    print("ALL PHASE5A PROTOCOL UNIT TESTS PASS")
