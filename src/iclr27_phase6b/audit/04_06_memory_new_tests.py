"""Phase 6B Audits 4-6 — novel memory legality and NEW path unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase6b/audit/memory_new_tests.json"
import sys
sys.path.insert(
    0, str(ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"))
from models.joint_query import SemanticMemory  # noqa: E402


def test_initial_empty():
    mem = SemanticMemory(known_ids=[4, 34, 35, 36], state_dim=16)
    mem.reset()
    return {
        "novel_protos_shape": list(mem.novel_protos.shape),
        "novel_ids_len": len(mem.novel_ids),
        "cat_slot_len": len(mem.cat_slot),
        "slot_cat_len": len(mem.slot_cat),
        "slot_birth_key_len": len(mem.slot_birth_key),
        "empty": (mem.novel_protos.shape[0] == 0
                  and len(mem.novel_ids) == 0),
    }


def test_birth_legality():
    mem = SemanticMemory(known_ids=[4, 34, 35, 36], state_dim=16)
    mem.reset()
    h = F.normalize(torch.randn(1, 16), dim=-1)
    # No EXISTING possible before any NEW: novel_ids empty -> existing branch
    # must be unreachable. Simulate the evaluator guard.
    can_existing_before_birth = len(mem.novel_ids) > 0
    slot = mem.teacher_birth(99, h[0], physical_key=(1, 1))
    legal = mem.slot_birth_key[slot] == (1, 1) and slot == 0
    return {
        "can_existing_before_birth": bool(can_existing_before_birth),
        "birth_created_slot": slot,
        "birth_key_legal": legal,
        "legality_ok": (not can_existing_before_birth) and legal,
    }


def test_new_path():
    mem = SemanticMemory(known_ids=[4, 34, 35, 36], state_dim=16)
    mem.reset()
    # Force assign_create to prefer NEW: bump the last-layer weight of class 1.
    with torch.no_grad():
        mem.assign_create.net[-1].weight[1, :] = 5.0
        mem.assign_create.net[-1].weight[0, :] = -5.0
        mem.assign_create.net[-1].bias[1] = 1.0
        mem.assign_create.net[-1].bias[0] = -1.0
    h = F.normalize(torch.randn(1, 16), dim=-1)
    logits = mem.forward_assign_create(h, obj_logit=0.5, age=1)
    action = int(logits.argmax().item())
    if action == 1:
        slot = mem.size
        mem.teacher_birth(None, h[0], physical_key=(1, 0))
    else:
        slot = None
    # Next observation must be able to reuse the born slot via sims.
    sims = mem.sims(h)[0]
    k0 = len(mem.known_ids)
    idx = int(sims.argmax().item())
    reuse_is_existing = (idx >= k0) and (mem.novel_ids[idx - k0] == slot)
    return {
        "new_logit_argmax": action,
        "slot_created": slot,
        "memory_size_after_birth": mem.size,
        "next_observation_can_reuse": bool(reuse_is_existing),
        "new_path_ok": (action == 1 and slot is not None
                        and mem.size == 1 and reuse_is_existing),
    }


def main():
    r4 = test_initial_empty()
    r5 = test_birth_legality()
    r6 = test_new_path()
    result = {
        "audit4_novel_memory_initial_state": r4,
        "audit5_birth_legality": r5,
        "audit6_new_path": r6,
        "all_ok": r4["empty"] and r5["legality_ok"] and r6["new_path_ok"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_ok"]:
        raise SystemExit("MEMORY_NEW_TESTS_FAILED")
    print("MEMORY_NEW_TESTS_OK")


if __name__ == "__main__":
    main()
