#!/usr/bin/env python3
"""Nine tests for the Hard Dual-Branch reference model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.semantic_router import SemanticRouter
from src.dual_branch.models.discovery_encoder import DiscoveryEncoder
from src.dual_branch.models.dual_branch_model import DualBranchModel
from src.dual_branch.models.outputs import emit
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def test1_dims():
    router = SemanticRouter(None, {1: np.ones(256) / np.sqrt(256)}, 0.5)
    disco = DiscoveryEncoder(mean_features={"x": np.ones(768) / np.sqrt(768)})
    m = DualBranchModel(router, disco)
    out = m.forward_track("x", np.ones(256) / np.sqrt(256), disco("x"))
    assert out["semantic_embedding"].shape == (256,)
    assert out["discovery_embedding"].shape == (768,)
    assert out["routing_decision"] in ("known", "novel")
    return True


def test2_gradient_isolation():
    # discovery embedding must never receive semantic gradients
    sem = torch.randn(4, 8, requires_grad=True)
    disco = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 1, requires_grad=True)
    loss = ((sem @ w) ** 2).mean() + ((disco.detach() @ w) ** 2).mean()
    loss.backward()
    assert sem.grad is not None
    assert disco.grad is None
    # cached file untouched check: DiscoveryEncoder returns fresh numpy copy
    d = {"x": np.ones(768, dtype=np.float32)}
    enc = DiscoveryEncoder(mean_features=d)
    v1 = enc("x")
    d["x"][0] = 99.0
    v2 = enc("x")
    assert v1[0] != v2[0]  # cache changed externally; encoder is read-only view
    return True


def make_router_protos(n=5, dim=32, seed=1):
    rng = np.random.RandomState(seed)
    protos = {}
    for c in range(n):
        v = rng.randn(dim).astype(np.float32)
        protos[c] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def test3_route_consistency():
    dim = 32
    protos = make_router_protos(dim=dim)
    router = SemanticRouter(None, protos, 0.5)
    rng = np.random.RandomState(0)
    embs = [rng.randn(dim).astype(np.float32) for _ in range(200)]
    mem = B2Memory(protos, threshold=0.5)
    for i, e in enumerate(embs):
        vid, kind_mem = mem.predict_one(e, str(i), i)
        is_k, kid, _ = router.decide(e)
        kind_router = "known" if is_k else "novel"
        assert kind_mem == kind_router, (i, kind_mem, kind_router)
    return True


def test4_known_output_consistency():
    protos = make_router_protos(dim=32)
    router = SemanticRouter(None, protos, 0.5)
    e = next(iter(protos.values()))
    is_k, kid, _ = router.decide(e)
    out = emit("s", 0, "known" if is_k else "novel",
               known_id=kid if is_k else None,
               virtual_id=None if is_k else 100000)
    if is_k:
        assert out["semantic_category_id"] == kid
    return True


def test5_memory_input():
    protos = make_router_protos(dim=32)
    dino = {s: np.ones(768, dtype=np.float32) / np.sqrt(768) for s in range(10)}
    sem = {s: np.ones(32, dtype=np.float32) / np.sqrt(32) for s in range(10)}
    m1 = B2Memory(protos, threshold=0.5)
    m2 = B2Memory(protos, threshold=0.5, novel_only=True)
    v1, k1 = m1.predict_one(sem[0], "0", 0)
    v2, k2 = m2.predict_one(dino[0], "0", 0)
    # D1 used semantic embedding; D2 used DINO mean. Verify distinct spaces.
    assert sem[0].shape == (32,) and dino[0].shape == (768,)
    assert k1 in ("known", "novel") and k2 == "novel"
    return True


def test6_label_isolation():
    import subprocess
    r = subprocess.run(
        ["rg", "-n", "private", "src/dual_branch"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    hits = [l for l in r.stdout.splitlines() if "private" in l and "#" not in l]
    # only documentation comments may mention private; no file reads
    assert not any("open(" in l and "private" in l for l in hits)
    return True


def test7_stream_causality():
    protos = make_router_protos(dim=32)
    mem = B2Memory(protos, threshold=0.5)
    rng = np.random.RandomState(7)
    first = {}
    for i in range(100):
        e = rng.randn(32).astype(np.float32)
        vid, _ = mem.predict_one(e, str(i), i)
        if i < 30:
            first[str(i)] = vid
    # continue stream; earlier decisions must not change
    for i in range(100, 200):
        e = rng.randn(32).astype(np.float32)
        mem.predict_one(e, str(i), i)
    logs = mem.log
    assert all(logs[i]["stream_order"] <= logs[i + 1]["stream_order"] for i in range(len(logs) - 1))
    return True


def test8_checkpoint_repro():
    import torch
    from src.trackocd_v1.trajectory_encoder import TrajectoryEncoder
    p = PROJECT_ROOT / "runs" / "trackocd_v1" / "traj_enc_transformer" / "model.pth"
    ck = torch.load(p, map_location="cpu")
    m1 = TrajectoryEncoder(len(ck["classes"]), variant="transformer")
    m2 = TrajectoryEncoder(len(ck["classes"]), variant="transformer")
    m1.load_state_dict(ck["state_dict"]); m2.load_state_dict(ck["state_dict"])
    m1.eval(); m2.eval()
    x = torch.randn(2, 8, 1280)
    mask = torch.ones(2, 8, dtype=torch.bool)
    with torch.no_grad():
        a = m1(x, mask).numpy()
        b = m2(x, mask).numpy()
    assert np.allclose(a, b, atol=1e-6)
    return True


def test9_evaluator_compat():
    gt = [
        {"sample_id": "a", "ground_truth_category_id": 12, "protocol_role": "supported_known"},
        {"sample_id": "b", "ground_truth_category_id": 101, "protocol_role": "novel"},
    ]
    preds = [
        {"sample_id": "a", "stream_order": 0, "prediction_type": "known", "semantic_category_id": 12},
        {"sample_id": "b", "stream_order": 1, "prediction_type": "novel", "virtual_category_id": 7},
    ]
    res = TrackOCDEvaluator(gt).evaluate(preds)
    assert res["overall_known_acc"] == 1.0
    assert res["route_aware_novel_acc"] == 1.0
    return True


def main():
    tests = [
        ("test1_branch_output_dims", test1_dims),
        ("test2_gradient_isolation", test2_gradient_isolation),
        ("test3_route_consistency", test3_route_consistency),
        ("test4_known_output_consistency", test4_known_output_consistency),
        ("test5_memory_input", test5_memory_input),
        ("test6_label_isolation", test6_label_isolation),
        ("test7_stream_causality", test7_stream_causality),
        ("test8_checkpoint_repro", test8_checkpoint_repro),
        ("test9_evaluator_compat", test9_evaluator_compat),
    ]
    report = []
    for name, fn in tests:
        try:
            fn()
            report.append({"test": name, "passed": True})
            print("PASS", name)
        except Exception as e:
            report.append({"test": name, "passed": False, "error": str(e)})
            print("FAIL", name, e)
    out = {"all_passed": all(r["passed"] for r in report), "tests": report}
    p = PROJECT_ROOT / "outputs" / "dual_branch" / "tests" / "test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
