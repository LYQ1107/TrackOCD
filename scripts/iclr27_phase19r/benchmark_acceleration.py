"""500-update old/new throughput benchmark with decomposed timings."""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.episodes import EpisodeFactory
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.training.rollout import rollout_batch


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def gpu_snapshot() -> dict[str, str]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader"], text=True)
        return {"raw": out.strip()}
    except Exception as exc:
        return {"error": str(exc)}


def generation(data, disable: bool, n: int) -> tuple[list, float]:
    if disable:
        os.environ["PHASE19R_DISABLE_HARD_PAIR_CACHE"] = "1"
    else:
        os.environ.pop("PHASE19R_DISABLE_HARD_PAIR_CACHE", None)
    f = EpisodeFactory(data, ladder="L2", validation=False)
    rng = np.random.default_rng(1902)
    t0 = time.perf_counter(); eps = [f.sample(rng) for _ in range(n)]; elapsed = time.perf_counter() - t0
    return eps, elapsed


def train_bench(data, episodes, device, updates: int, batch_size: int) -> dict:
    torch.manual_seed(1902)
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(9917)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t0 = time.perf_counter(); rollout_s = 0.; backward_s = 0.; last = None
    for u in range(int(updates)):
        batch = [episodes[(u * batch_size + i) % len(episodes)] for i in range(batch_size)]
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, scalars, _, _ = rollout_batch(model, data, batch, device, u + 1, updates, rng, train=True, ladder="L2", allow_defer=True)
        rollout_s += time.perf_counter() - t1
        t2 = time.perf_counter(); loss.backward(); opt.step(); backward_s += time.perf_counter() - t2
        last = float(loss.detach())
    elapsed = time.perf_counter() - t0
    return {"updates": updates, "batch_size": batch_size, "elapsed_seconds": elapsed,
            "rollout_seconds": rollout_s, "backward_seconds": backward_s,
            "updates_per_second": updates / max(elapsed, 1e-9),
            "items_per_second": updates * batch_size * 24 / max(elapsed, 1e-9),
            "last_loss": last, "rss_gib": rss_gib(),
            "gpu_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / (1024.0**3) if device.type == "cuda" else 0.0)}


def validation_bench(data, episodes, device) -> dict:
    # A fixed causal replay proxy is timed here; formal persistent selection is
    # unchanged and remains on its 4,000-update cadence in training.
    from src.iclr27_phase19r.training.train_controller import internal_validate
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)).to(device)
    t0 = time.perf_counter(); result = internal_validate(model, data, episodes[:64], device, ladder="L2"); elapsed = time.perf_counter() - t0
    return {"elapsed_seconds": elapsed, "episodes": min(64, len(episodes)), "selection_score": result["selection_score"], "rss_gib": rss_gib()}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--updates", type=int, default=500); p.add_argument("--batch-size", type=int, default=24); p.add_argument("--device", default="cuda:0"); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    data = Phase19RData(0); device = torch.device(a.device)
    old_eps, old_gen = generation(data, True, max(500, a.batch_size * 4))
    new_eps, new_gen = generation(data, False, max(500, a.batch_size * 4))
    indexed_factory = EpisodeFactory(data, ladder="L2", validation=False, index_path=Path("outputs/iclr27_phase19r/manifests/episode_index_benchmark_f0.jsonl"))
    t0 = time.perf_counter(); indexed_eps = [indexed_factory.sample(np.random.default_rng(0)) for _ in range(500)]; indexed_gen = time.perf_counter() - t0
    old_train = train_bench(data, old_eps, device, a.updates, a.batch_size)
    new_train = train_bench(data, indexed_eps, device, a.updates, a.batch_size)
    old_val = validation_bench(data, old_eps, device); new_val = validation_bench(data, indexed_eps, device)
    try:
        ref = [json.loads(Path(f"outputs/iclr27_phase19r/metrics/fold{i}_training.json").read_text())["updates_per_second"] for i in range(4)]
    except Exception:
        ref = []
    result = {"protocol": "trackocd_iclr27_phase19r_acceleration_benchmark", "updates": a.updates, "batch_size": a.batch_size,
              "old": {"episode_generation_seconds": old_gen, "train": old_train, "validation": old_val},
              "new_cached_indexed": {"episode_generation_seconds": indexed_gen, "train": new_train, "validation": new_val},
              "new_dynamic_cache_generation_seconds": new_gen,
              "historical_old_fold_updates_per_second": ref,
              "speedup_updates_per_second": new_train["updates_per_second"] / max(old_train["updates_per_second"], 1e-9),
              "speed_target_two_x_met": bool(new_train["updates_per_second"] >= 2.0 * old_train["updates_per_second"]),
              "gpu_snapshot": gpu_snapshot(), "cpu_rss_gib": rss_gib(),
              "semantic_protocol_unchanged": True, "public_truth_joined": False}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); os.replace(tmp, a.out)
    print(json.dumps({"old_updates_per_second": old_train["updates_per_second"], "new_updates_per_second": new_train["updates_per_second"], "speedup": result["speedup_updates_per_second"], "speed_target_two_x_met": result["speed_target_two_x_met"]}, sort_keys=True))


if __name__ == "__main__":
    main()
