"""Benchmark the fast causal-state path against the pre-repair 500-update run."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.episodes import EpisodeFactory
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.training.rollout import rollout_batch


def main() -> None:
    updates = 500; batch_size = 24; device = torch.device("cuda:0")
    data = Phase19RData(0)
    idx = Path("outputs/iclr27_phase19r/manifests/episode_index_benchmark_f0.jsonl")
    factory = EpisodeFactory(data, ladder="L2", validation=False, index_path=idx)
    episodes = [factory.sample(np.random.default_rng(0)) for _ in range(512)]
    torch.manual_seed(1902)
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4); rng = np.random.default_rng(9917)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter(); rollout_s = backward_s = 0.0; last = None
    for u in range(updates):
        batch = [episodes[(u * batch_size + i) % len(episodes)] for i in range(batch_size)]
        opt.zero_grad(set_to_none=True); t1 = time.perf_counter()
        loss, _, _, _ = rollout_batch(model, data, batch, device, u + 1, updates, rng, train=True, ladder="L2", allow_defer=True)
        rollout_s += time.perf_counter() - t1; t2 = time.perf_counter(); loss.backward(); opt.step(); backward_s += time.perf_counter() - t2; last = float(loss.detach())
    elapsed = time.perf_counter() - t0
    old = json.loads(Path("outputs/iclr27_phase19r/metrics/acceleration_benchmark.json").read_text())["old"]["train"]
    result = {"protocol": "trackocd_iclr27_phase19r_fast_path_benchmark", "updates": updates, "batch_size": batch_size,
              "old_reference": old, "new_fast": {"elapsed_seconds": elapsed, "rollout_seconds": rollout_s, "backward_seconds": backward_s,
                  "updates_per_second": updates / elapsed, "items_per_second": updates * batch_size * 24 / elapsed, "last_loss": last,
                  "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024.0**3)},
              "speedup_vs_old": (updates / elapsed) / max(float(old["updates_per_second"]), 1e-9),
              "speed_target_two_x_met": bool((updates / elapsed) >= 2.0 * float(old["updates_per_second"])),
              "state_semantics": "StateMemory fast_mode; snapshots/anchor values deferred, causal updates unchanged",
              "public_truth_joined": False}
    out = Path("outputs/iclr27_phase19r/metrics/acceleration_benchmark_fast.json"); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); tmp.replace(out)
    print(json.dumps({"old_updates_per_second": old["updates_per_second"], "new_updates_per_second": result["new_fast"]["updates_per_second"], "speedup": result["speedup_vs_old"], "speed_target_two_x_met": result["speed_target_two_x_met"]}, sort_keys=True))


if __name__ == "__main__": main()
