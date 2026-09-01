"""Train the frozen all-known Phase19R deployment model."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.iclr27_phase19r.training.train_controller import run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=50000); p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=1902); p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=["bf16", "fp32"], default="bf16"); p.add_argument("--log-interval", type=int, default=4000)
    p.add_argument("--best", type=Path, default=Path("outputs/iclr27_phase19r/checkpoints/final_rc_ms_best.pt"))
    p.add_argument("--latest", type=Path, default=Path("outputs/iclr27_phase19r/checkpoints/final_rc_ms_latest.pt"))
    p.add_argument("--summary", type=Path, default=Path("outputs/iclr27_phase19r/metrics/final_rc_ms_training.json"))
    p.add_argument("--done", type=Path, default=Path("outputs/iclr27_phase19r/completion/final_rc_ms.done"))
    a = p.parse_args()
    args = argparse.Namespace(fold=0, final=True, updates=a.updates, batch_size=a.batch_size,
                              seed=a.seed, device=a.device, amp=a.amp, ladder="L2", max_states=16,
                              validation_episodes=64, log_interval=a.log_interval, lr=3e-4,
                              best=a.best, latest=a.latest, summary=a.summary, done=a.done)
    run(args)


if __name__ == "__main__": main()
