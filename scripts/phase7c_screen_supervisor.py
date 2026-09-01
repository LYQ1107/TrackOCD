"""Run Phase 7C quick screening candidates (10 epochs each, <=4 GPUs)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PY = "/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"

JOBS = [
    ("outputs/iclr27_phase7c/training/screen_ce_only",
     ["--mode", "hard", "--epochs", "10", "--m-kp", "0", "--m-open", "0"], 0),
    ("outputs/iclr27_phase7c/training/screen_kpoc_hard",
     ["--mode", "hard", "--epochs", "10"], 1),
    ("outputs/iclr27_phase7c/training/screen_kpoc_random",
     ["--mode", "random", "--epochs", "10"], 2),
]


def run(job, gpu):
    out_dir, extra, _ = job
    log = ROOT / out_dir / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "src/iclr27_phase7c/training/train_kpoc.py",
           "--out", out_dir, "--device", f"cuda:{gpu}"] + extra
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p, log


def main():
    queue = list(JOBS)
    running = []
    while queue or running:
        while len(running) < 4 and queue:
            job = queue.pop(0)
            print(f"[{time.strftime('%H:%M:%S')}] launch {job[0]} "
                  f"gpu {job[2]}", flush=True)
            p, log = run(job, job[2])
            running.append((job, p, log))
        done = None
        while done is None:
            for item in list(running):
                job, p, log = item
                if p.poll() is not None:
                    done = item
                    break
            if done is None:
                time.sleep(10)
        job, p, log = done
        running.remove(done)
        print(f"[{time.strftime('%H:%M:%S')}] finished {job[0]} "
              f"code={p.returncode}", flush=True)
        if p.returncode != 0:
            print("\n".join(log.read_text().splitlines()[-20:]), flush=True)
    print("ALL_PHASE7C_SCREEN_DONE", flush=True)


if __name__ == "__main__":
    main()
