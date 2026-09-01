"""Second screening wave: 16-dim trajectory features, margins on/off."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PY = "/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"

JOBS = [
    ("outputs/iclr27_phase7c/training/screen2_ce16",
     ["--mode", "hard", "--epochs", "8", "--m-kp", "0", "--m-open", "0",
      "--w-kp", "0", "--w-open", "0"], 1),
    ("outputs/iclr27_phase7c/training/screen2_kp16",
     ["--mode", "hard", "--epochs", "8", "--m-kp", "0.5", "--m-open", "0.5",
      "--w-kp", "1.0", "--w-open", "1.0"], 4),
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
    running = []
    for job in JOBS:
        print(f"[{time.strftime('%H:%M:%S')}] launch {job[0]} "
              f"gpu {job[2]}", flush=True)
        p, log = run(job, job[2])
        running.append((job, p, log))
    while running:
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
    print("SCREEN2_DONE", flush=True)


if __name__ == "__main__":
    main()
