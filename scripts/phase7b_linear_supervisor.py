"""Run the Phase 7B architecture-switch (linear TOSE) training jobs."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PY = "/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"

JOBS = [
    ("outputs/iclr27_phase7b/training/tose_linear_main",
     ["--head-type", "linear"], 0),
    ("outputs/iclr27_phase7b/training/abl_linear_frame_level",
     ["--head-type", "linear", "--frame-level"], 1),
    ("outputs/iclr27_phase7b/training/abl_linear_no_dist",
     ["--head-type", "linear", "--no-dist"], 2),
    ("outputs/iclr27_phase7b/training/abl_linear_no_proxy",
     ["--head-type", "linear", "--no-proxy"], 3),
    ("outputs/iclr27_phase7b/training/abl_linear_classifier_conf",
     ["--head-type", "linear", "--classifier-conf"], None),
]


def run(job, gpu):
    out_dir, extra, _ = job
    log = ROOT / out_dir / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "src/iclr27_phase7b/training/train_explainability.py",
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
            gpu = job[2]
            if gpu is None:
                used = {job[2] for job, _, _ in running}
                gpu = next(g for g in range(4) if g not in used)
                job = (job[0], job[1], gpu)
            print(f"[{time.strftime('%H:%M:%S')}] launch {job[0]} "
                  f"gpu {job[2]}", flush=True)
            p, log = run(job, gpu)
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
    print("ALL_PHASE7B_LINEAR_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
