#!/usr/bin/env python3
"""Replay all TRAIN videos referenced by the frozen Phase74S event manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75b"
ARCHIVE = Path("/data2/usr_for_deadline/trackocd_phase75b")
MODEL_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
Q0_CHECKPOINT = ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"
Q0_STREAM = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
EXPECTED_CHECKPOINT_SHA = "809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738"
EXPECTED_STREAM_SHA = "112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac"
OVTR_PY = Path("/home/lwr/anaconda3/envs/ovtr/bin/python")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def event_video_ids() -> list[int]:
    videos: set[int] = set()
    for line in MODEL_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        videos.add(int(row["source_video"]))
        videos.add(int(row["target_video"]))
    return sorted(videos)


def preflight() -> dict[str, Any]:
    commands = []
    for command in ("free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd"):
        result = subprocess.run(command, shell=True, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        commands.append({"command": command, "exit_code": result.returncode, "stdout": result.stdout})
    return {"observed_utc": now(), "commands": commands, "gpu_count_used": 1, "gpu_mapping": {"event_full_sequence": 0}, "external_processes_touched": False, "ram_policy": "one OVTR worker with num_workers=2; retain >=25% RAM"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"phase75b-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}")
    args = parser.parse_args()
    if not MODEL_MANIFEST.is_file() or not Q0_CHECKPOINT.is_file() or not Q0_STREAM.is_file() or not OVTR_PY.is_file():
        raise FileNotFoundError("Phase74S model manifest, Q0 assets, or OVTR interpreter missing")
    model_sha = sha256(MODEL_MANIFEST)
    ckpt_sha, stream_sha = sha256(Q0_CHECKPOINT), sha256(Q0_STREAM)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA or stream_sha != EXPECTED_STREAM_SHA:
        raise RuntimeError(f"frozen Q0 hash mismatch checkpoint={ckpt_sha} stream={stream_sha}")
    videos = event_video_ids()
    if len(videos) != 91:
        raise RuntimeError(f"expected 91 unique event TRAIN videos, got {len(videos)}")
    run_dir = ARCHIVE / "event_full_sequence"
    public_dir = OUT / "replay/event_full_sequence"
    if (public_dir / ".done").exists():
        print(json.dumps({"status": "already_done", "run_dir": str(run_dir)}))
        return
    if (public_dir / ".launched").exists() or run_dir.exists():
        raise RuntimeError(f"unfinished event replay exists; inspect {public_dir} and {run_dir} before relaunch")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    public_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = public_dir.with_name(f".{public_dir.name}.{os.getpid()}.tmp")
    os.symlink(str(run_dir.resolve()), tmp_link)
    os.replace(tmp_link, public_dir)
    command = [
        str(OVTR_PY), str(ROOT / "scripts/iclr27_phase75a/ovtr_native_eval.py"),
        "--config_file", str(ROOT / "configs/iclr27_phase75b/ovtr_event_train.py"),
        "--dataset_file", "lvis_generated_img_seqs", "--batch_size", "1", "--with_box_refine", "--two_stage",
        "--pretrained", str(Q0_CHECKPOINT.resolve()), "--score_mode", "base", "--num_workers", "2", "--sampler_lengths", "2",
        "--score_thresh", *("0.19",) * 7, "--filter_score_thresh", *("0.19",) * 7, "--ious_thresh", *("0.45",) * 7,
        "--miss_tolerance", *("5",) * 7, "--maximum_quantity", "160", "--output_dir", str(run_dir), "--eval", "track",
        "--result_path_track", str(run_dir / "teta_results"), "--video_ids", json.dumps(videos), "--native-out", str(run_dir / "native_lineage.jsonl"),
    ]
    atomic_json(public_dir / ".launched", {"phase": "Phase75B", "run_id": args.run_id, "pid": os.getpid(), "started_utc": now(), "gpu": 0, "command": command, "model_manifest_sha256": model_sha})
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    pre = preflight()
    atomic_json(OUT / "audit/preflight.json", pre)
    proc = subprocess.run(command, cwd=ROOT / "third_party/research_refs_phase4n/OVTR/ovtr", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    atomic_text(run_dir / "eval_stdout.log", proc.stdout)
    native = run_dir / "native_lineage.jsonl"
    frames = run_dir / "native_lineage.frames.jsonl"
    teta = run_dir / "teta_results/tao_track.json"
    valid = proc.returncode == 0 and native.is_file() and frames.is_file() and teta.is_file()
    if valid:
        native_rows = [json.loads(x) for x in native.read_text(encoding="utf-8").splitlines() if x.strip()]
        frame_rows = [json.loads(x) for x in frames.read_text(encoding="utf-8").splitlines() if x.strip()]
        seen_videos = {int(x["video_id"]) for x in frame_rows}
        required_fields = {"frame_id", "proposal_local_id", "candidate_rank", "physical_track_id", "parent_physical_track_id", "lifecycle"}
        valid = bool(frame_rows) and seen_videos == set(videos) and all(required_fields.issubset(x) for x in native_rows)
    if not valid:
        failure = {"phase": "Phase75B", "status": "FAILED_OR_INCOMPLETE_EVENT_REPLAY", "exit_code": proc.returncode, "native_exists": native.is_file(), "frame_trace_exists": frames.is_file(), "teta_exists": teta.is_file(), "stdout_tail": proc.stdout[-6000:], "command": command, "event_video_count": len(videos)}
        atomic_json(run_dir / "failure.json", failure)
        atomic_json(OUT / "status.json", {"phase": "Phase75B", "status": "PHASE75B_BLOCKED_EVENT_REPLAY", "run_id": args.run_id, "failure": failure, "public_sealed_accessed": False})
        raise RuntimeError("Phase75B event replay failed; evidence retained")
    summary = {"phase": "Phase75B", "status": "PASS_FULL_EVENT_VIDEO_REPLAY", "run_id": args.run_id, "exit_code": proc.returncode, "event_video_count": len(videos), "frame_trace_count": len(frame_rows), "native_record_count": len(native_rows), "event_video_ids": videos, "model_manifest_sha256": model_sha, "q0_checkpoint_sha256": ckpt_sha, "q0_stream_sha256": stream_sha, "ovtr_commit": "500e72c", "score_mode": "base", "labels_joined_before_model": False, "future_rows_or_tracks": False, "public_sealed_accessed": False, "command": command}
    atomic_json(run_dir / "replay_summary.json", summary)
    atomic_text(public_dir / ".done", "complete\n")
    post = preflight()
    atomic_json(OUT / "audit/postflight.json", post)
    atomic_json(OUT / "status.json", {**summary, "preflight": pre, "postflight": post, "next_action": "run Phase75B observability-only prefix replay; no representation/controller until O gate"})
    atomic_text(OUT / "completion/phase75b.done", "PASS_FULL_EVENT_VIDEO_REPLAY\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
