#!/usr/bin/env python3
"""Phase75A supervisor: two independent, fixed-video Q0 control replays."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75a"
ARCHIVE = Path("/data2/usr_for_deadline/trackocd_phase75a")
Q0_STREAM = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
Q0_CHECKPOINT = ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"
Q0_ASSETS = ROOT / "outputs/iclr27_phase74/assets/q0_asset_manifest.jsonl"
CONTROL_SELECTION = ROOT / "outputs/iclr27_phase74/replay/control_video_selection.json"
EXPECTED_STREAM_SHA = "112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac"
EXPECTED_CHECKPOINT_SHA = "809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738"
OVTR_PY = Path("/home/lwr/anaconda3/envs/ovtr/bin/python")
OVTR_COMMIT = "500e72c"


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


def atomic_symlink(dst: Path, target: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve(strict=False) == target.resolve():
        return
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing replay path: {dst}")
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(str(target.resolve()), tmp)
    os.replace(tmp, dst)


def load_control_video_ids() -> dict[str, Any]:
    selected = json.loads(CONTROL_SELECTION.read_text(encoding="utf-8"))["videos"]
    assets: dict[str, int] = {}
    for line in Q0_ASSETS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assets.setdefault(str(row["canonical_video_key"]), int(row["video_id"]))
    missing = [key for key in selected if key not in assets]
    if missing:
        raise ValueError(f"control selection not present in Q0 asset manifest: {missing}")
    ids = [assets[key] for key in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("control selection maps to duplicate numeric video IDs")
    return {"selection_rule": json.loads(CONTROL_SELECTION.read_text(encoding="utf-8")).get("selection_rule"), "canonical_video_keys": selected, "video_ids": ids, "source_sha256": sha256(Q0_ASSETS)}


def command_for(run_dir: Path, video_ids: list[int]) -> list[str]:
    thresholds = ["0.19"] * 7
    ious = ["0.45"] * 7
    miss = ["5"] * 7
    return [
        str(OVTR_PY), str(ROOT / "scripts/iclr27_phase75a/ovtr_native_eval.py"),
        "--config_file", "./config/ovtr_lite_train_val.py",
        "--dataset_file", "lvis_generated_img_seqs", "--batch_size", "1",
        "--with_box_refine", "--two_stage", "--pretrained", str(Q0_CHECKPOINT.resolve()),
        "--score_mode", "base", "--num_workers", "2", "--sampler_lengths", "2",
        "--score_thresh", *thresholds, "--filter_score_thresh", *thresholds,
        "--ious_thresh", *ious, "--miss_tolerance", *miss, "--maximum_quantity", "160",
        "--output_dir", str(run_dir), "--eval", "track", "--result_path_track", str(run_dir / "teta_results"),
        "--video_ids", json.dumps(video_ids), "--native-out", str(run_dir / "native_lineage.jsonl"),
    ]


def canonical_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping: dict[tuple[int, int], int] = {}
    next_id = 0
    canonical: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["video_id"]), int(row["physical_track_id"]))
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out = {k: row.get(k) for k in ("video_id", "image_id", "frame_id", "proposal_local_id", "candidate_rank", "lifecycle", "bbox_xyxy", "base_score", "hit_count", "disappear_time", "score_mode")}
        out["physical_track_canonical_id"] = mapping[key]
        parent = row.get("parent_physical_track_id")
        out["parent_physical_track_canonical_id"] = None if parent is None else mapping.get((int(row["video_id"]), int(parent)))
        canonical.append(out)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return canonical, hashlib.sha256(payload).hexdigest()


def compare(a: Path, b: Path) -> dict[str, Any]:
    ra, ha = canonical_records(a)
    rb, hb = canonical_records(b)
    differences: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(ra, rb)):
        if left != right:
            differences.append({"index": index, "left": left, "right": right})
            if len(differences) >= 20:
                break
    if len(ra) != len(rb):
        differences.append({"length_a": len(ra), "length_b": len(rb)})
    return {"record_count_a": len(ra), "record_count_b": len(rb), "canonical_hash_a": ha, "canonical_hash_b": hb, "exact_canonical_match": bool(len(ra) == len(rb) and ha == hb and not differences), "differences_sample": differences}


def run_one(tag: str, run_dir: Path, video_ids: list[int]) -> dict[str, Any]:
    public_dir = OUT / "replay" / tag
    if (public_dir / ".done").exists():
        return {"tag": tag, "status": "already_done", "run_dir": str(run_dir)}
    if (public_dir / ".launched").exists() or run_dir.exists():
        raise RuntimeError(f"unit {tag} is launched/incomplete; inspect before relaunch: {public_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_symlink(public_dir, run_dir)
    command = command_for(run_dir, video_ids)
    atomic_json(public_dir / ".launched", {"tag": tag, "pid": os.getpid(), "command": command, "started_utc": now(), "gpu": 0})
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.run(command, cwd=ROOT / "third_party/research_refs_phase4n/OVTR/ovtr", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    atomic_text(run_dir / "eval_stdout.log", proc.stdout)
    result_path = run_dir / "teta_results/tao_track.json"
    native_path = run_dir / "native_lineage.jsonl"
    if proc.returncode != 0 or not result_path.is_file() or not native_path.is_file():
        failure = {"tag": tag, "status": "FAILED", "exit_code": proc.returncode, "result_exists": result_path.is_file(), "native_exists": native_path.is_file(), "stdout_tail": proc.stdout[-4000:], "command": command, "started_utc": now()}
        atomic_json(run_dir / "failure.json", failure)
        raise RuntimeError(f"{tag} replay failed; evidence retained at {run_dir}")
    rows = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        failure = {"tag": tag, "status": "FAILED_EMPTY_NATIVE_LINEAGE", "exit_code": proc.returncode, "result_exists": True, "native_exists": True, "command": command}
        atomic_json(run_dir / "failure.json", failure)
        raise RuntimeError(f"{tag} produced empty native lineage")
    summary = {"tag": tag, "status": "PASS_NATIVE_EXPORT", "exit_code": proc.returncode, "record_count": len(rows), "result_path": str(result_path), "native_path": str(native_path), "command": command, "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "ovtr_commit": OVTR_COMMIT}
    atomic_json(run_dir / "replay_summary.json", summary)
    atomic_text(public_dir / ".done", "complete\n")
    return summary


def preflight() -> dict[str, Any]:
    commands = []
    for command in ("free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd"):
        result = subprocess.run(command, shell=True, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        commands.append({"command": command, "exit_code": result.returncode, "stdout": result.stdout})
    return {"observed_utc": now(), "commands": commands, "gpu_count_used": 1, "gpu_mapping": {"control_run_A": 0, "control_run_B": 0}, "external_processes_touched": False, "ram_safety_floor": "retain >=25% RAM; serialized runs use one worker"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"phase75a-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not Q0_STREAM.is_file() or not Q0_CHECKPOINT.is_file() or not OVTR_PY.is_file():
        raise FileNotFoundError("Q0 stream/checkpoint/OVTR interpreter missing")
    stream_sha, ckpt_sha = sha256(Q0_STREAM), sha256(Q0_CHECKPOINT)
    if stream_sha != EXPECTED_STREAM_SHA or ckpt_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f"Q0 lineage hash mismatch stream={stream_sha} checkpoint={ckpt_sha}")
    controls = load_control_video_ids()
    atomic_json(OUT / "audit/preflight.json", preflight())
    atomic_json(OUT / "replay/control_video_manifest.json", {"schema_version": "phase75a.control_videos.v1", **controls, "q0_stream_sha256": stream_sha, "q0_checkpoint_sha256": ckpt_sha, "ovtr_commit": OVTR_COMMIT, "score_mode": "base"})
    results = []
    for tag in ("control_run_A", "control_run_B"):
        results.append(run_one(tag, ARCHIVE / tag, controls["video_ids"]))
    comparison = compare(ARCHIVE / "control_run_A/native_lineage.jsonl", ARCHIVE / "control_run_B/native_lineage.jsonl")
    atomic_json(OUT / "replay/control_replay_comparison.json", {"protocol": "phase75a_frozen_q0_control_replay", "run_id": args.run_id, "results": results, "comparison": comparison, "graph_matching": "track IDs canonicalized by first appearance; numeric IDs need not match", "required_fields": ["frame_id", "proposal_local_id", "candidate_rank", "physical_track_id", "parent_physical_track_id", "lifecycle"], "no_evaluator_join": True, "labels_joined_before_model": False})
    status = "PHASE75A_PASS_EXACT_CONTROL_REPLAY" if comparison["exact_canonical_match"] else "PHASE75A_BLOCKED_CONTROL_NONDETERMINISM"
    atomic_json(OUT / "audit/postflight.json", preflight())
    atomic_json(OUT / "status.json", {"phase": "Phase75A", "status": status, "run_id": args.run_id, "q0_stream_sha256": stream_sha, "q0_checkpoint_sha256": ckpt_sha, "ovtr_commit": OVTR_COMMIT, "score_mode": "base", "control_videos": controls, "comparison": comparison, "q0_model_invoked": True, "event_replay_started": False, "public_sealed_accessed": False, "next_action": "start Phase75B full TRAIN event-video replay only when exact control replay passes" if status == "PHASE75A_PASS_EXACT_CONTROL_REPLAY" else "repair replay implementation before event videos"})
    atomic_text(OUT / "completion/phase75a.done", status + "\n")
    print(json.dumps({"status": status, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
