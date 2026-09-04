#!/usr/bin/env python3
"""Run pinned OVTR Q0 on the complete Phase30 public TRAIN video universe.

This is an inference-only, resumable process.  It exports native physical
lineage before the TAO serializer drops candidate rank/frame fields.  The
annotation alias is a symlink to the frozen TRAIN annotation; no bytes are
copied and labels are never joined before model inference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
ARCHIVE = Path("/data2/usr_for_deadline/trackocd_phase83")
TRAIN = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
ALIAS = ARCHIVE / "phase83_train_validation.json"
Q0 = ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"
CSV = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
OVTR_PY = Path("/home/lwr/anaconda3/envs/ovtr/bin/python")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def public_videos() -> list[int]:
    import csv
    with CSV.open(newline="", encoding="utf-8") as f:
        return sorted({int(r["video_id"]) for r in csv.DictReader(f)})


def ensure_alias() -> dict[str, object]:
    if not TRAIN.is_file():
        raise FileNotFoundError(TRAIN)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    target = str(TRAIN.resolve())
    if os.path.lexists(str(ALIAS)):
        if not ALIAS.is_symlink() or str(ALIAS.resolve()) != target:
            raise RuntimeError(f"wrong existing alias {ALIAS}")
    else:
        tmp = ALIAS.with_name(f".{ALIAS.name}.{os.getpid()}.tmp")
        os.symlink(target, tmp)
        os.replace(tmp, ALIAS)
    return {"alias": str(ALIAS), "target": target, "target_sha256": sha(TRAIN), "symlink": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--max-videos", type=int, default=0, help="bounded smoke prefix; 0 is all R-universe videos")
    ap.add_argument("--tag", default="a2_full_q0")
    args = ap.parse_args()
    if not Q0.is_file() or not OVTR_PY.is_file():
        raise FileNotFoundError("pinned Q0 checkpoint or OVTR interpreter missing")
    vids = public_videos()
    if args.max_videos:
        if not 1 <= args.max_videos <= len(vids):
            raise ValueError("--max-videos out of range")
        vids = vids[: args.max_videos]
    alias = ensure_alias()
    run_dir = ARCHIVE / args.tag
    public_dir = OUT / "physical_a2" / args.tag
    if (public_dir / ".done").exists():
        print(json.dumps({"status": "already_done", "run_dir": str(run_dir)})); return
    if run_dir.exists() or os.path.lexists(str(public_dir)):
        raise RuntimeError(f"unfinished/duplicate A2 output exists: {run_dir} {public_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    public_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = public_dir.with_name(f".{public_dir.name}.{os.getpid()}.tmp")
    os.symlink(str(run_dir.resolve()), tmp_link); os.replace(tmp_link, public_dir)
    command = [str(OVTR_PY), str(ROOT / "scripts/iclr27_phase75a/ovtr_native_eval.py"), "--config_file", str(ROOT / "configs/iclr27_phase83/ovtr_q0_public_train.py"), "--dataset_file", "lvis_generated_img_seqs", "--batch_size", "1", "--with_box_refine", "--two_stage", "--pretrained", str(Q0.resolve()), "--score_mode", "base", "--num_workers", "2", "--sampler_lengths", "2", "--score_thresh", *("0.19",) * 7, "--filter_score_thresh", *("0.19",) * 7, "--ious_thresh", *("0.45",) * 7, "--miss_tolerance", *("5",) * 7, "--maximum_quantity", "160", "--output_dir", str(run_dir), "--eval", "track", "--result_path_track", str(run_dir / "teta_results"), "--video_ids", json.dumps(vids), "--native-out", str(run_dir / "native_lineage.jsonl")]
    launched = {"phase": "Phase83", "branch": "A2", "tag": args.tag, "pid": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu": int(args.gpu), "videos": vids, "command": command, "q0_checkpoint": str(Q0.resolve()), "q0_checkpoint_sha256": sha(Q0), "train_annotation_alias": alias, "labels_joined_before_model": False, "public_dev_q1_sealed_accessed": False}
    (public_dir / ".launched").write_text(json.dumps(launched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(command, cwd=ROOT / "third_party/research_refs_phase4n/OVTR/ovtr", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (run_dir / "eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    native = run_dir / "native_lineage.jsonl"; frames = run_dir / "native_lineage.frames.jsonl"; teta = run_dir / "teta_results/tao_track.json"
    valid = proc.returncode == 0 and native.is_file() and frames.is_file() and teta.is_file() and native.stat().st_size > 0 and frames.stat().st_size > 0 and teta.stat().st_size > 0
    if valid:
        observed = set()
        for line in frames.read_text(encoding="utf-8").splitlines():
            if line.strip(): observed.add(int(json.loads(line)["video_id"]))
        valid = observed == set(vids)
    if not valid:
        atomic_json(run_dir / "failure.json", {"status": "A2_Q0_INFERENCE_FAILED", "exit_code": proc.returncode, "native_exists": native.is_file(), "frames_exists": frames.is_file(), "teta_exists": teta.is_file(), "observed_video_count": len(observed) if 'observed' in locals() else None, "requested_video_count": len(vids), "stdout_tail": proc.stdout[-8000:], "command": command})
        raise RuntimeError("A2 Q0 inference failed; evidence retained")
    summary = {"phase": "Phase83", "branch": "A2", "status": "A2_FULL_Q0_LINEAGE_COMPLETE", "tag": args.tag, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "videos": vids, "video_count": len(vids), "q0_checkpoint_sha256": sha(Q0), "native_lineage": str(native), "native_lineage_sha256": sha(native), "frame_trace": str(frames), "frame_trace_sha256": sha(frames), "teta_results": str(teta), "teta_results_sha256": sha(teta), "command": command, "labels_joined_before_model": False, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "train_annotation_alias": alias}
    atomic_json(run_dir / "a2_summary.json", summary)
    (public_dir / ".done").write_text("complete\n", encoding="utf-8")
    atomic_json(OUT / "audit/a2_full_q0_lineage.json", summary)
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": summary["status"], "next_action": "extract only missing corrected DINO and run full-coverage temporal appearance R", "a2": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
