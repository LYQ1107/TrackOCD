#!/usr/bin/env python3
"""Read-only full frozen RPN-cache feasibility; no association/GT/pixels/weights.

A fresh single-CPU child avoids inherited terminal ru_maxrss. This is not a
tracker replay, primary qualification or permission to launch a full experiment.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker():
    import json,time,resource,hashlib
    from pathlib import Path
    import numpy as np
    from src.trackocd_v2.io import sha256_file
    from src.trackocd_core.physical_qualification import completed_video
    from src.trackocd_core.rpn_bytetrack import read_detector_video,detector_digest
    from scripts.trackocd_core.run_rpn_bytetrack_diagnostic import input_barrier
    from scripts.trackocd_core.smoke_masa_amg import memory
    start=time.monotonic()
    root=Path.cwd(); run=root/"outputs/trackocd_core/physical/masa_full_val_annotated"
    config=root/"configs/trackocd_core/masa_physical_qualification.json"
    private=root/"outputs/trackocd_core/audit/masa_physical_qualification_plan.json"
    mpath=run/"prediction_manifest.json"
    manifest_before=sha256_file(mpath)
    assert manifest_before == "8564226a67298040af4a51c66ca6c9e1a72d169fa454329fa5eb09e496c1f592"
    plan=json.loads(private.read_text()); manifest=json.loads(mpath.read_text()); digest=sha256_file(config)
    assert sha256_file(private)==manifest["private_plan_sha256"]=="bb4ae9b1ce346d490021ab2b70a5e35078e38134184a2bed58d66d5af12ac179"
    assert digest==manifest["config_sha256"]
    assert manifest["status"]=="SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM"
    records={r["video_id"]:r for r in manifest["videos"]}
    assert len(records)==len(plan["videos"])==988
    input_barrier()
    count=total=size=at06=at01=maxframes=maxrawbytes=0
    minimum=1.; maximum=sumscore=0.; identity=hashlib.sha256(); hist=np.zeros(20,dtype=np.int64)
    initial_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for v in plan["videos"]:
        mem=memory()
        assert mem["MemAvailable"]>=mem["MemTotal"]*.25
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=256*1024*1024
        assert time.monotonic()-start<120
        row=completed_video(run,v,digest); assert row==records[v["video_id"]]
        path=run/"shards"/row["npz_filename"]; data=read_detector_video(path,v); q=data["det_score"]
        assert sha256_file(path) == row["npz_sha256"]
        assert len(q)==row["detection_rows"]
        total+=len(q); count+=len(v["images"]); size+=row["npz_bytes"]
        if len(q): minimum=min(minimum,float(q.min())); maximum=max(maximum,float(q.max()))
        sumscore+=float(q.astype(np.float64).sum())
        at06+=int((q>=.6).sum()); at01+=int((q>=.1).sum())
        hist+=np.histogram(q,bins=np.linspace(0,1,21))[0]
        identity.update(str(v["video_id"]).encode()+row["npz_sha256"].encode()+detector_digest(data).encode())
        maxframes=max(maxframes,len(v["images"])); maxrawbytes=max(maxrawbytes,sum(a.nbytes for a in data.values()))
        del data
    assert count==manifest["images"]==36375 and total==manifest["total_detection_rows"]==1811677
    assert sha256_file(mpath) == manifest_before
    print(json.dumps({"status":"AVAILABLE_VERIFIED_FULL_CACHED_RAW_DETECTIONS_NOT_REPLAYED_OR_QUALIFIED","videos":988,"images":count,"raw_detections":total,"source_npz_bytes":size,"source_manifest_sha256":sha256_file(mpath),"source_config_sha256":digest,"source_plan_sha256":sha256_file(private),"ordered_source_and_detector_array_identity_sha256":identity.hexdigest(),"max_video_frames":maxframes,"max_single_video_detector_arrays_bytes":maxrawbytes,"score_min":minimum,"score_max":maximum,"score_mean":sumscore/total,"scores_ge_0_6":at06,"scores_ge_0_1":at01,"score_histogram_width_0_05":hist.tolist(),"seconds":time.monotonic()-start,"initial_peak_rss_bytes":initial_peak,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"source_mutated":False,"gt_pixels_weights_test_read":False,"association_or_model_executed":False},indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(ROOT))
        worker()
        return 0
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                          cwd=ROOT, env=env, timeout=150, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
