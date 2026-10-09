#!/usr/bin/env python3
"""Separate preregistered <=64 Val-image M1 diagnostic; no primary freeze."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children, owned_process_tree_rss, snapshot_instances

PUBLIC = ROOT / "configs/trackocd_core/masa_physical_diagnostic.json"
PRIVATE = ROOT / "outputs/trackocd_core/audit/masa_physical_diagnostic_plan.json"
RUN = ROOT / "outputs/trackocd_core/physical/masa_val4_first16"
ENV = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")
FRAMES = Path("/data3/liuyeqiang/TAO-Amodal/frames")


def worker() -> int:
    started = time.monotonic()
    config, plan = json.loads(PUBLIC.read_text()), json.loads(PRIVATE.read_text())
    result = {"schema_version":"trackocd.core.masa_physical_prediction.v1","status":"RUNNING",
              "started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"real_image_forwards_started":0,"videos":[],
              "config_sha256":sha256_file(PUBLIC),"private_plan_sha256":sha256_file(PRIVATE),
              "boundary":{"gt_or_category_model_input":False,"training":False,"future_model_input":False,
                          "test_access":False,"offline_filtering":False,"extra_weight_download":False,"foreign_process_interference":False}}
    try:
        if sha256_file(PRIVATE) != config["private_plan_sha256"]:
            raise ValueError("Sealed image plan changed")
        deny_network_and_external_children()
        def deny_annotations(event,args):
            if event == "open" and isinstance(args[0],(str,bytes)):
                path=os.fsdecode(args[0])
                if "/TAO-Amodal/annotations/" in path or "/recovered_splits/" in path or path.endswith("/roles.json") or "/frames/test/" in path:
                    raise PermissionError("Prediction worker cannot open GT/role/Test input")
        sys.addaudithook(deny_annotations)
        import numpy as np
        import torch
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model,infer_current_frame
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        model, proof=build_frozen_model()
        digest=hashlib.sha256(json.dumps(proof["native_model_config"],sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        if digest != config["native_model_config_sha256"] or proof["checkpoint_sha256"] != config["native_checkpoint_sha256"]:
            raise ValueError("Frozen native configuration/checkpoint changed")
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite frozen model tensor")
        result["frozen_model"]=proof
        torch.cuda.set_per_process_memory_fraction(min(1.,config["limits"]["max_gpu_bytes"]/torch.cuda.get_device_properties(0).total_memory))
        model.to("cuda:0"); torch.cuda.reset_peak_memory_stats()
        for video in plan["videos"]:
            model.tracker.reset()
            arrays={"image_id":[],"frame_index":[],"frame_offsets":[0],"track_id":[],"boxes":[],"score":[],"det_offsets":[0],"det_boxes":[],"det_score":[]}
            row={"video_id":video["video_id"],"frames":[]}
            result["videos"].append(row)
            for ordinal,image in enumerate(video["images"]):
                relative=Path(image["image_path"]); path=FRAMES/relative
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0]!="val" or not path.resolve().is_relative_to((FRAMES/"val").resolve()):
                    raise ValueError("Unexpected local Val image")
                if sha256_file(path)!=image["image_sha256"]:
                    raise ValueError("Preregistered image changed")
                if result["real_image_forwards_started"]>=64 or time.monotonic()-started>600:
                    raise RuntimeError("Registered image/time ceiling")
                with Image.open(path) as raw:
                    pixels=np.asarray(raw.convert("RGB"),dtype=np.uint8).copy()
                result["real_image_forwards_started"]+=1
                atomic_json(RUN/"prediction_manifest.json",result)
                detections,tracks=infer_current_frame(model,pixels,ordinal,"cuda:0")
                torch.cuda.synchronize()
                d,t=snapshot_instances(detections,False),snapshot_instances(tracks,True)
                row["frames"].append({"ordinal":ordinal,"image_sha256":image["image_sha256"],"detections":d["count"],"tracks":t["count"]})
                arrays["image_id"].append(image["image_id"]); arrays["frame_index"].append(image["frame_index"])
                arrays["track_id"].extend(tracks.instances_id.cpu().numpy().tolist())
                arrays["boxes"].extend(tracks.bboxes.cpu().numpy().tolist()); arrays["score"].extend(tracks.scores.cpu().numpy().tolist())
                arrays["frame_offsets"].append(len(arrays["track_id"]))
                arrays["det_boxes"].extend(detections.bboxes.cpu().numpy().tolist()); arrays["det_score"].extend(detections.scores.cpu().numpy().tolist())
                arrays["det_offsets"].append(len(arrays["det_score"]))
                if torch.cuda.max_memory_reserved()>config["limits"]["max_gpu_bytes"]:
                    raise RuntimeError("Registered GPU ceiling")
            for key in arrays:
                arrays[key]=np.asarray(arrays[key],dtype=np.float32 if key in {"boxes","score","det_boxes","det_score"} else np.int64)
            arrays["boxes"]=arrays["boxes"].reshape(-1,4); arrays["det_boxes"]=arrays["det_boxes"].reshape(-1,4)
            path=RUN/f"video_{video['video_id']:04d}.npz"
            np.savez_compressed(path,**arrays)
            row.update(npz_bytes=path.stat().st_size,npz_sha256=sha256_file(path),npz_filename=path.name)
            atomic_json(RUN/"prediction_manifest.json",result)
            print(json.dumps({"completed_videos":len(result["videos"]),"real_image_forwards_started":result["real_image_forwards_started"]}),flush=True)
        result.update(status="SEALED_BOUNDED_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION",peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),gt_or_role_files_opened=False)
    except Exception as exc:
        result.update(status="BLOCKED_BOUNDED_PHYSICAL_PREDICTION",error=type(exc).__name__+": "+str(exc))
        import traceback; traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic()-started,peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(RUN/"prediction_manifest.json",result)
    return 0 if result["status"].startswith("SEALED") else 1


def parent(commit: str) -> int:
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes
    frozen=subprocess.check_output(["git","show",commit+":configs/trackocd_core/masa_physical_diagnostic.json"],cwd=ROOT)
    if hashlib.sha256(frozen).hexdigest()!=sha256_file(PUBLIC):
        raise ValueError("Diagnostic is not the committed preregistration")
    delivery=json.loads((ROOT/"outputs/trackocd_core/audit/masa_physical_preregistration_delivery.json").read_text())
    if delivery["commit"]!=commit or not delivery["remote_verified"] or delivery["config_sha256"]!=sha256_file(PUBLIC):
        raise ValueError("Verified remote preregistration required before execution")
    if RUN.exists():
        raise ValueError("Preserve existing bounded physical run; no automatic repeat/resampling")
    gpu_query=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=10)
    occupied=set(subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader,nounits"],text=True,timeout=10).split())
    selected=None
    for line in gpu_query.splitlines():
        uuid,used,free,util=[v.strip() for v in line.split(",")]
        if uuid not in occupied and int(used)<100 and int(free)>=9216 and int(util)==0:
            selected=uuid; break
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU with registered headroom")
    RUN.mkdir(parents=True)
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=selected,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLCONFIGDIR=str(RUN/"matplotlib"),
                                    HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
    install=json.loads((ROOT/"outputs/trackocd_core/audit/masa_runtime_install.json").read_text())
    roots=[ENV,RUN,ROOT/"checkpoints/masa_sam_candidate",ROOT/"outputs/trackocd_core/audit/frontend_alternatives_source",ROOT/"outputs/trackocd_core/audit/masa_candidate_source",ROOT/"outputs/trackocd_core/audit/masa_runtime_resolution"]
    roots.extend((ROOT/"outputs/trackocd_core/audit").glob("masa-native-smoke-*"))
    roots.extend(ROOT/p for p in [install["task_temporary_directory"],*install.get("retained_temporary_directories",[])])
    started=time.monotonic(); peak_storage=peak_rss=0; error=None
    print(json.dumps({"status":"START_PREREGISTERED_VAL4_FIRST16","gpu_uuid":selected}),flush=True)
    with (RUN/"worker.log").open("wb") as output:
        process=subprocess.Popen([str(ENV/"bin/python"),str(Path(__file__).resolve()),"--worker"],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                peak_storage=max(peak_storage,allocated_bytes(roots)); peak_rss=max(peak_rss,owned_process_tree_rss(process.pid))
                mem=dict(s.split(":",1) for s in Path("/proc/meminfo").read_text().splitlines())
                if (time.monotonic()-started>600 or peak_rss>4*1024**3 or peak_storage>8*1024**3
                        or allocated_bytes([RUN])>10*1024**2 or int(mem["MemAvailable"].split()[0])<.25*int(mem["MemTotal"].split()[0])):
                    raise RuntimeError("Registered bounded physical resource guard")
                time.sleep(1)
        except BaseException as exc:
            error=str(exc)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM); process.wait(timeout=30)  # Only this freshly created owned worker.
    result={"schema_version":"trackocd.core.masa_physical_supervisor.v1","preregistration_commit":commit,"remote_preregistration_verified":True,
            "worker_returncode":process.returncode,"error":error,"gpu_uuid":selected,"fresh_gpu_query":gpu_query,
            "peak_candidate_allocated_bytes":peak_storage,"peak_worker_tree_rss_bytes":peak_rss,"elapsed_seconds":time.monotonic()-started}
    atomic_json(RUN/"supervisor.json",result)
    print(json.dumps(result),flush=True)
    return 0 if process.returncode==0 and error is None else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--execute",action="store_true"); p.add_argument("--preregistration-commit"); p.add_argument("--worker",action="store_true")
    a=p.parse_args()
    if a.worker:
        raise SystemExit(worker())
    if not a.execute or not a.preregistration_commit:
        raise SystemExit("Explicit execute and remotely verified preregistration commit required")
    raise SystemExit(parent(a.preregistration_commit))
