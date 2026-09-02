#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/iclr27_phase76a"
TAG="${1:-phase76a_formal1}"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
mkdir -p "$OUT/completion" "$OUT/logs" "$OUT/metrics" "$OUT/validation"
export PYTHONPATH="$ROOT"
"$PY" - "$ROOT" "$TAG" <<'PY'
import datetime as dt, json, os, pathlib, subprocess, sys, tempfile
root = pathlib.Path(sys.argv[1]); tag = sys.argv[2]; out = root/'outputs/iclr27_phase76a/audit'; out.mkdir(parents=True, exist_ok=True)
def run(cmd):
    p = subprocess.run(cmd, text=True, capture_output=True, check=False); return p.stdout.strip()
value = {'phase':'Phase76A','tag':tag,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cwd':str(root),'gpu':run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),'memory':run(['free','-h']),'process_count':len(run(['ps','-eo','pid=']).splitlines()),'disk':run(['df','-h',str(root)]),'gpu_mapping':{'0':4,'1':5,'2':6,'3':7},'worker_count':4,'estimated_peak_rss_per_worker_gb':2.0,'ram_safety_floor':'>=25% free','held_event_accessed_for_model':False,'sealed_accessed':False}
fd,tmp=tempfile.mkstemp(prefix='.phase76a_preflight.',dir=str(out)); os.close(fd)
try:
    pathlib.Path(tmp).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out/'formal_preflight.json')
finally:
    if os.path.exists(tmp): os.unlink(tmp)
PY
pids=()
for pair in '0:4' '1:5' '2:6' '3:7'; do
  fold="${pair%:*}"; gpu="${pair#*:}"; run="${TAG}_f${fold}"
  if [[ -f "$OUT/completion/${run}.done" ]]; then continue; fi
  if [[ -f "$OUT/completion/${run}.launched" ]]; then echo "refusing duplicate launched ${run}" >&2; exit 2; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/iclr27_phase76a/train_relation_fold.py" --fold "$fold" --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" --validation-limit 64 >"$OUT/logs/${run}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
"$PY" - "$ROOT" "$TAG" "$status" <<'PY'
import datetime as dt, json, os, pathlib, subprocess, sys, tempfile
root=pathlib.Path(sys.argv[1]); tag=sys.argv[2]; status=int(sys.argv[3]); out=root/'outputs/iclr27_phase76a/audit'
def run(cmd): return subprocess.run(cmd,text=True,capture_output=True,check=False).stdout.strip()
v={'phase':'Phase76A','tag':tag,'status':'COMPLETE' if status==0 else 'FAILED','exit_code':status,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cwd':str(root),'gpu':run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),'memory':run(['free','-h']),'process_count':len(run(['ps','-eo','pid=']).splitlines()),'disk':run(['df','-h',str(root)]),'held_event_accessed_for_model':False,'sealed_accessed':False}
fd,tmp=tempfile.mkstemp(prefix='.phase76a_postflight.',dir=str(out)); os.close(fd)
try:
 pathlib.Path(tmp).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out/'formal_postflight.json')
finally:
 if os.path.exists(tmp): os.unlink(tmp)
PY
exit "$status"

