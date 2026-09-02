#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/iclr27_phase76r"
mkdir -p "$OUT/completion" "$OUT/pareto" "$OUT/logs"
export PYTHONPATH="$ROOT"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
"$PY" - "$ROOT" <<'PY'
import datetime as dt, json, os, pathlib, subprocess, tempfile
root = pathlib.Path(__import__('sys').argv[1]); out = root/'outputs/iclr27_phase76r/audit'
out.mkdir(parents=True, exist_ok=True)
def snapshot(path):
    def run(cmd):
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return p.stdout.strip()
    value = {
        'phase': 'Phase76R', 'kind': path.stem,
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'cwd': str(root),
        'gpu': run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),
        'memory': run(['free','-h']),
        'process_count': len(run(['ps','-eo','pid=']).splitlines()),
        'disk': run(['df','-h',str(root)]), 'gpu_count': 0,
        'held_event_accessed_for_model': False, 'sealed_accessed': False,
    }
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent)); os.close(fd)
    try:
        pathlib.Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
snapshot(out/'resource_preflight.json')
PY
for fold in 0 1 2 3; do
  if [[ -f "$OUT/completion/pareto_f${fold}.done" ]]; then continue; fi
  if [[ -f "$OUT/completion/pareto_f${fold}.launched" ]]; then
    echo "refusing duplicate launched fold $fold" >&2; exit 2
  fi
  "$PY" "$ROOT/scripts/iclr27_phase76r/pareto_audit_fold.py" --fold "$fold" >"$OUT/logs/pareto_f${fold}.log" 2>&1 &
done
wait
"$PY" - "$ROOT" <<'PY'
import datetime as dt, json, pathlib, tempfile, os, subprocess
root=pathlib.Path(__import__('sys').argv[1]); out=root/'outputs/iclr27_phase76r'; rows=[]
for fold in range(4):
    p=out/'pareto'/f'fold{fold}.json'
    d=json.loads(p.read_text()); rows.extend(d['checkpoints'])
rows.sort(key=lambda x:(x['fold'],x['step']))
fd,tmp=tempfile.mkstemp(prefix='.all_checkpoints.',dir=str(out/'pareto')); os.close(fd)
try:
    with open(tmp,'w') as h:
        for r in rows: h.write(json.dumps(r,sort_keys=True)+'\n')
        h.flush(); os.fsync(h.fileno())
    os.replace(tmp,out/'pareto/all_checkpoints.jsonl')
finally:
    if os.path.exists(tmp): os.unlink(tmp)
safe=[r for r in rows if r.get('safe_window')]
payload={'phase':'Phase76R','checkpoint_count':len(rows),'safe_window_count':len(safe),'window_found':bool(safe),'decision':'PHASE76R_PARETO_WINDOW_FOUND' if safe else 'PHASE76R_NO_SAFE_FEATURE_ADAPTER_WINDOW','safe_window_definition':{'global_unsafe':0,'legal_unsafe':0,'global_delta_r1':-0.005,'global_delta_map':-0.002,'legal_delta_r1':'>0','legal_delta_map':'>0','mean_raw_adapt_cosine':0.98},'checkpoints':safe}
fd,tmp=tempfile.mkstemp(prefix='.pareto_front.',dir=str(out/'pareto')); os.close(fd)
try:
    with open(tmp,'w') as h: json.dump(payload,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
    os.replace(tmp,out/'pareto/pareto_front.json')
finally:
    if os.path.exists(tmp): os.unlink(tmp)
plot={'phase':'Phase76R','rows':[{'fold':r['fold'],'step':r['step'],'legal_delta_map':r['legal_delta_map'],'global_delta_map':r['global_delta_map'],'legal_delta_r1':r['legal_delta_r1'],'global_unsafe':r['global_unsafe'],'mean_raw_adapt_cosine':r['mean_raw_adapt_cosine'],'global_delta_r1':r['global_delta_r1'],'legal_learned_r1':r['legal_learned_r1']} for r in rows]}
plot_path=out/'pareto/plot_data.json'; fd,tmp=tempfile.mkstemp(prefix='.plot_data.',dir=str(plot_path.parent)); os.close(fd)
try:
    pathlib.Path(tmp).write_text(json.dumps(plot,indent=2,sort_keys=True)+'\n'); os.replace(tmp,plot_path)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
post=out/'audit/resource_postflight.json'; post.parent.mkdir(parents=True,exist_ok=True)
value={'phase':'Phase76R','kind':'resource_postflight','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cwd':str(root),'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,check=False).stdout.strip(),'memory':subprocess.run(['free','-h'],text=True,capture_output=True,check=False).stdout,'process_count':len(subprocess.run(['ps','-eo','pid='],text=True,capture_output=True,check=False).stdout.splitlines()),'disk':subprocess.run(['df','-h',str(root)],text=True,capture_output=True,check=False).stdout.strip(),'gpu_count':0,'held_event_accessed_for_model':False,'sealed_accessed':False}
fd,tmp=tempfile.mkstemp(prefix='.resource_postflight.',dir=str(post.parent)); os.close(fd)
try:
    pathlib.Path(tmp).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,post)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
print(json.dumps({'phase':'Phase76R','checkpoint_count':len(rows),'safe_window_count':len(safe),'decision':payload['decision']},sort_keys=True))
PY
