#!/usr/bin/env python3
"""Evaluate corrected Q0/improved physical vectors on the frozen R universe."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, tempfile
from pathlib import Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records

OUT = ROOT / "outputs/iclr27_phase85"; EPISODES = ROOT / "outputs/iclr27_phase30/manifests"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def load_vec(path: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=False); return np.asarray(z["vectors"], np.float32), [str(x) for x in z["keys"]]

def main() -> None:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--candidate", "--improved", dest="candidate", type=Path, default=OUT / "manifests/physical_r_improved_improved_single_anchor_v2_vectors.npz")
    ap.add_argument("--q0", type=Path, default=OUT / "manifests/physical_r_q0_q0_parity_v5_vectors.npz")
    ap.add_argument("--candidate-name", default="improved")
    ap.add_argument("--output-tag", default="physical_r_comparison")
    args = ap.parse_args()
    table = load_frozen_tracks(); improved, keys = load_vec(args.candidate); q0, q0_keys = load_vec(args.q0)
    if keys != q0_keys: raise RuntimeError("Q0/candidate adapter key order mismatch")
    if improved.shape != q0.shape or improved.shape[0] != len(PREFIXES) or improved.shape[2] != 768: raise RuntimeError(f"bad vector shape {improved.shape} vs {q0.shape}")
    index = {k: i for i, k in enumerate(keys)}; fold_rows=[]
    for fold in range(4):
        manifest = json.loads((EPISODES / f"episode_manifest_f{fold}.json").read_text(encoding="utf-8"))
        val = sorted({str(r["query_track_key"]) for r in manifest.get("records", []) if r.get("split") == "val" and str(r.get("query_track_key")) in index})
        vids = np.asarray([table.metadata[k]["video"] for k in val]); cats = np.asarray([table.metadata[k]["category"] for k in val])
        for pi, prefix in enumerate(PREFIXES):
            records=[]
            for i, q in enumerate(val):
                ci = [j for j in range(len(val)) if j != i and vids[j] != vids[i]]
                candidates = [val[j] for j in ci]; positives = [val[j] for j in ci if cats[j] == cats[i]]; negatives = [val[j] for j in ci if cats[j] != cats[i]]
                if not positives or not negatives: continue
                qi=index[q]; scores=[float(improved[pi,qi] @ improved[pi,index[c]]) for c in candidates]; raw_scores=[float(q0[pi,qi] @ q0[pi,index[c]]) for c in candidates]
                records.append({"query_key":q,"category":int(cats[i]),"video":int(vids[i]),"candidates":candidates,"positives":positives,"negatives":negatives,"scores":scores,"raw_scores":raw_scores})
            mm=score_records(records); fold_rows.append({"fold":fold,"prefix":prefix,"metrics":mm,"validation_queries":len(val),"manifest":str((EPISODES/f"episode_manifest_f{fold}.json").resolve()),"manifest_sha256":sha256(EPISODES/f"episode_manifest_f{fold}.json")})
    aggregate={str(p):aggregate_fold_metrics([x["metrics"] for x in fold_rows if x["prefix"]==p]) for p in PREFIXES}; p16=aggregate["16"]
    folds16=[x["metrics"] for x in fold_rows if x["prefix"]==16]
    gate={"p16":{k:p16[k] for k in ("r1","raw_r1","map","raw_map","hard_negative_gap","raw_hard_negative_gap","unsafe_flip_count")},"folds_non_decreasing_both":sum(int(m["r1"]>=m["raw_r1"] and m["map"]>=m["raw_map"]) for m in folds16),"unsafe_flip_count":p16["unsafe_flip_count"],"status":"PHYSICAL_TO_R_DIAGNOSTIC"}
    gate["candidate_name"] = args.candidate_name
    result={"schema_version":"trackocd.phase85.physical_r_metrics.v1","phase":"Phase85 P5","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"candidate_name":args.candidate_name,"q0_vectors":str(args.q0.resolve()),"q0_vectors_sha256":sha256(args.q0),"candidate_vectors":str(args.candidate.resolve()),"candidate_vectors_sha256":sha256(args.candidate),"query_denominator":sum(m["queries"] for m in folds16),"prefix":aggregate,"folds":fold_rows,"gate_diagnostic":gate,"same_candidate_order":True,"same_video_exclusion":True,"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"controller_run":False,"sealed_run":False}
    out_stem = args.output_tag
    atomic_json(OUT/f"metrics/{out_stem}.json",result); atomic_json(OUT/f"audit/{out_stem}.json",result); atomic_json(OUT/f"completion/{out_stem}.done",{"status":"DONE","metrics":str((OUT/f"metrics/{out_stem}.json").resolve())}); print(json.dumps({"query_denominator":result["query_denominator"],"p16":p16,"gate":gate},indent=2,sort_keys=True))

if __name__ == "__main__": main()
