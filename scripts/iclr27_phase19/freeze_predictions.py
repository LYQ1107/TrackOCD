"""Freeze final Phase19 candidate/configuration before public label joining."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def sha(path: Path)->str:
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def atomic(path: Path, value):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); os.replace(tmp,path)

def main():
 files=[
  ROOT/'configs/iclr27_phase19/main/ra_ocd.json',ROOT/'configs/iclr27_phase19/fallback/raw_controller.json',
  ROOT/'outputs/iclr27_phase19/manifests/fold_manifest.json',ROOT/'outputs/iclr27_phase19/manifests/fallback_selection.json',
  ROOT/'outputs/iclr27_phase19/checkpoints/final_fallback_a_best.pt',
  ROOT/'src/iclr27_phase19/models/ra_ocd.py',ROOT/'src/iclr27_phase19/runtime/state_machine.py',
  ROOT/'src/iclr27_phase19/data/stream.py',ROOT/'src/iclr27_phase19/training/train_rollout.py',
  ROOT/'src/iclr27_phase19/evaluation/evaluate.py',ROOT/'src/iclr27_phase19/evaluation/score_public.py',
 ]
 assert all(p.is_file() for p in files),[str(p) for p in files if not p.is_file()]
 freeze={"protocol":"trackocd_iclr27_phase19_prediction_freeze","candidate":"final_fallback_a",
         "true_novel_labels_joined":False,"devplus_q1_accessed":False,"public_event_membership_changed":False,
         "files":{str(p.relative_to(ROOT)):sha(p) for p in files}}
 blob=json.dumps(freeze,sort_keys=True).encode(); freeze['freeze_sha256']=hashlib.sha256(blob).hexdigest()
 atomic(ROOT/'outputs/iclr27_phase19/manifests/prediction_freeze.json',freeze)
 marker=ROOT/'outputs/iclr27_phase19/completion/public_predictions.frozen'; tmp=marker.with_name(marker.name+'.tmp'); tmp.write_text(freeze['freeze_sha256']+'\n'); os.replace(tmp,marker)
 print(json.dumps(freeze,indent=2))

if __name__=='__main__':main()
