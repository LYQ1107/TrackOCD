#!/usr/bin/env python3
"""Train one bounded Phase76S HELP/HARM/NEUTRAL router fold."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase76s.evaluator import evaluate_examples, p16
from src.iclr27_phase76s.router import SelectiveRelationRouter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76s"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase76s/checkpoints")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try: torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists(): path.unlink()
    path.symlink_to(target.resolve())


def load_rows(fold: int, split: str) -> list[dict[str, Any]]:
    payload = json.loads((OUT / "examples" / f"examples_f{fold}.json").read_text())
    return payload[split]


def class_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    count = np.bincount([int(r["label"]) for r in rows], minlength=3).astype(np.float32)
    total = float(count.sum()); values = np.where(count > 0, total / np.maximum(3.0 * count, 1.0), 0.0)
    return torch.tensor(values, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--tag", default="phase76s_formal"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--targeted", action="store_true"); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); steps = 100 if args.smoke else (500 if args.targeted else int(args.steps)); visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, visible={visible}")
    torch.set_num_threads(1); device = torch.device(args.device if torch.cuda.is_available() else "cpu"); seed = 767700 + args.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = load_rows(args.fold, "fit"); val_rows = load_rows(args.fold, "val"); x = torch.tensor(np.asarray([r["features"] for r in rows], dtype=np.float32), device=device); y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long, device=device); weights = class_weights(rows).to(device)
    run = f"{args.tag}_{'smoke_' if args.smoke else ('targeted_' if args.targeted else '')}f{args.fold}"; comp = OUT / "completion"; marker = comp / f"{run}.launched"; done = comp / f"{run}.done"; failed = comp / f"{run}.failed"; metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists(): raise RuntimeError(f"already complete {run}")
    if marker.exists() and not args.resume: raise RuntimeError(f"launched marker exists {run}; use new tag or --resume")
    atomic_json(marker, {"phase":"Phase76S","run":run,"fold":args.fold,"pid":os.getpid(),"gpu":args.expected_physical_gpu,"seed":seed,"steps":steps,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    model = SelectiveRelationRouter().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4); start=0
    if args.resume:
        latest=OUT/"checkpoints"/f"{run}_latest.pt"
        if latest.exists():
            try: ck=torch.load(latest,map_location=device,weights_only=False)
            except TypeError: ck=torch.load(latest,map_location=device)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); start=int(ck.get("step",0))
    history=[]; val_history=[]; best_key=None; best_step=0
    def save(step: int, summary: dict[str, Any]) -> Path:
        target=CHECKPOINT_ROOT/f"{run}_step{step:05d}.pt"; atomic_torch(target,{"phase":"Phase76S","model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"fold":args.fold,"seed":seed,"validation":summary,"class_weights":weights.detach().cpu().tolist(),"protocol":"frozen Phase76AR relation; HELP-only router; exact raw fallback"}); link(OUT/"checkpoints"/target.name,target); latest=CHECKPOINT_ROOT/f"{run}_latest.pt"; latest.unlink(missing_ok=True); latest.symlink_to(target.resolve()); link(OUT/"checkpoints"/latest.name,latest); return target
    try:
        for step in range(start+1, steps+1):
            model.train(); logits=model(x); loss=F.cross_entropy(logits,y,weight=weights); optimizer.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0).detach().cpu()); optimizer.step()
            if step==1 or step%100==0 or step==steps: history.append({"step":step,"loss":float(loss.detach().cpu()),"grad_norm":grad,"label_counts":np.bincount(y.detach().cpu().numpy(),minlength=3).tolist()})
            if step%500==0 or step==steps:
                val=evaluate_examples(model,val_rows,device); summary=p16(val); val_history.append({"step":step,"validation":summary}); cp=save(step,summary); key=(summary["unsafe_flip_count"],-summary["map"],-summary["hard_negative_gap"],-summary["r1"])
                if best_key is None or key<best_key: best_key=key; best_step=step; best=CHECKPOINT_ROOT/f"{run}_best.pt"; best.unlink(missing_ok=True); best.symlink_to(cp.resolve()); link(OUT/"checkpoints"/best.name,best)
        final=evaluate_examples(model,val_rows,device); summary=p16(final); payload={"phase":"Phase76S","fold":args.fold,"run":run,"steps":steps,"seed":seed,"fit_examples":len(rows),"val_examples":len(val_rows),"fit_label_counts":np.bincount([int(r["label"]) for r in rows],minlength=3).tolist(),"val_label_counts":np.bincount([int(r["label"]) for r in val_rows],minlength=3).tolist(),"class_weights":weights.detach().cpu().tolist(),"history":history,"validation_history":val_history,"best_step":best_step,"best_checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt"),"latest_checkpoint":str(OUT/"checkpoints"/f"{run}_latest.pt"),"validation":summary,"config":{"architecture":"14->32 LN GELU->3","decision":"HELP argmax else raw","optimizer":"AdamW","lr":5e-4,"steps":steps,"checkpoint_every":500,"validation_every":500},"gpu":args.expected_physical_gpu,"device":str(device),"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}; atomic_json(metrics_path,payload); atomic_json(done,{"phase":"Phase76S","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt")}); print(json.dumps({"phase":"Phase76S","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"p16":summary},sort_keys=True))
    except Exception as exc:
        atomic_json(failed,{"phase":"Phase76S","fold":args.fold,"run":run,"error":repr(exc)}); raise


if __name__ == "__main__": main()
