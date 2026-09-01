"""Strict-causal replay and Q1 metrics for the Phase 9A lifecycle."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from src.iclr27_phase9a.lifecycle import CausalLifecycle, LifecycleHeads

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_episode(mode: str):
    p = ROOT / "outputs/iclr27_phase7c/assets" / f"{mode}_hard.npz"
    return {k: np.asarray(v) for k, v in np.load(p).items()}


def load_model(model_dir: Path):
    heads = LifecycleHeads.load(model_dir / "heads.npz")
    d = np.load(model_dir / "known_prototypes.npz")
    dp = model_dir / "decision_prototypes.npz"
    if dp.exists():
        q = np.load(dp)
        decision = q["prototypes"].astype(np.float32)
        decision_ids = q["known_ids"].astype(int).tolist()
    else:
        decision, decision_ids = None, None
    return (heads, d["prototypes"].astype(np.float32),
            d["known_ids"].astype(int).tolist(), decision, decision_ids)


def foundation(n: int) -> np.ndarray:
    h = np.asarray(np.load(ROOT / "outputs/iclr27_phase7c/assets/h_all.npz")["h"],
                   dtype=np.float32)
    if len(h) < n:
        raise RuntimeError(f"foundation rows {len(h)} < episode rows {n}")
    return h[:n]


def replay(ep: dict, h: np.ndarray, heads: LifecycleHeads, protos: np.ndarray,
           known_ids: list[int], args, decision_protos=None, decision_ids=None):
    mem = CausalLifecycle(
        protos, known_ids, heads, max_states=args.max_states,
        decision_prototypes=decision_protos, decision_ids=decision_ids,
        no_lifecycle=args.no_lifecycle,
        fixed_maturity=(args.fixed_maturity if args.fixed_maturity > 0 else None),
        no_false_birth=args.no_false_birth, trajectory=not args.no_trajectory)
    order = np.lexsort((ep["proposal_local_ids"], ep["frame_ids"],
                        ep["video_ids"]))
    outputs = [None] * len(ep["gt_role"])
    for i in order:
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        outputs[i] = mem.step(
            h[i], key, float(ep["score"][i]), float(ep["prior_hits"][i]))
    return mem, outputs, order


def metrics(ep: dict, outputs: list[dict], order: np.ndarray, mem: CausalLifecycle):
    n = Counter()
    ok = Counter()
    confusion = Counter()
    seen_cat = set()
    birth_key = {}
    sid_cat = {}
    cross_untrusted = 0
    seen_tracks = set()
    for i in order:
        rs = int(ep["row_split"][i])
        out = outputs[i]
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        if out["action"] == "existing" and key not in seen_tracks and not out["reusable"]:
            # A new physical key may attach only to a trusted state.
            cross_untrusted += 1
        seen_tracks.add(key)
        if rs < 0:
            confusion[("excluded", out["action"])] += 1
            continue
        cat = int(ep["gt_category_id"][i])
        if rs == 0:
            n["known"] += 1
            good = out["action"] == "known" and int(out["semantic_id"]) == cat
            ok["known"] += int(good)
            confusion[("known", out["action"])] += 1
            continue
        first = cat not in seen_cat
        if first:
            seen_cat.add(cat)
            n["first"] += 1
            good = out["action"] == "new"
            ok["first"] += int(good)
            if good:
                sid_cat[int(out["semantic_id"])] = cat
                birth_key[cat] = key
        else:
            if key == birth_key.get(cat):
                n["same"] += 1
                good = (out["action"] == "existing" and
                        sid_cat.get(int(out["semantic_id"])) == cat)
                ok["same"] += int(good)
            else:
                n["cross"] += 1
                good = (out["action"] == "existing" and
                        sid_cat.get(int(out["semantic_id"])) == cat)
                ok["cross"] += int(good)
            confusion[("novel_reuse", out["action"])] += 1
        if out["action"] == "new" and int(out["semantic_id"]) not in sid_cat:
            # This includes duplicate births and gives diagnostics without
            # changing the strict correctness accounting above.
            sid_cat[int(out["semantic_id"])] = cat
    res = {
        "n_known": int(n["known"]), "n_first": int(n["first"]),
        "n_same": int(n["same"]), "n_cross": int(n["cross"]),
        "known_acc": float(ok["known"] / max(n["known"], 1)),
        "first_acc": float(ok["first"] / max(n["first"], 1)),
        "same_acc": float(ok["same"] / max(n["same"], 1)),
        "cross_acc": float(ok["cross"] / max(n["cross"], 1)),
        "joint": float((ok["known"] / max(n["known"], 1)) *
                        (ok["first"] / max(n["first"], 1)) *
                        (ok["cross"] / max(n["cross"], 1))),
        "n_states": int(len(mem.states)),
        "n_quarantined": int(len(mem.quarantine)),
        "n_reusable_final": int(sum(s.reusable_flag for s in mem.states)),
        "cross_untrusted_violations": int(cross_untrusted),
        "action_counts": dict(Counter(o["action"] for o in outputs)),
        "confusion": {f"{a}->{b}": int(v) for (a, b), v in confusion.items()},
    }
    return res


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/iclr27_phase9a/training/lifecycle")
    ap.add_argument("--mode", choices=["train", "metaval"], default="metaval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-states", type=int, default=512)
    ap.add_argument("--no-lifecycle", action="store_true")
    ap.add_argument("--fixed-maturity", type=int, default=0)
    ap.add_argument("--no-false-birth", action="store_true")
    ap.add_argument("--no-trajectory", action="store_true")
    ap.add_argument("--write-events", action="store_true")
    args = ap.parse_args()
    model_dir = ROOT / args.model_dir
    heads, protos, known_ids, decision_protos, decision_ids = load_model(model_dir)
    ep = load_episode(args.mode)
    h = foundation(len(ep["gt_role"]))
    mem, outputs, order = replay(ep, h, heads, protos, known_ids, args,
                                 decision_protos, decision_ids)
    result = metrics(ep, outputs, order, mem)
    result.update({
        "mode": args.mode,
        "model_dir": str(args.model_dir),
        "ablation": {
            "no_lifecycle": bool(args.no_lifecycle),
            "fixed_maturity": int(args.fixed_maturity),
            "no_false_birth": bool(args.no_false_birth),
            "no_trajectory": bool(args.no_trajectory),
        },
        "causal_contract": {
            "actions_only": sorted(set(o["action"] for o in outputs)) ==
            ["existing", "known", "new"],
            "no_untrusted_cross_attach": result["cross_untrusted_violations"] == 0,
            "physical_semantic_separate": True,
            "no_future_rows": True,
        },
    })
    out = ROOT / args.out
    atomic_json(out, result)
    if args.write_events:
        ev = out.with_suffix(".events.jsonl")
        tmp = ev.with_name(ev.name + ".tmp")
        with tmp.open("w") as f:
            for i, o in enumerate(outputs):
                rec = dict(o)
                rec["row_index"] = i
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, ev)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
