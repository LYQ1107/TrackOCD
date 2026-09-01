"""Architecture B: amortized assign-or-create state inference.

Same legal episodic protocol as Architecture A, but the existing-state
matching is temperature-scaled attention over an online prototype memory and
the create decision is amortized by a learned head over the trajectory
representation + physical-stream reliability features.  No Gaussian
posterior, no rho threshold, no 3-way KNOWN/EXISTING/NEW branches.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase8a.model.adapter import (
    CausalTrajectoryAdapter,
    TorchSemanticStateSet,
)
from src.iclr27_phase8a.model.create_head import CreateHead
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase8a.training.train_bsp import (
    chunk_rows,
    compute_centroids,
    eval_metaval as _eval_metaval_a,
    load_assets,
    video_ranges,
)


def phys_vec(score, prior, age, device):
    return torch.tensor([
        float(score),
        min(float(prior), 20.0) / 20.0,
        min(float(age), 50.0) / 50.0,
        1.0 - float(score),
    ], device=device)


def replay_rows(adapter, create_head, states, rows, ep, z_all, device, args,
                known_slot_of_class, track_state, track_count, class_slot,
                class_first, opt=None, accum=None, metrics=None):
    losses = []
    for ri in rows:
        rs = int(ep["row_split"][ri])
        cat = int(ep["gt_category_id"][ri])
        key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
        prev = track_state.get(key)
        if prev is None:
            prev = adapter.new_state()
        z = torch.from_numpy(z_all[ri]).to(device).unsqueeze(0)
        h, state_new = adapter(z, prev)
        h = h[0]
        age = int(track_count.get(key, 0) + 1)
        track_count[key] = age
        track_state[key] = state_new.detach()
        w = 1.0 if args.frame_level else float(age)

        is_known = rs == 0
        is_novel_target = rs == 1 or (
            rs == 0 and int(cat) not in known_slot_of_class)
        target = None
        target_slot = None
        force_spawn = False
        if is_known and int(cat) in known_slot_of_class:
            target = known_slot_of_class[cat]
            target_slot = target
        elif is_novel_target and not args.known_only:
            if cat not in class_first:
                class_first.add(cat)
                target = states.n
                force_spawn = True
            elif cat in class_slot:
                target = class_slot[cat]
                target_slot = target
            else:
                target = states.n
                force_spawn = True

        scores = states.log_scores(h, w)
        best_sim = scores.max() if states.n else torch.zeros(
            (), device=device)
        phys = phys_vec(ep["score"][ri], ep["prior_hits"][ri], age, device)
        # Existing attention scores are temperature-scaled cosine logits.
        # Put the amortized create score in the same units before competing
        # or applying the birth/attach margins.
        create_logit = args.temp * create_head(h, phys, best_sim)
        logits = states.logits(h, w, create_logit.reshape(1))
        pred = int(torch.argmax(logits))
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target], device=device))
            if is_known:
                loss = args.w_known * loss
                margin = torch.clamp(
                    create_logit - logits[target] + args.m_known, min=0.0)
                loss = loss + args.w_known_margin * margin
            else:
                loss = args.w_novel * loss
                if force_spawn:
                    margin = torch.clamp(
                        best_sim - create_logit + args.m_birth, min=0.0)
                    loss = loss + args.w_birth_margin * margin
                else:
                    margin = torch.clamp(
                        create_logit - logits[target] + args.m_attach,
                        min=0.0)
                    loss = loss + args.w_attach_margin * margin
            if metrics is not None:
                if is_known:
                    metrics["known"] += (pred == target)
                    metrics["n_known"] += 1
                elif force_spawn:
                    metrics["first"] += (pred == states.n)
                    metrics["n_first"] += 1
                else:
                    metrics["attach"] += (pred == target)
                    metrics["n_attach"] += 1
        elif rs < 0:
            if pred == states.n and states.n > 0:
                pen = torch.clamp(
                    logits[states.n] - logits[:states.n].max() +
                    args.fp_margin, min=0.0)
                loss = args.w_fp * pen
        if loss is not None:
            losses.append(loss)

        if target_slot is not None:
            states.assign(target_slot, h, w)
        elif force_spawn:
            slot = states.spawn(h, w)
            if slot is not None:
                class_slot[cat] = slot
        elif rs < 0:
            states.decide(h, w, create_logit)

        if opt is not None and len(losses) >= accum:
            torch.stack(losses).sum().backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(create_head.parameters()),
                args.grad_clip)
            opt.step()
            opt.zero_grad()
            losses = []
    if opt is not None and losses:
        torch.stack(losses).sum().backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(create_head.parameters()),
            args.grad_clip)
        opt.step()
        opt.zero_grad()


def run_epoch(adapter, create_head, ep, z_all, split, device, args, rng,
              train=True):
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_all, device)
    ranges = video_ranges(ep)
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    if train:
        rng.shuffle(vids)
    chunks = chunk_rows(ep, ranges, vids, args, rng, train)
    states = TorchSemanticStateSet(
        dim=args.dim, max_slots=args.max_slots, sigma2=1.0,
        score_mode="cosine", cosine_temp=args.temp,
        no_update=args.no_state_update).to(device)
    track_state = {}
    track_count = {}
    metrics = defaultdict(float)
    for chunk in chunks:
        states.reset()
        class_slot = {}
        class_first = set()
        known_slot_of_class = {}
        vis = set(split["train_visible"])
        if train:
            kvis = min(args.visible_known_per_chunk,
                       len(split["train_visible"]))
            vis = set(rng.sample(sorted(split["train_visible"]), kvis))
        mu_k = mu[[known_ids.index(c) for c in sorted(vis)]]
        cnt_k = cnt[[known_ids.index(c) for c in sorted(vis)]]
        states.init_known(mu_k.detach(), cnt_k.detach())
        for j, c in enumerate(sorted(vis)):
            known_slot_of_class[c] = j
        replay_rows(
            adapter, create_head, states, chunk, ep, z_all, device, args,
            known_slot_of_class, track_state, track_count, class_slot,
            class_first, opt=args.optimizer if train else None,
            accum=args.grad_accum if train else None,
            metrics=metrics if train else None)
    return {
        "n_known": int(metrics["n_known"]),
        "n_first": int(metrics["n_first"]),
        "n_attach": int(metrics["n_attach"]),
        "known_acc": metrics["known"] / max(metrics["n_known"], 1),
        "first_acc": metrics["first"] / max(metrics["n_first"], 1),
        "attach_acc": metrics["attach"] / max(metrics["n_attach"], 1),
    }


def eval_metaval(adapter, create_head, ep, z_all, split, device, args):
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_all, device)
    states = TorchSemanticStateSet(
        dim=args.dim, max_slots=args.max_slots, sigma2=1.0,
        score_mode="cosine", cosine_temp=args.temp,
        no_update=args.no_state_update).to(device)
    vis = sorted(set(split["train_visible"]))
    states.init_known(
        mu[[known_ids.index(c) for c in vis]].detach(),
        cnt[[known_ids.index(c) for c in vis]].detach())
    known_slot = {c: j for j, c in enumerate(vis)}
    ranges = video_ranges(ep)
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    chunks = chunk_rows(ep, ranges, vids, args, rng=None, train=False)
    track_state = {}
    track_count = {}
    class_first = set()
    class_slot = {}
    slot_class = {}
    slot_birth = {}
    n = {"known": 0, "first": 0, "same": 0, "cross": 0}
    ok = {"known": 0, "first": 0, "same": 0, "cross": 0}
    with torch.no_grad():
        for chunk in chunks:
            for ri in chunk:
                rs = int(ep["row_split"][ri])
                if rs < 0:
                    continue
                cat = int(ep["gt_category_id"][ri])
                key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
                prev = track_state.get(key)
                if prev is None:
                    prev = adapter.new_state()
                z = torch.from_numpy(z_all[ri]).to(device).unsqueeze(0)
                h, state_new = adapter(z, prev)
                h = h[0]
                age = int(track_count.get(key, 0) + 1)
                track_count[key] = age
                track_state[key] = state_new.detach()
                w = 1.0 if args.frame_level else float(age)
                scores = states.log_scores(h, w)
                best_sim = scores.max() if states.n else torch.zeros(
                    (), device=device)
                phys = phys_vec(ep["score"][ri], ep["prior_hits"][ri], age,
                                device)
                create_logit = args.temp * create_head(h, phys, best_sim)
                logits = states.logits(h, w, create_logit.reshape(1))
                pred = int(torch.argmax(logits))
                if pred == states.n:
                    slot = states.spawn(h, w)
                    if rs == 1 and slot is not None:
                        slot_class.setdefault(slot, cat)
                        slot_birth.setdefault(slot, key)
                        class_slot.setdefault(cat, slot)
                    if rs == 0:
                        n["known"] += 1
                    elif rs == 1:
                        if cat not in class_first:
                            class_first.add(cat)
                            n["first"] += 1
                            ok["first"] += True
                        else:
                            ts = class_slot.get(cat)
                            if ts is not None and key == slot_birth.get(ts):
                                n["same"] += 1
                            else:
                                n["cross"] += 1
                else:
                    slot = pred
                    prov = int(states.provenance[slot])
                    states.assign(slot, h, w)
                    if rs == 0:
                        n["known"] += 1
                        ok["known"] += (prov == 0 and slot == known_slot[cat])
                    elif rs == 1:
                        if cat not in class_first:
                            class_first.add(cat)
                            n["first"] += 1
                        else:
                            ts = class_slot.get(cat)
                            correct = (
                                prov == 1 and ts is not None
                                and slot == ts and slot_class.get(slot) == cat)
                            if ts is not None and key == slot_birth.get(ts):
                                n["same"] += 1
                                ok["same"] += correct
                            else:
                                n["cross"] += 1
                                ok["cross"] += correct
    res = {
        "n_known": n["known"], "n_first": n["first"], "n_same": n["same"],
        "n_cross": n["cross"],
        "known_acc": ok["known"] / max(n["known"], 1),
        "first_acc": ok["first"] / max(n["first"], 1),
        "same_acc": ok["same"] / max(n["same"], 1),
        "cross_acc": ok["cross"] / max(n["cross"], 1),
        "n_states": states.n,
    }
    res["joint"] = res["known_acc"] * res["first_acc"] * res["cross_acc"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase8a/training/b_main")
    ap.add_argument("--mode", choices=["hard", "random"], default="hard")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--chunk-videos", type=int, default=10)
    ap.add_argument("--visible-known-per-chunk", type=int, default=6)
    ap.add_argument("--w-known", type=float, default=1.0)
    ap.add_argument("--w-novel", type=float, default=5.0)
    ap.add_argument("--w-fp", type=float, default=0.1)
    ap.add_argument("--fp-margin", type=float, default=1.0)
    ap.add_argument("--w-known-margin", type=float, default=1.0)
    ap.add_argument("--m-known", type=float, default=1.0)
    ap.add_argument("--w-birth-margin", type=float, default=5.0)
    ap.add_argument("--m-birth", type=float, default=1.0)
    ap.add_argument("--w-attach-margin", type=float, default=2.0)
    ap.add_argument("--m-attach", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=512)
    ap.add_argument("--temp", type=float, default=20.0)
    # Architecture B is causal trajectory-level by default.  Keep an
    # explicit frame-level switch only for the ablation; defaulting to True
    # would silently bypass the GRU and violate the Phase 8A representation
    # protocol.
    ap.add_argument("--frame-level", action="store_true", default=False)
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--no-state-update", action="store_true")
    ap.add_argument("--adapt-reg", type=float, default=0.0)
    ap.add_argument("--fp-keep", type=float, default=0.25)
    ap.add_argument("--track-cap-novel", type=int, default=80)
    ap.add_argument("--track-cap-other", type=int, default=80)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--seed", type=int, default=2717)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tr, va, split = load_assets(args.mode)
    if args.smoke:
        for name, ep in (("train", tr), ("val", va)):
            vids = np.unique(ep["video_ids"])[:2]
            m = np.isin(ep["video_ids"], vids)
            for k in ep:
                ep[k] = ep[k][m]
        args.epochs = 1
        args.chunk_videos = 2
        args.visible_known_per_chunk = 2
        args.track_cap_other = 0

    model, _, _ = load_tse(dev)
    z_tr = project(dev, model, tr["feats"].astype(np.float32))
    z_va = project(dev, model, va["feats"].astype(np.float32))
    adapter = CausalTrajectoryAdapter(
        dim=args.dim, rho_init=0.0, sigma2=1.0,
        frame_level=args.frame_level).to(dev)
    create_head = CreateHead(dim=args.dim).to(dev)
    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(create_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    args.optimizer = opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    best = None
    for ep_i in range(1, args.epochs + 1):
        t0 = time.time()
        trm = run_epoch(adapter, create_head, tr, z_tr, split, dev, args, rng,
                        train=True)
        msg = (f"epoch {ep_i}/{args.epochs} {time.time()-t0:.0f}s "
               f"train k={trm['known_acc']:.3f} "
               f"first={trm['first_acc']:.3f} "
               f"attach={trm['attach_acc']:.3f}")
        if ep_i % args.val_every == 0 or ep_i == args.epochs:
            vam = eval_metaval(adapter, create_head, va, z_va, split, dev,
                               args)
            msg += (f" | val k={vam['known_acc']:.3f} "
                    f"first={vam['first_acc']:.3f} "
                    f"cross={vam['cross_acc']:.3f} "
                    f"joint={vam['joint']:.4f} states={vam['n_states']}")
            if best is None or vam["joint"] > best["val"]["joint"]:
                best = {"epoch": ep_i, "val": vam, "train": trm}
                torch.save({
                    "adapter": {k: v.detach().cpu()
                                for k, v in adapter.state_dict().items()},
                    "create_head": {k: v.detach().cpu()
                                    for k, v in create_head.state_dict().items()},
                    "temp": args.temp,
                    "args": vars(args),
                    "epoch": ep_i,
                    "val_metrics": vam,
                }, out / "best.pth")
        print(msg, flush=True)
        torch.save({
            "adapter": {k: v.detach().cpu()
                        for k, v in adapter.state_dict().items()},
            "create_head": {k: v.detach().cpu()
                            for k, v in create_head.state_dict().items()},
            "temp": args.temp,
            "args": vars(args),
            "epoch": ep_i,
            "train_metrics": trm,
        }, out / f"checkpoint_{ep_i:03d}.pth")
        sched.step()
    if best:
        (out / "best.json").write_text(json.dumps(best, indent=2,
                                                  default=float))
        print("BEST", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
