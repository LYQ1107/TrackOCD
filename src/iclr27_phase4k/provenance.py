"""Phase 4K provenance logging for novel semantic memory.

Records, strictly online and causally:

- prototype birth events (video/frame/track/age/score/semantic evidence);
- prototype update events (support, same/cross-track contributor);
- prototype reuse events (compatibility, selected, update);
- association pair events (appearance-only score, semantic contribution,
  final score, assignment result).

Embeddings are stored in a per-run float16 npz and referenced by index so
the JSONL event streams stay small.  GT is never read here (offline
analysis only).
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


class ProvenanceLogger:
    def __init__(self, out_dir: Path, tag: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = out_dir / f"prototype_event_log_{tag}.jsonl"
        self.emb_path = out_dir / f"embeddings_{tag}.npz"
        self.events = []
        self.embeddings = []
        self._z_index = {}
        self.creator = {}          # sem_id -> (video_id, track_id)

    def _zidx(self, z):
        if z is None:
            return -1
        key = z.tobytes()
        if key not in self._z_index:
            self._z_index[key] = len(self.embeddings)
            self.embeddings.append(np.asarray(z, dtype=np.float16))
        return self._z_index[key]

    def log_birth(self, video_id, frame_id, sem_id, track_key, track_age,
                  track_len, det_score, p_known, best_known, known_margin,
                  best_novel, novel_conf, z, action, support_after):
        key = tuple(track_key) if isinstance(track_key, list) else track_key
        self.creator[int(sem_id)] = key
        self.events.append({
            "kind": "birth", "video_id": int(video_id),
            "frame_id": int(frame_id), "sem_id": int(sem_id),
            "track_key": list(track_key) if isinstance(track_key, tuple)
            else track_key,
            "track_age": int(track_age), "track_len": int(track_len),
            "det_score": float(det_score), "p_known": float(p_known),
            "best_known": float(best_known),
            "known_margin": float(known_margin),
            "best_novel": float(best_novel),
            "novel_conf": float(novel_conf),
            "z_idx": self._zidx(z), "action": action,
            "support_after": int(support_after),
        })

    def log_update(self, video_id, frame_id, sem_id, track_key, support_after,
                   compat_best, z):
        key = tuple(track_key) if isinstance(track_key, list) else track_key
        same_track = (self.creator.get(int(sem_id)) == key)
        self.events.append({
            "kind": "update", "video_id": int(video_id),
            "frame_id": int(frame_id), "sem_id": int(sem_id),
            "track_key": list(track_key) if isinstance(track_key, tuple)
            else track_key,
            "support_after": int(support_after),
            "compat_best": float(compat_best),
            "z_idx": self._zidx(z),
            "same_track": int(bool(same_track)),
        })

    def log_reuse(self, video_id, frame_id, sem_id, track_key, compat,
                  selected, triggered_update):
        self.events.append({
            "kind": "reuse", "video_id": int(video_id),
            "frame_id": int(frame_id), "sem_id": int(sem_id),
            "track_key": list(track_key) if isinstance(track_key, tuple)
            else track_key,
            "compat": float(compat),
            "selected": int(bool(selected)),
            "triggered_update": int(bool(triggered_update)),
        })

    def flush(self):
        with open(self.event_path, "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        if self.embeddings:
            np.savez_compressed(
                self.emb_path,
                embeddings=np.stack(self.embeddings).astype(np.float16))
        print(f"provenance: {len(self.events)} events, "
              f"{len(self.embeddings)} embeddings -> {self.event_path}")


class AssociationInterventionLogger:
    """Tracker-side per-decision association attribution."""

    def __init__(self, out_dir: Path, tag: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"association_decisions_{tag}.jsonl"
        self.rows = []

    def log(self, video_id, frame_id, raw_det_idx, appearance_best_idx,
            appearance_best_score, final_best_idx, final_best_score,
            sem_delta_appearance, sem_delta_final, chosen_memo_idx,
            chosen_track_id, assigned_id, conf, ap_track_id, fn_track_id,
            sem_delta_chosen):
        self.rows.append({
            "video_id": int(video_id), "frame_id": int(frame_id),
            "raw_det_idx": int(raw_det_idx),
            "appearance_best_idx": int(appearance_best_idx),
            "appearance_best_score": float(appearance_best_score),
            "final_best_idx": int(final_best_idx),
            "final_best_score": float(final_best_score),
            "sem_delta_appearance": float(sem_delta_appearance),
            "sem_delta_final": float(sem_delta_final),
            "chosen_memo_idx": int(chosen_memo_idx),
            "chosen_track_id": int(chosen_track_id),
            "assigned_id": int(assigned_id),
            "conf": float(conf),
            "ap_track_id": int(ap_track_id),
            "fn_track_id": int(fn_track_id),
            "sem_delta_chosen": float(sem_delta_chosen),
        })

    def finish_frame(self, sem_manager, det_obs, lambda_s):
        """Enrich this frame's decision rows with the semantic identities
        that were active *before* association (causal point)."""
        for r in self.rows:
            raw = r["raw_det_idx"]
            det = det_obs[raw] if 0 <= raw < len(det_obs) else {}
            r["det_novel_id"] = det.get("novel_id")
            r["det_p_known"] = det.get("p_known", 0.0)
            for key, tid in (("ap", r["ap_track_id"]),
                             ("fn", r["fn_track_id"]),
                             ("chosen", r["chosen_track_id"])):
                t = sem_manager.tracks.get(tid) if tid >= 0 else None
                r[f"{key}_novel_id"] = t.novel_id if t is not None else None
                r[f"{key}_belief"] = t.known_belief if t is not None else 0.0
            r["ap_consistency"] = (r["sem_delta_appearance"] / lambda_s
                                   if lambda_s > 0 else 0.0)
            r["fn_consistency"] = (r["sem_delta_final"] / lambda_s
                                   if lambda_s > 0 else 0.0)
            r["chosen_consistency"] = (r["sem_delta_chosen"] / lambda_s
                                       if lambda_s > 0 else 0.0)

    def flush(self):
        with open(self.path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        print(f"association decisions (sem-active): "
              f"{len(self.rows)} -> {self.path}")
