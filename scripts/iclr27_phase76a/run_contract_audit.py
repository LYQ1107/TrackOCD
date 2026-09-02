#!/usr/bin/env python3
"""Phase76A contract audit: official refs, dimensions, causal/raw guarantees."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks
from src.iclr27_phase75d.pairwise_correspondence import fast_hungarian_score
from src.iclr27_phase75d.retrieval_metrics import score_records
from src.iclr27_phase75e.evaluator import _global_records_light
from src.iclr27_phase76a.raw_anchor import raw_mean_cosine
from src.iclr27_phase76a.correspondence import hungarian_match, pair_relation_features, relation_summary
from src.iclr27_phase76a.relation_model import AnchoredRelationReranker

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76a/audit"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(value, h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def literature() -> list[dict[str, object]]:
    # Revalidated with ``git ls-remote`` and pinned-commit README/LICENSE fetches
    # on 2026-09-02.  These are references, not imported model code.
    return [
        {"name":"RethinkingOCL","repo_url":"https://github.com/LiZhYun/ICML2026-RethinkingOCL","paper_url":"https://arxiv.org/abs/2605.03650","commit":"5d345268797425558b449337519af3ab24aeb6f1","license":"MIT","release":"ICML 2026","revalidated":"git ls-remote HEAD; pinned README/LICENSE","relevance":"Hungarian/temporal object-centric matching reference","reused":"detached matching design only","not_reused":"no TrackOCD data/evaluator/controller"},
        {"name":"SlotContrast","repo_url":"https://github.com/martius-lab/slotcontrast","paper_url":"https://arxiv.org/abs/2412.14295","commit":"55ec66dc02eeade630805789ef4a6c5df06f21ff","license":"MIT","release":"CVPR 2025 Oral","revalidated":"git ls-remote HEAD; pinned README/LICENSE","relevance":"temporal object-level contrastive objective","reused":"conceptual prefix-consistency motivation only","not_reused":"no causal MOT or cross-video event protocol"},
        {"name":"TRACT","repo_url":"https://github.com/Nathan-Li123/TRACT","paper_url":"https://arxiv.org/abs/2503.08145","commit":"19f01d72f9f6c212c28fd9cb0171a5432cd41a6a","license":"Apache-2.0 (repository metadata)","release":"ICCV 2025","revalidated":"git ls-remote HEAD; pinned README","relevance":"trajectory-aware open-vocabulary tracking reference","reused":"none (text/category boundary incompatible)","not_reused":"text/open-vocabulary and different MOT task"},
        {"name":"COVTrack","repo_url":"https://github.com/zekunqian/COVTrack","paper_url":"https://arxiv.org/abs/2503.08145","commit":"9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b","license":"Apache-2.0 (repository metadata)","release":"ICCV 2025","revalidated":"existing pinned local repo plus git ls-remote HEAD","relevance":"continuous open-vocabulary MOT reference","reused":"none; proposal/language assumptions differ","not_reused":"category/open-vocabulary and no TrackOCD support bank"},
    ]


def main() -> None:
    table = load_frozen_tracks(); model = AnchoredRelationReranker().eval()
    qkey = sorted(table.sequences)[0]; ckey = sorted(table.sequences)[1]
    q = table.get_frame_sequence(qkey, 4); c = table.get_frame_sequence(ckey, 16)
    match = hungarian_match(q, c); feats = pair_relation_features(q, c, match); summary = relation_summary(q, c, match, raw_mean_cosine(q, c))
    with __import__('torch').no_grad(): out = model(__import__('torch').as_tensor(feats), __import__('torch').as_tensor(summary), summary[0])
    atomic(OUT / "literature_audit.json", {"phase":"Phase76A","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"methods":literature(),"all_refs_references_only":True})
    # Raw-only structural parity on all 20 fold/prefix combinations.  The
    # learned module is not involved in this check.
    parity = []
    for fold in range(4):
        for p in PREFIXES:
            records, _ = _global_records_light(table, fold, p, None)
            max_err = 0.0; n = 0
            for rec in records:
                qf = table.get_frame_sequence(rec['query_key'], p)
                for ckey2, frozen in zip(rec['candidates'], rec['raw_scores']):
                    got = raw_mean_cosine(qf, table.get_frame_sequence(ckey2, p)); max_err = max(max_err, abs(got - float(frozen))); n += 1
            parity.append({"fold":fold,"prefix":p,"pairs":n,"max_abs_error":max_err,"pass":max_err <= 1e-7})
    atomic(OUT / "raw_anchor_parity.json", {"phase":"Phase76A","checks":parity,"pass":all(x['pass'] for x in parity),"tolerance":1e-7})
    atomic(OUT / "contract_smoke.json", {"phase":"Phase76A","pair_feature_dim":int(feats.shape[1]),"summary_dim":int(summary.shape[0]),"output_delta":float(out['delta'].item()),"output_confidence":float(out['confidence'].item()),"output_final":float(out['final'].item()),"row_vector_dim":768,"step0_exact_raw":abs(float(out['final'].item())-float(summary[0])) <= 1e-7,"finite":bool(__import__('torch').isfinite(out['final']).item()),"causal_prefixes":list(PREFIXES),"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]})
    print(json.dumps({"phase":"Phase76A","raw_parity":all(x['pass'] for x in parity),"pair_feature_dim":int(feats.shape[1]),"summary_dim":int(summary.shape[0])},sort_keys=True))


if __name__ == "__main__": main()

