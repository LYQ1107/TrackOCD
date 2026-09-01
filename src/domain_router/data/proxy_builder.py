from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def split_categories_for_fold(classes_in_domain, seed):
    """Split a target domain's known classes into proxy-known/proxy-novel
    (deterministic seed); both roles are present in the target fold."""
    rng = random.Random(seed)
    cs = sorted(classes_in_domain)
    rng.shuffle(cs)
    half = max(1, len(cs) // 2)
    return set(cs[:half]), set(cs[half:])


def build_p1_folds(seed=1027):
    """P1 source-domain held-out folds.
    Returns list of folds: {target_domain, source_domains, proxy_known,
    proxy_novel, target_positive_ids, target_negative_ids,
    source_positive_ids (for prototypes), category_missing_from_source}."""
    from src.domain_router.data.domain_metadata import (
        load_train_domains, load_train_known_rows,
    )
    vid2ds = load_train_domains()
    rows = load_train_known_rows()
    by_ds = {}
    for r in rows:
        ds = vid2ds.get(r["video_id"], "<none>")
        by_ds.setdefault(ds, []).append(r)
    domains = sorted(by_ds)
    folds = []
    for target in domains:
        source = [d for d in domains if d != target]
        target_rows = by_ds[target]
        source_rows = [r for d in source for r in by_ds[d]]
        classes_target = {r["category_id"] for r in target_rows}
        pk, pn = split_categories_for_fold(classes_target, seed)
        # proxy-known must also exist in source for prototypes
        source_classes = {r["category_id"] for r in source_rows}
        missing = sorted(pk - source_classes)
        pk = pk & source_classes
        if len(pk) < 1 or len(pn) < 1:
            continue
        folds.append({
            "target_domain": target,
            "source_domains": source,
            "proxy_known": sorted(pk),
            "proxy_novel": sorted(pn),
            "target_positive_ids": [r["sample_id"] for r in target_rows if r["category_id"] in pk],
            "target_negative_ids": [r["sample_id"] for r in target_rows if r["category_id"] in pn],
            "source_positive_ids": [r["sample_id"] for r in source_rows if r["category_id"] in pk],
            "source_negative_ids": [r["sample_id"] for r in source_rows if r["category_id"] in pn],
            "category_missing_from_source": missing,
            "target_total_tracks": len(target_rows),
            "target_known_classes": len(classes_target),
        })
    return folds


def freeze_proxy(folds, out_root=None):
    out_root = out_root or PROJECT_ROOT / "data" / "domain_router" / "proxy_protocol"
    (out_root / "folds").mkdir(parents=True, exist_ok=True)
    (out_root / "hashes").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "P1_source_domain_held_out", "seed": 1027, "folds": []}
    for i, fold in enumerate(folds):
        p = out_root / "folds" / f"fold_{i}.json"
        p.write_text(json.dumps(fold, indent=2))
        manifest["folds"].append({"index": i, "file": str(p.relative_to(out_root))})
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_root
