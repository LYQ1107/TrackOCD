#!/usr/bin/env python3
"""Prepare all Phase76A metadata/index caches in one bounded CPU process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76a.candidate_bank import build_banks, save_banks
from src.iclr27_phase76a.pair_cache import build_pair_cache, cache_hash

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "outputs/iclr27_phase30/manifests"
OUT = ROOT / "outputs/iclr27_phase76a/banks"


def main() -> None:
    table = load_frozen_tracks(); summary = []
    for fold in range(4):
        manifest = MANIFESTS / f"episode_manifest_f{fold}.json"; mh = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for split in ("fit", "val"):
            banks = build_banks(manifest, fold, split, table, seed=7600 + fold)
            bank_path = OUT / f"{split}_f{fold}.json"; save_banks(bank_path, banks, manifest_sha256=mh, seed=7600 + fold)
            cache_path = OUT / f"pair_cache_{split}_f{fold}.json"; payload = build_pair_cache(banks, table, cache_path)
            summary.append({"fold":fold,"split":split,"banks":len(banks),"pairs":payload["pair_count"],"bank_sha256":hashlib.sha256(bank_path.read_bytes()).hexdigest(),"pair_cache_sha256":cache_hash(cache_path)})
            print(summary[-1], flush=True)
    (OUT / "prepare_summary.json").write_text(json.dumps({"phase":"Phase76A","summary":summary}, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__": main()

