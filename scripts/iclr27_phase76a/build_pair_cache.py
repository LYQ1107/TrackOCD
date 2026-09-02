#!/usr/bin/env python3
"""Build detached Hungarian-index metadata; frozen feature arrays are not copied."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76a.candidate_bank import load_banks
from src.iclr27_phase76a.pair_cache import build_pair_cache, cache_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76a/banks"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--split", choices=["fit", "val"], default="fit")
    args = ap.parse_args(); table = load_frozen_tracks(); banks = load_banks(OUT / f"{args.split}_f{args.fold}.json")
    payload = build_pair_cache(banks, table, OUT / f"pair_cache_{args.split}_f{args.fold}.json")
    print(json.dumps({"phase":"Phase76A","fold":args.fold,"split":args.split,"pair_count":payload["pair_count"],"sha256":cache_hash(OUT/f'pair_cache_{args.split}_f{args.fold}.json')},sort_keys=True))


if __name__ == "__main__": main()

