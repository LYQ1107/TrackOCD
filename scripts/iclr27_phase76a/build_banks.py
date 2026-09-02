#!/usr/bin/env python3
"""Materialise deterministic Phase30 fit/validation candidate-bank metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76a.candidate_bank import build_banks, save_banks

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "outputs/iclr27_phase30/manifests"
OUT = ROOT / "outputs/iclr27_phase76a/banks"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--split", choices=["fit", "val"], default="fit")
    args = ap.parse_args(); table = load_frozen_tracks(); manifest = MANIFESTS / f"episode_manifest_f{args.fold}.json"
    banks = build_banks(manifest, args.fold, args.split, table, seed=7600 + args.fold)
    save_banks(OUT / f"{args.split}_f{args.fold}.json", banks, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), seed=7600 + args.fold)
    print(json.dumps({"phase":"Phase76A","fold":args.fold,"split":args.split,"banks":len(banks),"path":str(OUT/f'{args.split}_f{args.fold}.json')},sort_keys=True))


if __name__ == "__main__": main()

