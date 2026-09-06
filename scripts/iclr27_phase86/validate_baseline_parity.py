#!/usr/bin/env python3
"""Compare the frozen Phase86 D0 replay with the historical Phase72 record."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase86/audit"
D0 = ROOT / "outputs/iclr27_phase86/diagnostic_ocd/summary.json"
HIST = ROOT / "outputs/iclr27_phase72/metrics/phase19r_raw_baseline.json"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main() -> None:
    d0 = json.loads(D0.read_text())['streams']['D0_historical_Q0_RCMSOCD']
    hist = json.loads(HIST.read_text())
    d0_folds = {int(x['fold']): x['metrics']['commit_ct'] for x in d0['folds']}
    hist_folds = {int(x['fold']): x['metrics']['commit_ct'] for x in hist['folds']}
    rows = []
    for fold in sorted(set(d0_folds) | set(hist_folds)):
        rows.append({'fold': fold, 'phase86_correct': d0_folds.get(fold, {}).get('correct'),
                     'phase86_eligible': d0_folds.get(fold, {}).get('eligible'),
                     'phase72_correct': hist_folds.get(fold, {}).get('correct'),
                     'phase72_eligible': hist_folds.get(fold, {}).get('eligible')})
    obj = {
        'schema_version': 'trackocd.phase86.controller_baseline_parity.v1',
        'diagnostic_only': True,
        'phase86_d0_summary': str(D0.resolve()), 'phase86_d0_sha256': sha(D0),
        'phase72_historical_summary': str(HIST.resolve()), 'phase72_historical_sha256': sha(HIST),
        'folds': rows,
        'phase86_aggregate': d0['aggregate_commit_ct'],
        'phase72_aggregate_recorded': {'correct': sum(int(x['metrics']['commit_ct']['correct']) for x in hist['folds']),
                                       'eligible': sum(int(x['metrics']['commit_ct']['eligible']) for x in hist['folds'])},
        'protocol_equal': False,
        'reason_protocol_equal': 'Phase72 historical JSON uses a 24-event-per-fold legacy aggregate while Phase86 replays the frozen 76-event manifest; this is a lineage comparison, not a replacement metric.',
        'public_dev_q1_sealed_accessed': False,
        'used_for_model_selection': False,
    }
    atomic(OUT / 'controller_baseline_parity.json', obj)
    print(json.dumps(obj, indent=2, sort_keys=True))

if __name__ == '__main__': main()
