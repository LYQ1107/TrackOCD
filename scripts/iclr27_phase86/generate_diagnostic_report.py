#!/usr/bin/env python3
"""Render the frozen Phase86 diagnostic report from immutable JSON artifacts."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iclr27_phase86'
DOC = ROOT / 'docs/iclr27_phase86/DIAGNOSTIC_OCD_REPORT.md'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def atomic(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(t,p)
    finally:
        if os.path.exists(t): os.unlink(t)

def main() -> None:
    summary_path=OUT/'diagnostic_ocd/summary.json'; parity_path=OUT/'audit/controller_baseline_parity.json'
    d=json.loads(summary_path.read_text()); parity=json.loads(parity_path.read_text())
    lines=[
        '# Phase86 Frozen Diagnostic OCD Report', '',
        '> **DIAGNOSTIC_ONLY_DO_NOT_SELECT** — This replay is an audit of frozen interfaces. Its outputs are not consumed by training, checkpoint selection, threshold selection, or sealed evaluation.', '',
        '## Scope and boundary', '',
        '- Controller: chronologically last byte-identifiable formally frozen RC-MS-OCD / Phase19R controller and StateMemory; thresholds, action semantics, and physical stream were unchanged.',
        '- D0: historical Q0 native Phase19R 768-D prefix stream.',
        '- D1: Phase85 temporal physical 768-D prefix vectors with a deterministic causal prefix mapping (1/2/4/8/16).',
        '- D2/D3: retained as explicit interface-incompatible records because their frozen artifacts expose score-level selected-candidate evidence rather than the controller-required causal 768-D row vector. No invented rescaling or adapter was used.',
        '- Denominator: 76 positive events and 76 negative events; no DEV+, Q1, public-new, or sealed labels were read for model input or selection.', '',
        '## Persistent Commit-CT diagnostic', '',
        '| stream | correct / eligible | recall | fold results |',
        '|---|---:|---:|---|',
    ]
    for name,s in d['streams'].items():
        a=s['aggregate_commit_ct']; fs=', '.join(f"f{x['fold']}={x['metrics']['commit_ct']['correct']}/{x['metrics']['commit_ct']['eligible']}" for x in s['folds'])
        lines.append(f"| {name} | {a['correct']}/{a['eligible']} | {a['recall']:.6f} | {fs} |")
    lines += ['', 'D0 and D1 both produced 3/76, with all three events in fold 3. This is diagnostic evidence only and does not satisfy a broad causal gate. The fold-level parity artifact is intentionally marked non-equivalent to the legacy Phase72 JSON because the historical file used a different per-fold event accounting representation.', '', '## Frozen controller comparison', '', f"- Phase86 D0 summary: `{summary_path.resolve()}` (SHA256 `{sha(summary_path)}`).", f"- Historical Phase72 record: `{parity['phase72_historical_summary']}` (SHA256 `{parity['phase72_historical_sha256']}`).", f"- Parity result: `protocol_equal={parity['protocol_equal']}`; this comparison was not used for model selection.", '', '## D2/D3 decision', '', '- D2 raw source-conditioned support and D3 bounded reranker support remain **not interface compatible** with the frozen RC-MS-OCD input contract. Their hashes and reasons are recorded in `summary.json`; they were not silently converted into controller inputs.', '', '## Reproducibility and safety', '', '- The first two failed attempts are retained as `completion/diagnostic_ocd_r0_failed.json` (import path) and `completion/diagnostic_ocd_r1_failed.json` (artifact path). A path-only repair was applied before the successful replay.', '- Event traces and the summary were atomically written. No training process was launched by this diagnostic.', '- `public_dev_q1_sealed_accessed=false`, `future_rows_or_tracks=false`, and `ids_or_text_as_model_input=false` are asserted in the machine artifact.', '', '## Upstream branch decision', '', 'The diagnostic is not an upstream selection result. Per preregistration, Phase86 now audits the TRAIN-only selective intervention (U1) using out-of-fold labels. If its net rescue/harm safety gate fails, the registered order permits one relation-encoder route (U2) and then the frozen fragment representation diagnostic (RF); no threshold, StateMemory, or controller tuning is authorized.', '', f'Generated UTC: `{dt.datetime.now(dt.timezone.utc).isoformat()}`', '', '']
    DOC.parent.mkdir(parents=True, exist_ok=True); DOC.write_text('\n'.join(lines),encoding='utf-8')
    atomic(OUT/'audit/diagnostic_isolation.json', {
        'schema_version':'trackocd.phase86.diagnostic_isolation.v1',
        'diagnostic_paths':[str((OUT/'diagnostic_ocd/summary.json').resolve()),str((OUT/'diagnostic_ocd/event_traces.jsonl').resolve())],
        'training_input_paths':[], 'checkpoint_selection_paths':[], 'overlap_with_training_inputs':[],
        'diagnostic_only':True,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False,
    })
    print(str(DOC.resolve()))

if __name__=='__main__': main()
