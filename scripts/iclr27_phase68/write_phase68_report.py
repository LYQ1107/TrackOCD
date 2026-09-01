#!/usr/bin/env python3
"""Write the Phase68 baseline report from frozen Q0/TrackEval artifacts."""
from __future__ import annotations
import json, pathlib, tempfile, os, hashlib

ROOT=pathlib.Path(__file__).resolve().parents[2]
AUD=ROOT/'outputs/iclr27_phase68/audit/full_sequence_baseline.json'
AGG=ROOT/'outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json'
DOC=ROOT/'docs/iclr27_phase67/PHASE68_OVTR_FULL_SEQUENCE_BASELINE_REPORT.md'

def atomic_json(path,obj):
    fd,n=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(n,path)
    finally:
        if os.path.exists(n): os.unlink(n)

def main():
    base=json.loads(AUD.read_text()); agg=json.loads(AGG.read_text())
    base['trackeval']={'status':'COMPLETE','aggregate_path':str(AGG),'aggregate_sha256':hashlib.sha256(AGG.read_bytes()).hexdigest(),'class_summary_count':agg['class_summary_count'],'macro':agg['macro'],'count_sums':agg['count_sums'],'count_weighted_detection_identity':agg['count_weighted_detection_identity'],'run_note':str(ROOT/'outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval/run_note.json'),'gt_protocol':'pinned TrackEval TAO validation.json'}
    atomic_json(AUD,base)
    r=base['recall']; t20=r['topk']['20']; t0=r['topk']['0']; ta=base['trackeval']
    lines=[]
    lines += ['# Phase68 OVTR Full-Sequence MOT Baseline','',
              '状态：COMPLETE（冻结 Q0 资产；不训练、不访问 DEV+/Q1/public/sealed labels）。','',
              '## 结论','',
              'Phase4Q Q0 是本阶段的真实 OVTR full-sequence 基线：本地 checkpoint 已完成 7 个 epoch（每 epoch 15,000 iter 的历史 schedule），预测 JSON 含 1,268,113 条 detection/track rows。原始帧级 proposal recall 在 top-20、IoU≥0.5 为 **0.629993 (71,062/112,798)**，明显强于 Phase57–61 从零 RGB detector。TrackEval TAO 运行已完成，并保留 304 个 class summary/detailed 文件；汇总值明确标注为 per-class macro/count aggregates，不能与 TAO TETA 或 TrackOCD persistent Commit-CT 混同。','',
              '## Frozen inputs and protocol','',
              f'- Q0 checkpoint: `{base["q0_json"]["path"]}`; SHA256 `{base["q0_json"]["sha256"]}` (score-corrected output lineage).',
              f'- Prediction JSON: `{base["q0_json"]["path"]}`; {base["prediction_count"]:,} rows across {base["prediction_images"]:,} images.',
              f'- Proposal recall GT: `{base["gt_json"]["path"]}`; {base["gt_annotations"]:,} non-crowd rows, {base["gt_images"]:,} images.',
              '- Main recall uses OVTR `validation_ours_v1.json`; official TrackEval symlink uses pinned TAO `validation.json` because the vendored TAO adapter requires its sequence format. The two annotation files are not byte-identical (112,798 vs 113,112 annotations), so values are reported separately.',
              '- Score field is the corrected detection score, not bbox y2. Physical track IDs remain bookkeeping only; no semantic module consumes them.',
              '', '## Proposal recall (class agnostic, offline diagnostic)', '']
    for k in ['0','1','5','20','100']:
        x=r['topk'][k]; lines.append(f'- top-{k if k!="0" else "all"}: IoU≥0.3 `{x["thresholds"]["0.3"]["matched_rows"]:,}/{base["gt_annotations"]:,} = {x["thresholds"]["0.3"]["recall"]:.6f}`; IoU≥0.5 `{x["thresholds"]["0.5"]["matched_rows"]:,}/{base["gt_annotations"]:,} = {x["thresholds"]["0.5"]["recall"]:.6f}`; IoU≥0.7 `{x["thresholds"]["0.7"]["matched_rows"]:,}/{base["gt_annotations"]:,} = {x["thresholds"]["0.7"]["recall"]:.6f}`.')
    lines += ['', '## Full-sequence TrackEval output', '', f'- TrackEval output: `{ROOT}/outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval/OVTR_Q0/` ({ta["class_summary_count"]} class summaries plus detailed CSVs).', f'- Macro over class summaries: HOTA `{ta["macro"]["HOTA"]:.6f}`, DetA `{ta["macro"]["DetA"]:.6f}`, AssA `{ta["macro"]["AssA"]:.6f}`, LocA `{ta["macro"]["LocA"]:.6f}`, OWTA `{ta["macro"]["OWTA"]:.6f}`, MOTA `{ta["macro"]["MOTA"]:.6f}`, IDF1 `{ta["macro"]["IDF1"]:.6f}`, IDSW `{ta["macro"]["IDSW"]:.3f}`, Frag `{ta["macro"]["Frag"]:.3f}`.', f'- Exact class-summary count sums: CLR_TP `{ta["count_sums"]["CLR_TP"]:.0f}`, CLR_FN `{ta["count_sums"]["CLR_FN"]:.0f}`, CLR_FP `{ta["count_sums"]["CLR_FP"]:.0f}`, IDTP `{ta["count_sums"]["IDTP"]:.0f}`, IDFN `{ta["count_sums"]["IDFN"]:.0f}`, IDFP `{ta["count_sums"]["IDFP"]:.0f}`, IDSW `{ta["count_sums"]["IDSW"]:.0f}`, Frag `{ta["count_sums"]["Frag"]:.0f}`.', f'- Count-weighted detection recall/precision: `{ta["count_weighted_detection_identity"]["CLR_Re"]:.6f}` / `{ta["count_weighted_detection_identity"]["CLR_Pr"]:.6f}`; identity recall/precision: `{ta["count_weighted_detection_identity"]["IDR"]:.6f}` / `{ta["count_weighted_detection_identity"]["IDP"]:.6f}`.', '- The vendored TAO adapter emits per-class files and does not produce a single combined table in this environment. We do not fabricate a combined score; the JSON aggregate preserves the exact source files and arithmetic.', '', '## Track continuity diagnostic', '', f'- GT-track proxy: `{base["track_continuity_proxy"]["gt_tracks"]}` tracks; mean reliable fraction `{base["track_continuity_proxy"]["mean_reliable_fraction"]:.6f}`; median `{base["track_continuity_proxy"]["median_reliable_fraction"]:.6f}`; mean longest reliable run `{base["track_continuity_proxy"]["mean_longest_reliable_run"]:.3f}`; tracks reliable at least once `{base["track_continuity_proxy"]["tracks_reliable_at_least_once"]}`.', '- This is an IoU≥0.5 frame proxy, not a MOT metric; official HOTA/CLEAR/Identity artifacts above are authoritative for TrackEval.', '', '## Resource/process audit', '', '- Before launch: 125 GiB RAM, approximately 118 GiB available; `/data1` had approximately 106 GiB free; GPUs 0–9 were idle. Phase68 used no GPU for TrackEval.', '- Two duplicate task-owned TrackEval processes were discovered. PID 19045 (compatibility wrapper) was explicitly SIGTERM-ed after process-tree verification; PID 27179 (direct evaluator) was retained and completed. No external process was touched; no broad kill was used. This event is recorded in `research_log.md`.', '- NumPy `np.int`/`np.float` aliases were process-local in the direct evaluator. Vendored TrackEval source was not modified.', '', '## Phase68 decision', '', '- **Baseline accepted for Phase69 initialization**: real full-sequence Q0 output and score lineage are reproducible, and proposal recall is materially stronger than the from-scratch detector route.', '- This is a MOT baseline, not OCD success. No semantic/controller or persistent Commit-CT result is claimed here.', '', '## Reproduction', '', '```bash', 'cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT', '/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase68/reproduce_full_sequence.py', 'python scripts/iclr27_phase68/summarize_trackeval.py', 'python scripts/iclr27_phase68/run_trackeval_direct.py', 'python scripts/iclr27_phase68/write_phase68_report.py', '```', '']
    DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text('\n'.join(lines))
    print(DOC)
if __name__=='__main__': main()
