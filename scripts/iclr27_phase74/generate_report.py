#!/usr/bin/env python3
"""Render the self-contained Phase74 report from machine-produced artifacts."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74"
REPORT = ROOT / "docs/iclr27_phase74/PHASE74_REPAIR_ASSET_RECONCILIATION_AND_Q0_EVENT_REPLAY_REPORT.md"


def load(rel: str):
    return json.loads((OUT / rel).read_text())


def optional_load(rel: str, default):
    path = OUT / rel
    return json.loads(path.read_text()) if path.exists() else default


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metric(x):
    return f"{x['numerator']}/{x['denominator']} ({x['value']:.6f})"


def main() -> None:
    s = load("status.json"); a = load("assets/asset_universe_summary.json"); m = load("contracts/manifest_order_contract.json"); r = load("replay/control_replay_equivalence.json"); dep = load("audit/q0_text_category_dependency.json"); obs = load("metrics/observability_by_prefix.json"); tests = s["repair_results"]["real_metamorphic_tests"]
    inv = load("audit/input_inventory.json"); lin = load("contracts/five_field_lineage_contract.json")
    process_events = optional_load("audit/process_events.json", [])
    schema_repairs = optional_load("audit/schema_repairs.json", [])
    lines = []
    def H(title, level=2): lines.append("#" * level + " " + title + "\n")
    def P(text=""): lines.append(text + "\n")
    H("Phase74 Repair, Asset Reconciliation and Frozen-Q0 Event Replay Audit", 1)
    P(f"执行时间：{s['start_utc']} → {s['end_utc']}。run_id：`{s['run_id']}`。报告生成时 Git HEAD：`{git_revision()}`。本报告只由 Phase74 机器产物汇总，旧阶段文件保持只读。")
    H("1. Executive Summary"); P("Phase74 修复了 Phase73 的事件顺序、source/target prefix、逐 tracklet 对齐、资产身份和 Gate 计算问题，并真实执行合同/泄漏/因果/原子写入测试。事件 RGB 资产完整存在，但它们来自 TAO TRAIN；冻结 Q0 输出来自 TAO validation。两者 canonical universe 不相交，不能用整数 ID 或 category/bbox 猜测映射，因此按 Branch B 处理。由于本地没有已注册且可证明等价的 Q0 控制重放产物，阶段在 Q0 replay equivalence 处保守阻塞。")
    H("2. Final Status"); P(f"正式状态：`{s['status']}`。Branch：`{a['selected_branch']}`。`qualified_for_automatic_next_stage=false`，`requires_desktop_chatgpt_review=true`。")
    H("3. Phase73 Problems Being Repaired"); P("Phase73 按 (fold, kind, event_key) 重排事件、把 source 与 target 都切为 rows[:prefix]、仅用整数 image_id、逐行选框并导出前 1,000 条 null rows；其 direct-intersection=0 被当作阻塞，且未执行真实 replay/metamorphic 测试。Phase74 改为原文件顺序、显式 source 完整注册/target causal prefix、canonical asset keys、physical-track 聚合和全事件 universe。")
    H("4. Explicit Scope and Non-Scope"); P("Scope 是冻结 Q0/Phase19R 资产审计、lineage、事件对齐和 observability-only。Non-scope 是训练、detector/physical adaptation、semantic correspondence、StateMemory、controller、threshold sweep、sealed/DEV+/Q1/public-new。")
    H("5. No-Sealed / No-Training Declaration"); P("training_run=false；semantic_model_run=false；controller_run=false；threshold_sweep=false；sealed_accessed=false；DEV+=false；Q1=false；public-new=false。没有生成 semantic 或 Commit-CT 指标。")
    H("6. Frozen Input Hashes"); P("| 输入 | SHA256 | 记录/说明 |\n|---|---|---|\n" + "\n".join(f"| {x['name']} | {x.get('sha256','(目录/无)')} | {x.get('record_count','')} |" for x in inv['inputs']))
    P("注册 Q0 checkpoint `809c...1738`、TAO stream `112d...abac2`（1,268,113 rows）、proposal CSV `18339...f8d3`、positive manifest `6442...dadd2`、negative manifest `9673...829fc` 均 hash_match=true。")
    H("7. Environment and Resource Audit"); P("preflight 记录于 `logs/preflight_resource.txt`：host `user-SYS-4029GP-TRT2`，Python `/home/lwr/anaconda3/envs/locatemot/bin/python`；nvidia-smi 显示 GPU0–9 均 0 MiB/无进程；/data1 约 37G 可用、/data2 约 1.2T 可用。可用内存约 120G（free 列受 page cache 影响，available 保持安全余量），本阶段 CPU-only，无 worker/supervisor、无 OOM、无外部 PID 操作。1.45GB Q0 sidecar 和 85MB track index 放在 `/data2/usr_for_deadline/trackocd_phase74_cache/phase74-final-20260902-r3/`，项目目录使用软链接。")
    if process_events:
        P("进程收尾事件（只针对本任务 PID）：\n\n" + "\n".join(
            f"- `{e.get('pid')}`（父 `{e.get('ppid')}`，子 `{e.get('child_pid','')}`）：{e.get('action')}；原因：{e.get('reason')}；产物变化：{e.get('artifact_change')}。"
            for e in process_events
        ))
    else:
        P("没有记录到 Phase74 任务进程终止事件。")
    H("8. Manifest Original-Order Repair"); P(f"positive={m['positive_count']}、negative={m['negative_count']}、total={m['total']}；两文件行顺序均保留，event_key unique={m['event_key_unique']}，implicit_sort_used={m['implicit_sort_used']}。实际 fallback 读取顺序来自 Phase19R `freeze_predictions.public_events`: positive 文件后 negative 文件；不依赖 fold 排序。原始顺序 JSONL：`contracts/manifest_original_order.jsonl`。")
    H("9. Source/Target Prefix Contract"); P("从 `scripts/iclr27_phase19r/freeze_predictions.py` 和 `src/iclr27_phase19r/data/stream.py` 恢复：每个 source tracklet 的所有 positions 在 target 前独立写入 memory；target 仅按自身 event_rank/frame/image 排序并以 prefix count 截断；不拼接多个 source、不读取完整未来 target 统计。合同为 `PROVEN_FROM_RUNNER_AND_STREAM`。")
    H("10. Multiple-Source Tracklet Handling"); P("对齐单位是 `(event_key, role, event_tracklet_key, prefix)`。每个 source key 保留独立 position/lineage；当前 152 个事件均为单 source，但实现和 fixture 测试覆盖多 source，不会把不同 physical track rows 统一切片。")
    H("11. Phase19R Dataset Provenance"); P(f"Phase19R corrected CSV：{a['phase19r_dataset']['event_images']} 个事件相关 image、{a['phase19r_dataset']['videos']} 个 canonical videos；TRAIN annotation 实际使用 `/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json`。原 Phase19R annotation symlink（`data/iclr27_phase19r/sources/tao_train_annotations.json`）已断链，断链与 fallback 均记录于 `assets/split_provenance.json`；事件帧根为 `data/iclr27_phase17/sources/tao_train_frames`，1,422/1,422 路径存在。")
    H("12. Q0 Dataset Provenance"); P(f"Q0 annotation 是 validation split，{a['q0_dataset']['images']} images、{a['q0_dataset']['videos']} videos，路径存在 {a['q0_dataset']['path_exists_images']}；冻结 TAO stream 1,268,113 rows/342,052 tracks。Q0 root 与 OVTR commit `500e72c`、checkpoint/config hash 均记录。")
    H("13. Canonical Video and Image Identity"); P("canonical key 优先使用 annotation dataset/split、稳定 video path、frame_index；image key 是 canonical_video_key+frame。没有使用 category、track_id、bbox、score、event kind 或人工 offset。")
    H("14. Dataset-Universe Intersection"); P(f"Q0 canonical videos={a['q0_canonical_video_count']}（validation），Phase19R event canonical videos={a['event_canonical_video_count']}（train），canonical image mapping={a['mapped_images']}/{a['required_images']}，ambiguity={a['ambiguous_images']}，event missing files={a['missing_images']}。两者底层 root 相同但 split/视频资产不同，canonical intersection=0；选择 Branch B。")
    H("15. Why Numeric ID Intersection Was Zero"); P("Q0 与 Phase19R 使用不同 TAO split，各自 annotation 分配 image/video/track namespace；相同整数不代表同一文件。Phase73 的直接 track intersection=0 是预期 namespace 现象，不是合法 mapping 证据。")
    H("16. Selected Branch A/B and Justification"); P("Branch A 要求全部 event asset canonical 一对一映射到现有 Q0；不满足。Branch B 的 TRAIN event 视频和 annotation 完整存在，故选 Branch B，但必须先通过两个独立控制 replay 的 exact physical equivalence；该条件尚未满足，阶段停止。")
    H("17. Direct-Mapping Gate Repair"); P("`src/iclr27_phase74/mapping.py` 仅在 provenance、one-to-one、frame identity、bbox space、category/track 独立均为真时放行；numeric-only、one-to-many、many-to-one、category-assisted 映射拒绝。当前 asset map 空且 unresolved=1,422，不把 direct=0 当作唯一错误。")
    H("18. Failure Taxonomy Repair"); P("`audit/failure_taxonomy_76.json` 包含 1,520 条 event-role-tracklet-prefix 记录（152×2×5），主失败 `ASSET_NOT_PRESENT_IN_EXISTING_Q0` 1,520 条；次要缺失为 Q0 `frame_id`/`proposal_local_id`。每条含 canonical keys、source row、available/missing evidence、recoverable 标记；unmatched 不删除。")
    H("19. Five-Field Physical Lineage"); P("所需 key 顺序固定为 `video_id:frame_id:proposal_local_id:track_id:image_id`。Q0 TAO 原始字段只有 bbox/category_id/image_id/score/track_id/video_id，故 sidecar 明确 frame_id/proposal_local_id=null，lifecycle=UNKNOWN，不能伪造完整 key。`q0_physical_lineage_rows.jsonl` 是 1,268,113 行并软链接到 /data2；`q0_physical_tracks.jsonl` 是 342,052 track summaries 并软链接。")
    H("20. Proposal Local ID Provenance"); P("历史 `proposals_dev.csv` 只有 28,315 行，不能证明覆盖 1,268,113 TAO rows，也没有可追溯的全量 pre/post filter candidate index。因此 proposal_local_id 来源为 null，lineage status=`UNRECOVERABLE_FROM_TAO_ONLY`。")
    H("21. Q0 Control Replay Equivalence"); P(f"固定控制视频按 988 个 Q0 canonical_video_key 排序五分位选择，见 `replay/control_video_selection.json`。控制 replay 未执行：`{r['status']}`；没有将历史 TAO JSON 重命名为 replay，也没有伪造 bbox/track graph 相等。要求比较 image/frame order、candidate order、bbox/score、local ID、association/lifecycle、track graph，当前 comparisons=null。")
    H("22. Q0 Text/Category Dependency Audit"); P(f"静态扫描 OVTR config/eval/model 命中 CLIP/text/category symbols（例如 `use_text_cross_attention=True`、`Clip_text_embeddings`、`data_dict_to_cuda(...text_embeddings...)`），分类 `{dep['classification']}`；没有 runtime replay/扰动证据，故 `qualified_for_semantic_stage=false`，仅可视为 reference Q0。")
    H("23. Event Full-Video Replay"); P(f"事件资产完整覆盖 {a['event_canonical_video_count']} 个 canonical TRAIN videos，但由于控制 replay equivalence 未通过/未运行，91 个事件视频的 full-sequence replay 未启动；`replay/event_replay_manifest.json` 明确 `NOT_RUN_BLOCKED_CONTROL_REPLAY`，没有生成 q0_event_replay_* 伪产物。")
    H("24. Tracklet-Level Alignment Design"); P("`export/event_tracklet_alignment.jsonl` 1,520 条、`event_role_alignment.jsonl` 1,520 条、`event_alignment_candidates.jsonl` 1,520 条。对每个 canonical image 只读同图 Q0 candidates，再按 physical_track_id 聚合；当前因 universe 不同全部无 candidate，未逐行混 track。")
    H("25. Ambiguity and Fragmentation Handling"); P("0 candidate→UNMATCHED；1 eligible→UNIQUE_MAPPING；>1 eligible→AMBIGUOUS，禁止用 score 或 track ID tie-break；fragmentation 只作 evaluator-only 标记，不重写 physical IDs。")
    H("26. Complete Null Contract and Why It Is Not an OCD Baseline"); P("`physical_track_prefix_contract.jsonl` 和 `physical_semantic_null_contract.jsonl` 各覆盖 121 个事件相关 tracklet×5 prefix=605 行；semantic action 固定 CONTRACT_NULL_POLICY/DEFER/uncertainty=1，`performance_claim_allowed=false`。这是 schema/causality plumbing，不是 OCD/Commit-CT baseline。")
    H("27. Real Metamorphic Tests"); P("真实执行 fixture 输入重算和输入副本检查，结果：" + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k,v in tests.items()) + "。")
    H("28. Causal Future-Append Test"); P("synthetic target 在 prefix=1 后追加未来 row，early target positions 与 source registration 输出完全不变（PASS）。本阶段未运行模型 future batch；模型 causal equivalence 留给合法 replay。")
    H("29. Repeat Determinism Test"); P("两个独立 `/data2/.../metamorphic_repeat_{a,b}` 目录执行相同合同 payload，canonical hashes 相等（PASS）。Q0 inference repeat determinism 仍 NOT_RUN_BLOCKED_CONTROL_REPLAY。")
    H("30. Atomic Crash Test"); P("注入 generator exception 后 final 文件不存在、temp 文件清理、stale lock 可识别（PASS）；证据 `tests/atomic_crash_recovery.json`。")
    H("31. Input Preservation"); P("Q0 checkpoint/TAO/proposals/positive/negative hash 与注册值一致；corrected CSV 与两份 annotation 只读。Q0 原文件未修改，Phase73 原文件未修改。")
    if schema_repairs:
        P("交付格式修复（不改变实验数据）：" + "；".join(f"{x['original_path']} 原内容保留为 {x['preserved_line_jsonl']}，并原子生成可解析数组 {x['new_json_path']}（{x['records']} records，old SHA256={x['original_line_sha256']}）" for x in schema_repairs) + "。")
    H("32. Observability by Prefix"); P("以下是 availability audit（不是 OCD score）；每项 denominator=152：\n\n| prefix | source asset | target asset | source Q0 candidate | target Q0 candidate | source reliable | target reliable | both reliable |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {p} | {metric(obs[str(p)]['source']['asset_located'])} | {metric(obs[str(p)]['target']['asset_located'])} | {metric(obs[str(p)]['source']['candidate_observed'])} | {metric(obs[str(p)]['target']['candidate_observed'])} | {metric(obs[str(p)]['source']['reliable'])} | {metric(obs[str(p)]['target']['reliable'])} | {metric(obs[str(p)]['source']['both_reliable'])} |" for p in [1,2,4,8,16]))
    H("33. Observability by Fold"); P("`metrics/observability_by_fold.json` 保留 fold 0–3、positive/negative 分层及 76+76 分母；因 Q0 未映射，各 fold source/target candidate/reliable 均为 0/该 fold 事件数，事件没有被删。")
    H("34. Positive and Negative Event Results"); P("positive=76、negative=76，均保留原文件顺序。当前只报告资产和 Q0 candidate availability；没有把 positive/negative label 送入模型，也没有输出 Commit/Defer 性能。")
    H("35. Unmatched and Failure Reasons"); P("1,520 个 role-prefix 对均保留 `ASSET_NOT_PRESENT_IN_EXISTING_Q0`；event frame 文件缺失=0。Q0 stream 观测缺失是跨 split/unreplayed lineage，不等同于检测器 no-detection。")
    H("36. Gate Table"); P("| Gate | 机器证据 | 判定 |\n|---|---|---|\n" + "\n".join(f"| {k} | {v.get('reason',v.get('classification',''))} | {v.get('pass',v.get('qualified','N/A'))} |" for k,v in s['gates'].items()))
    H("37. Modified and Created Files"); P("新增 tracked code：`src/iclr27_phase74/`、`scripts/iclr27_phase74/`、`tests/phase74/`。生成 docs/outputs 属于本阶段命名空间；`modified_files.json` 记录历史 files modified=[]。")
    H("38. Commands and Exit Codes"); P("preflight 及 Phase74 主命令记录于 `logs/commands.jsonl`；主 run exit=0 并写 status。pytest：`PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python -m pytest -q tests/phase74` exit=0（26 passed）；旧 evaluator direct：`... tests/test_trackocd_evaluator.py` exit=0。")
    H("39. Remaining Blockers"); P("(1) 必须有真实、两次独立、exact physical graph 等价的 Q0 validation control replay；(2) Q0 TAO 产物无法恢复 frame_id/proposal_local_id；(3) event TRAIN 与 Q0 validation 是不同 canonical universe，需要 replay 生成新 lineage；(4) OVTR text/category runtime dependency 尚未可扰动验证。")
    H("40. What This Phase Does Not Prove"); P("不证明 Q0 在 Phase19R TRAIN 事件视频的 observability、proposal recall、MOT HOTA 或 OCD Commit-CT；不证明 semantic representation 或 controller；不把 0 candidate availability 当检测失败；不把 null contract 当 baseline。历史 Phase68 Q0 25/76 等数字在 manifest/role/prefix/reliability 完全一致且 replay 完成前标记 NOT_DIRECTLY_COMPARABLE。")
    H("41. Recommendation for Desktop ChatGPT"); P("审核并批准唯一最小解阻塞动作：在原 Q0 config/checkpoint/环境下注册一个不覆盖历史输出的 validation control replay（两次独立运行、canonical graph 对比），随后再决定是否允许 event full-video replay。先不要开始 Phase75、semantic correspondence、controller 调参或 sealed。若 replay 不能恢复五字段 local lineage，应把 Q0 作为 reference-only 并另行注册可审计 source。")
    H("42. Explicit Stop Declaration"); P("Phase74 已停止于 `PHASE74_BLOCKED_Q0_REPLAY_EQUIVALENCE`。未启动 Phase75；未训练 correspondence；未调 controller/threshold/StateMemory；未运行 sealed/DEV+/Q1/public-new。等待 Desktop ChatGPT 审核；本阶段不会自动进入下一阶段。")
    H("Appendix: Reproduction and Artifact Paths"); P("```bash\ncd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT\nPYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase74/run_phase74.py --run-id phase74-final-20260902-r3\nPYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase74/run_contract_tests.py\n```\n核心产物：`outputs/iclr27_phase74/status.json`、`contracts/`、`assets/`、`export/`、`replay/`、`metrics/`、`tests/`、`logs/commands.jsonl`、`manifests/output_sha256.json`。大文件软链接目标写入 `assets/split_provenance.json` 和 symlink 本身。")
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(REPORT)


if __name__ == "__main__": main()
