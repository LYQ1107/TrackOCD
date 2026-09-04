# TrackOCD Research Log

## Phase 8A — Architecture Reset: Causal Semantic State Inference

### 简短研究计划

1. 淘汰 RACC/TOSE/KPOC 等 3-way logit 语义架构，只保留为 baseline。
2. 核验 2025/2026 先验（AGE/TALON/LTC/PACO/DP-BOA）。
3. Architecture A：causal trajectory adapter + Gaussian semantic state
   posterior + posterior-predictive assign-vs-spawn（unified KNOWN/EXISTING
   vs NEW），legal class-held-out episodic 训练（train_visible /
   hidden_train / hidden_val）。
4. meta-val 三连门（known assign / hidden spawn / cross-track reuse）通过后
   完整训练；Q1 DEV + heldout 单次评估；6 项消融；完整报告。
5. 若 A 完整失败最多切一次 Architecture B。

### 日志

- 2026-08-20: 先验核验完成（AGE ICLR26 / TALON CVPR26 / LTC CVPR26 Findings
  均有官方 repo；PACO arXiv 2604.11484 与 DP-BOA ECCV26 arXiv 2607.13504
  暂未找到官方 code，记为 NO_CONFIRMED_OFFICIAL_CODE）。
- 2026-08-20: 冻结特征 BSP 快速验证：sigma2=0.05 时 known assign ~0.98 但
  first/cross 全 0；sigma2=0.001 时 800 槽被 novel 填满、cross reuse 仍 0。
  结论：无 representation adaptation 的纯冻结特征 BSP 过不了 gate C。
- 2026-08-20: 实现 Architecture A（CausalTrajectoryAdapter + 
  TorchSemanticStateSet + 可训练 rho；episodic teacher-forced 训练）。
  关键教训：sigma2 必须小（0.001）才能给出足够宽的 score 动态范围；
  sigma2=0.05 时 score 全部挤在 ~37-70，rho 无法分离 birth。
- 2026-08-20: 6-epoch pilot（sigma2=0.001, rho_init=-100, w_novel=5,
  birth margin 5/2）：meta-val joint（known*first*cross）0.078 → 0.127；
  known 0.725 / first 0.538 / cross 0.325。进入 30-epoch 主训练。
- 2026-08-20: 并行启动 30-epoch bsp_main (GPU1) 与 frame-level (GPU7)、
  cosine-memory (GPU3) 消融；冻结特征消融 (GPU9)。

- 2026-08-20: BSP 主训练已完成 30 epoch；合法 hard meta-val strict replay
  复核为 known=0.759 / first=0.462 / same=0.865 / cross=0.252，
  但 Q1 DEV strict 为 Known=0.294、first=0、reuse=0、cross=0；锁定
  heldout 为 Known=0.169、reuse=0.027、cross=0。meta-val 三连未迁移到
  frozen Q1，A 判定 `ARCHITECTURE_A_FAILED`，不再调 rho/threshold。
- 2026-08-20: 按架构预算切换一次 Architecture B（amortized create head +
  temperature-scaled cosine online memory）。B causal smoke 通过；初始
  pilot 暴露 create-logit 与 existing cosine-logit 尺度不一致（val first
  全 0）。最小修复为将 create head 输出乘同一 `temp`，并把 B 的
  `--frame-level` 默认改为 false，恢复 causal trajectory 表示。
- 2026-08-20: 修复后 B 6-epoch pilot 的最优 epoch=3：meta-val
  Known=0.940 / first=0.846 / same=0.956 / cross=0.338（joint=0.269）。
  真实 Q1 replay：DEV Known=0.412 / first=0.222 / reuse=0.039 / cross=0；
  heldout Known=0.330 / first=0.250 / reuse=0.323 / cross=0.029。因 Known
  仍低于 0.60 且 DEV cross=0，B 也未达到 Phase 8A operating-point gate。
- 2026-08-20: B DEV/heldout causal contract 全通过（无 future/no relabel、
  novel memory legality、即时首帧 action、objectness invariance）；未启动
  B 完整训练，Phase 8A 以 `ARCHITECTURE_A_FAILED` /
  `ARCHITECTURE_B_Q1_OPERATING_POINT_FAILED` /
  `TRACKOCD_NOT_YET_ICLR_LEVEL` 收束。GPU6 为 B pilot 使用；未终止他人
  任务，无 OOM/near-OOM 事件。

## Phase 9A — Causal Semantic State Lifecycle / Maturation (historical exploratory path)

### 研究计划与执行边界

1. 冻结 Phase 6B DSCT physical/objectness、trajectory features、bbox/data
   修复和严格 evaluator；只在 CPU 上训练小型 lifecycle heads，避免大规模
   backbone 训练。
2. 用 `train_hard` 的 visible-known / hidden-novel / false-birth rows 做
   合法 causal episodes；在线状态立即公开 NEW_NOVEL，但只有 learned
   maturity 才能进入 reusable memory。
3. 先做 meta-val 三连，再做一次 Q1 DEV 严格回放；若 Q1 gate 不过，不跑
   heldout 或大训练，不用 Q1 标签调阈值/挑 checkpoint。

### 结果与失败现象

- 2026-08-21: 初版 semantic-only lifecycle（保留历史三路 action 头）在
  max-states=0 诊断上复现 Known=0.628 / First=0.778，但任何共享状态均会
  把 FP-born state 提前当作 reusable；cat 611 的 842→843 cross 始终为 0。
  该三路头仅保留为历史消融，不作为 Phase 9A 架构。
- 2026-08-21: 发现并修复 maturity 训练/推理特征错位：训练使用当前帧
  knownness 与 age_norm，推理使用 birth knownness 与 support_tracks。
  `maturity_matrix` 现在逐轨迹重建 birth score、uncertainty、consistency、
  evidence count 和 support count，且 false-birth 负样本按正事件/年龄分层
  采样；修复后 meta Known 0.819 / First 0.462 / Cross 0.065，但 Q1 Known
  0.556 / CT-Reuse 0。
- 2026-08-21: 按架构要求移除三分类 identity head，改为 binary learned
  knownness evidence + state-only maturity + pairwise reuse。`lifecycle_binary`
  的合法 meta-val 为 Known=0.838 / First=0.692 / Cross=0.188（joint=0.109，
  causal contract 全通过），但 Q1 DEV 为 Known=0.539 / First=0.222 /
  Reuse=0.196 / CT-Reuse=0；known→existing=0.380。Q1 gate 未通过。
- 2026-08-21: Q1 cat 611 审计确认根因：长 FP 轨迹在 causal state 中获得
  learned maturity 后先于真实 novel track 进入共享 memory，造成语义吸收；
  这是 representation/supervision transfer 限制，不通过固定 frame gate、
  prior offset、evaluator 或 Q1 checkpoint 选择修补。

### 合法消融（binary lifecycle）

- meta no-lifecycle: Known 0.785 / First 0.154 / Cross 0.042；fixed-3:
  0.821 / 0.385 / 0.048；no-trajectory: 0.781 / 0.846 / 0.276。
- no-false-birth-training（不提供 FP maturity negatives）meta: Known
  0.832 / First 0.615 / Cross 0.099；Q1 Known=0.539 / CT=0。
- 主 binary Q1 与所有消融均保持 `no_future_rows`、
  `no_untrusted_cross_attach`、physical/semantic separation；无 OOM，未终止
  其他进程。

### 保留/拒绝

- 保留：causal state fields、birth-immediate public IDs、learned maturity
  lifecycle、binary knownness evidence、pairwise reuse head、false-birth
  training protocol、严格 protocol/evaluator。
- 拒绝：历史三分类 action、instant reusable birth、fixed frame maturity、
  known prior/threshold offset、Q1-specific selection；Phase 9A 状态为
  `PHASE9A_Q1_GATE_FAILED`，不宣称 ICLR-level success。

### Phase 9A Stage-1 reliability validation (authoritative, 2026-08-21)

- 复核当前任务要求后，冻结同一个 Phase 8A Architecture-B checkpoint
  `outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth`、同一个 Q1 DEV
  DSCT stream 和 `strict_eval_any`；单独重跑 baseline，结果与 Phase 8A
  已记录值完全一致（Known 0.412104 / CT-Reuse 0）。
- 新增 `replay_b_reliability.py`：只 mask 未可信 online-born novel state
  的跨 physical track attach；birth 行仍立即公开 NEW。每个 state 记录
  prototype/uncertainty/age/evidence_count/trajectory_consistency/
  feature_variance/reliability_score/reuse_allowed。Reliability 使用
  Beta(2,2) 概率先验，每次已接受 prefix evidence 用 trajectory consistency、
  dimension-free feature stability、B-logit certainty 的几何均值更新；没有
  固定 frame 数、Q1 label、future confirmation 或 threshold sweep。
- Stage-1 Q1 DEV：B baseline → B+gate：Known 0.412104→0.414986，First
  0.222222→0.222222，Novel reuse 0.039216→0.039216，CT-Reuse 0→0，
  Known→Existing 0.299712→0.219020，Novel→Known 0.513514→0.504505。
  Gate 阻止 287 次未可信跨轨 attach，但新增 NEW 269→438；CT-Reuse 没有一条
  正确恢复。
- GT 仅用于 replay 后的 contamination join：baseline births=6 true-novel /
  13 known-confusion / 250 FP-noisy；gate candidates=10/23/405；gate 仅
  2 true-novel、6 known-confusion、92 FP-noisy 进入 trusted。完整事件与来源
  对照见 `outputs/iclr27_phase9a/eval/stage1_b_reliability_gate/`。
- 判断：Known 提升仅 +0.00288 且 CT-Reuse 仍为 0，故
  `PHASE9A_STAGE1_FAILED_STOPPED`。不运行 Stage2、heldout、reliability-v2、
  更多 loss 或 threshold tuning；下一步改查 foundation/VLM semantic
  discovery、video TTA、unlabeled-video pretraining、continual open-world。

## Phase 4Z — Trajectory-Level Open-World Routing

### 简短研究计划

1. 复现/固化 O1c oracle 结果与下游冻结路径（d2_joint_v2 TSR + T3/D2 heads）。
2. Q1 dev 全轨迹 routing 机制审计（此前 dump 只在首个决策步中断，本次重建
   31,650 步 full dump）。
3. 搜索 2025/2026 先验（COLOSEO / STE-CapsNet / CEO-TAD / SCOPE / TRACT /
   COVTrack / GOVTrack / NOVA / ROMOT 等）。
4. 基于审计选择路由公式（GRU 证据累加器 + UNRESOLVED），实现强静态/均值池化/
   聚合基线。
5. 400 train + 200 meta-dev genuine-OOV episodes（冻结 O1c 证据），flat 标签
   训练，meta-dev 选阈值，Q1 dev 端到端一次评估。
6. 输出完整 47 节报告 + blocking runner。

### 日志

- 2026-08-15 22:20: Phase 4Z 启动。旧 routing dump 发现只有 1 步/track
  （脚本在首个非 defer 决策处 break，而 Stage C 全部 track 在 age-1 决策）。
  重建 full dump：31,650 steps / 13,468 tracks / 97 aligned。
- 2026-08-15: 机制审计结论：
  - Stage C router `use_defer=False`，所有 aligned track 均在 prefix age 1
    决策；novel routing recall 1/21 ≈ 0.048。
  - 20 个被吸收 novel 中 13 个 LOW_CONF、6 个 UNSTABLE、仅 2 个
    STABLE_HIGH_CONF。
  - single-frame age-1 AUROC ≈0.70；prefix consistency 在 age 8–12 达
    0.66–0.83 → 顺序证据值得做。
- 2026-08-15: 先验审计完成并写入 2025_2026_PLUS_PRIOR_ART.md；选定
  ROUTING_FORMULATION_DECISION：一个因果 GRU 证据累加器（TSER），
  KNOWN/NOVEL/UNRESOLVED。
- 2026-08-15: 建立 frozen-O1c 证据 episodes（active pseudo-known subset +
  pseudo-novel，TRAIN stream 真实 tracklet）。
- 2026-08-15: Pilot 关键协议实验：
  - decision-age sampling（defer 训练过量）→ meta-dev novel RR 仅 0.15–0.52；
  - **flat per-step labels + 推理阈值** → novel RR 0.82–0.87、known RR 0.99；
  - 输入 z-score 归一化反而显著变差（static 0.599→0.296），冻结为不归一化；
  - 去掉 full-48 统计也变差（0.618→0.337），保留 30-d 证据。
  - Pilot meta-dev balanced：singleframe 0.905 / static 0.916 / meanpool
    0.918 / GRU 0.912 / aggregated 0.929（GRU 尚未超过 aggregated，待 full
    data 定论）。
- 2026-08-15: Full-data 关键发现：
  - flat 标签 + 阈值 deferral 后，meta-dev 全部候选大幅提升；GRU 无 dropout
    过拟合（train loss 0.067 vs meta-dev 差），加入 dropout 0.3 + meta-dev
    early stopping 后 GRU 成为最优（known RR 0.997 / novel RR 0.862 /
    balanced 0.930）。
  - **Frozen Stage C L1 probs 作为特征会让新 router 直接复制已知偏置**
    （Q1 dev RR 0.045）；删除后 RR 0.136、absorption 0.909→0.545。
  - 增大 active known set（16/24）与 prefix 24 的 repair 在 meta-dev 略降、
    dev 未转出（RR 0.045 at meta-dev-selected τ）。
  - Q1 dev 最优（gru_nol1, τ=0.45）：Known 0.500 / RN-Acc 0.136 / RR 0.136 /
    absorption 0.545 / unresolved novel 0.318。O1c oracle：0.566 / 0.591 /
    0.955。→ OPEN_WORLD_ROUTING_SUPERVISION_LIMITED，Pareto 未破。
  - Ablation：prefix age 1→8 balanced 0.893→0.930（轨迹证据存在）；shuffle
    顺序后 balanced 0.942–0.944（顺序不必要，TSR state 已编码因果顺序）→
    SEQUENTIAL_KNOWN_EXPLAINABILITY_NOT_SUPPORTED。
  - Q2 cross-frontend（frozen）：Known 0.474 / RN 0.136 / RR 0.182。
  - 报告：docs/iclr27_phase4z/PHASE4Z_COMPLETE_COPYABLE_REPORT.md（47 节，
    含全部 topic docs 汇总）。

## Phase 4O — Open-World Object Frontend Benchmark and Detector Re-entry

### 简短研究计划

1. D0 detector-only benchmark（用 Phase 4N detection population 直接计算）。
2. 检查当前 SimOWT/IDOL detector 训练配置与数据，决定是否可合法 retrain（D1 control）。
3. 重新核验 2025/2026 detector：WeDetect / YOLO-UniOW / YOLOE / OWOBJ / OW-OVD（clone + commit + weights）。
4. 对可行候选跑 dev 帧 detector-only，统一 IoU≥0.5，输出 Novel-Recall–FP Pareto。
5. 通过 pass gate 的 detector 接入 TrackOCD（T0–T3），dev 评估，freeze ≤2，corrected 24-video held-out，memory provenance。
6. 最终报告：docs/iclr27_phase4o/PHASE4O_COMPLETE_COPYABLE_REPORT.md。

### 日志

- 2026-08-08: 开始 Phase 4O。AGENTS.md 已加入工作节奏与 research_log 要求。
- 2026-08-08: D0 detector-only 基准完成（dev/heldout）：novel recall @1 FP/frame =
  dev 0.865 / heldout 0.763；novel recall 0.7 需 FP/frame dev 0.78 / heldout 0.92。
  D1（当前 detector retrain）未执行：COCO-2017 train 数据不在服务器，R-50 backbone
  缺失，完整训练 163k iters 不可行；记录为 RETRAINING_CURRENT_DETECTOR_NOT_EXECUTED。
- 2026-08-08: 外部 detector 候选验证：YOLOE-v8l-seg-pf（HF 权重，prompt-free）与
  WeDetect-Base-Uni（HF 权重，universal proposals）均可实际 inference。
  Detector-only 结果（同一 IoU≥0.5 / PR 协议）：
  - D0 frozen: novel recall 0.865 @ 1 FP/frame (dev)。
  - D2 WeDetect-Uni: novel recall 0.062 @ 1 FP/frame (dev)；novel recall 0.3 需 29.9 FP/frame。
  - D4 YOLOE-PF: novel recall 0.072 @ 1 FP/frame (dev)；novel recall 0.3 需 13.5 FP/frame。
  坐标/IoU sanity check 通过（top proposal 与 GT IoU 0.88/0.94）。两候选均被 D0 支配。
  结论：OFF_THE_SHELF_DETECTORS_INSUFFICIENT；不进入 TrackOCD re-entry；
  下一阶段 TRACKING_AWARE_OBJECTNESS_RESEARCH_REQUIRED。

## Phase 4P — Joint Trainable Detection–Tracking–Open-World Discovery

### 简短研究计划

1. 盘点本地训练数据与合法监督（TAO train / COCO / LVIS / Objects365 / 已有 cache）。
2. 输出 TRAINING_SUPERVISION_PROTOCOL；若无合法训练数据 → JOINT_TRAINING_PROTOCOL_BLOCKED。
3. 轨迹信号审计（TRUE_NOVEL vs PERSISTENT_FP）— 用现有 dev/heldout 数据直接回答 Q1。
4. 搜索 2025/2026 joint MOT backbone（query-based, joint det+track, 权重可跑）。
5. 选择 backbone（优先当前 SimOWT/IDOL 可改造性），smoke test。
6. 若数据+backbone 可行：最小 joint model（TCO）→ pilot → full train → dev → heldout。
7. 若协议阻塞：如实冻结并输出报告。

### 日志

- 2026-08-08: 开始 Phase 4P。先做训练数据盘点与轨迹信号审计。
- 2026-08-08: 发现关键数据 bug：`det_local_id` 每帧重置，旧 detection
  population/cache 只保留每个视频最后一帧（dev 仅 1898 行，tracklet ≤2 帧）。
  已修复为 (video, frame, det_local_id) 键控并重建：
  - det_z_cache_fixed / det_z_cache_heldout_fixed（20/24 视频全部完成）
  - detection_population_dev_fixed.csv 60,098 行
  - detection_population_heldout_corrected_fixed.csv 67,968 行
  - novel tracklet 最大长度 dev 41 / heldout 39（此前最大 2）。
  Phase 4M/4N 旧语义记忆与轨迹结论在修复数据上不可信，需重推。
- 2026-08-08: Q1 frame-online 轨迹信号审计（每行特征仅用当前帧 + 同一
  physical track 的 ≤t-1 前缀，dev 5-fold CV + dev→heldout 迁移）：
  - persistent FP 子集（prior_hits≥2）causal-only: dev CV AUROC 0.8216 /
    heldout transfer 0.8345；static+causal: 0.9130 / 0.8827。
  - LR（dev 训练）在 heldout persistent 子集 novel recall 0.5 时 precision
    1.0、FP kept 0；recall 0.7 时 precision 0.517、FP kept 5.3%。
  - 简单 age/score confirmation heuristic 最高 precision 仅 0.04。
  - 结论：TRAJECTORY_OBJECTNESS_SIGNAL_STRONG（非简单 track confirmation）。
- 2026-08-08: 训练数据盘点更正：官方 TAO train 500 视频 / 18,274 标注帧
  全部在盘（data/raw/tao/frames/train，534,094 帧，59G），此前“帧路径不匹配”
  判断错误（检查基准多套了一层 train/）。COCO train2017 图片仍缺失。
  训练协议：box + physical track ID 可作为 class-agnostic 监督；48 个
  supported-known 语义标签可训练；novel-role 语义标签禁止。
- 2026-08-08: IDOL 10-iter TAO train smoke 跑通（约 1s/iter，单卡 11.2G，
  loss_ce/bbox/giou 正常，model_final.pth 已保存）。同步完成
  OVTR/COVTrack/OVTrack/MOTRv2/MOTIP-2 克隆与 commit 记录。
- 2026-08-08: 【暂停指令】用户要求暂停 Phase 4P 进一步架构实现与长训，
  不得因 smoke 跑通就锁定 SimOWT/IDOL 为最终 backbone；转入：
  (a) cache bug 历史影响审计；(b) corrected trajectory audit 固化；
  (c) OVTR 代码级审计；(d) COVTrack++/COVTrack/VOVTrack/OVTrack 协议再审计；
  (e) backbone 重新决策。TAO train 8 卡导出已停（eval 路径 COCO_PRETRAIN=True
  与 YTVIS 格式不兼容，修复方向为 INPUT.COCO_PRETRAIN False；暂停期间不修）。
- 2026-08-08: 协议再审计完成（docs/iclr27_phase4p/METHOD_PROTOCOL_REAUDIT.md）：
  - 更正旧判断 A：OVTrack/COVTrack/VOVTrack/OVTR 均为 base-only 训练，
    novel 不进监督（OVTrack 用 lvis_v1_train+coco_mask_v1_base.json，
    COVTrack 用 ctao_base.json，OVTR 用过滤后 lvis_clear_75_60.json）；
  - 更正旧判断 B：COVTrack detector 是 OVTrack 同款 DetPro/CLIP 蒸馏
    Faster R-CNN（非 OV-DETR），且 freeze_detector=True 确认；
  - 更正旧判断 C：IDOL 确为 online VIS+ReID，非 persistent track-query；
  - COVTrack++ 官方代码未发布；TCP 只做正向恢复（boost 低置信检测），
    无 persistent-FP suppression；MGA 是特征门控非 FP 抑制；
  - VOVTrack（ICCV25）代码/权重可用：LVIS base DetPro + state-prompt
    检测 + 无标注 TAO 自监督关联，novel TETA val 35.3/test 29.8；
  - OVTR 代码级确认：fp_ratio=0.3 训练期把 inactive queries 重注入
    为伪 FP；CIP 做类别信息传播；无推理期 FP 门控。
- 2026-08-08: Backbone re-decision 草稿完成
  （docs/iclr27_phase4p/PHASE4P_BACKBONE_AND_NOVELTY_REDECISION.md）：
  - 主选 A：OVTR（端到端 persistent query，协议内唯一含 FP 注入与
    per-object 持久状态的方法）；备选 B：COVTrack/VOVTrack 风格
    （frozen detector + online association + Q1 轨迹门控）；
  - IDOL/SimOWT 明确不作为最终主架构（无开词语义/persistent query）；
  - 未启动任何长训/新架构实现；下一最小实验：OVTR TAO eval 可行性
    smoke + Q1 gate 全流 novel recall@FP 曲线重算，待用户确认。
- 2026-08-08: 用户批准三项最小实验。完成：
  - COVTrack 资源下载（/data3/exp/COVTrack/saved_models/）：
    ctao_base.json（490,210 图/1,489,637 标注，strict base-only）、
    ctao_public.pth、detpro_prompt.pt、ovtrack_pair.pth；
  - OVTR 资源下载（/data3/exp/OVTR/）：ovtr_5_frame.pth（239M）、
    iou_neg5_ens.pth、clip_image_embedding_all.pt、validation_ours_v1.json、
    tao_test_burst_v1.json；data/TAO 软链到本地 frames（路径验证通过）；
  - 全流 TCO gate 实验（src/iclr27_phase4p/full_stream_tco_gate.py）：
    * score_only 复现 D0（dev 0.0487 / heldout 0.0222 @1 FP/frame）✓；
    * all_novel（static+causal LR）：dev 0.139 / heldout 0.090 @1 FP/frame
      （2.9×/4.1×）；@recall 0.3 所需 FP/frame dev 12.1→3.4、ho 10.9→6.0；
    * causal 单独弱于 static（全流多为 age-0/1 检测）；static+causal 最优；
    * all_valid（known+novel vs FP）保 known recall 的同时 heldout novel
      recall 0.022→0.063 —— 联合目标必须含 known 保留；
    * 文档：docs/iclr27_phase4p/FULL_STREAM_TCO_GATE_RESULTS.md。
  - OVTR 环境：新建 conda env ovtr（py3.8 + torch 1.10.1+cu113），
    正在安装 mmcv-full 1.3.17/mmdet 2.23/CLIP/detectron2（cu113 预编译
    wheel），之后编译 DeformAttn ops（CUDA_HOME=/usr/local/cuda-11.6）。
- 2026-08-09: 暂停期复核。用户再次强调：不默认 IDOL 为最终 backbone、
  OVTR 升为最高优先级、COVTrack++ 按“最近方法边界”研究、先完成审计再长训。
  核对结果：六项审计文档均已存在且内容完整
  （cache bug impact / causal trajectory / protocol re-audit /
  backbone re-decision / full-stream TCO gate / supervision protocol）。
  状态检查：此前启动的 OVTR TAO val eval（GPU3, PID 35124）与
  COVTrack TAO val eval（GPU5, PID 5330）均已不在运行，且
  /data3/exp/OVTR/ovtr/results/teta_results_5_frame_val/ 与
  /data3/exp/COVTrack/results/ctao_results/ 没有结果文件——两个
  inference 会话随交互终端终止，结果未落盘，记为
  NOT_COMPLETED（需用户批准后按原命令重跑，暂停期不自动重启）。
  未启动任何 full training；IDOL 10-iter smoke 结果保留。
  将 PHASE4P_BACKBONE_AND_NOVELTY_REDECISION.md 状态由 DRAFT 更新为
  COMPLETE（A/B 仍为候选，等待 OVTR/COVTrack eval 结果后定稿）。
- 2026-08-09: 用户授权“继续一直推进”。发现 /data3/exp 下 OVTR/COVTrack
  目录属主 lwr(0700)，当前会话 user 不可读，无法复用旧下载；改为在项目内
  重建本地评估：
  - OVTR：clone commit 500e72c（third_party/research_refs_phase4n/OVTR），
    下载 ovtr_5_frame.pth / iou_neg5_ens.pth / clip_image_embedding_all.pt /
    validation_ours_v1.json / tao_test_burst_v1.json 到 checkpoints/ovtr 与
    data/external_annotations/ovtr；编译 DeformAttn ops；已启动 TAO val eval
    （GPU3，num_workers=8，结果写到 runs/iclr27_phase4p/ovtr_eval/）。
  - COVTrack：clone commit 9b0ced5（third_party/research_refs_phase4n/COVTrack），
    从 HF clarkqian/COVTrack 下载 ctao_base.json / ctao_public.pth /
    detpro_prompt.pt / ovtrack_pair.pth；已建 data/saved_models 软链；
    已启动 TAO val eval（GPU5，workers=4，结果写到
    runs/iclr27_phase4p/covtrack_eval/）。
  - 未启动任何训练；两个 eval 完成后解析 TETA + novel recall@FP。
- 2026-08-09: 两个 TAO val eval 均完成并落盘：
  - OVTR：TETA combined 35.34 / base 35.99 / novel 30.47；统一协议下
    novel recall @1 FP/frame = dev 0.020 / heldout 0.076；
    FP/frame @ recall0.3 = dev 16.9 / heldout 6.1。
  - COVTrack：TETA combined 37.13 / base 37.75 / novel 32.50；统一协议下
    novel recall @1 FP/frame = dev 0.056 / heldout 0.169；
    FP/frame @ recall0.3 = dev 3.3 / heldout 2.1。
  - 对比 D0 fixed（0.049/0.022；12.1/10.9）：COVTrack 前端显著更优，
    OVTR dev 反而低于 D0；已更新
    PHASE4P_BACKBONE_AND_NOVELTY_REDECISION.md §9。
  - 决策：BACKBONE_A=OVTR（长期 joint 目标），BACKBONE_B=COVTrack 风格
    （当前最强 empirical 前端）；IDOL/SimOWT 保留 baseline。
  - 下一步最小实验（待批准）：B 路 COVTrack 输出接 Q1 gate；A 路 OVTR
    TAO train 1–2 epoch smoke。不启动长训。
- 2026-08-09: 本轮推进到自然决策点：两个 eval 完成、re-decision 定稿、
  对比文档生成。A 路训练 smoke 被 `lvis_filtered_train_images.h5` 的
  HF 401/gated 阻塞；B 路需要为 COVTrack proposals 重建 Phase 4M/4N
  特征管线（DINO + semantic z + causal prefix），不是短任务。为避免在
  用户睡眠期间擅自启动可能改变协议的重型特征重建，本轮在此停止；
  未启动任何 full training，未遗留运行中进程。
- 2026-08-09 睡眠周期完成（A/B 最小实验全做完，未长训）：
  - OVTR 数据恢复：HF 直连下载 H5（16.2 GB），GDrive 下载
    `lvis_clear_75_60.json`（382 MB）与 `ovtr_det_pretrain.pth`（226 MB）；
    校验 H5 keys / 注释 / 权重载入均正常。
  - OVTR 20 迭代训练 smoke PASS：total loss 25.46→20.97，frame_1 CE
    1.056→0.953，无 NaN，checkpoint 293 MB 可保存；改动仅为
    `--max_train_iters` 测试参数（patch 已存）。
  - B 路：COVTrack proposals 严格因果特征重建完成（dev 23,623 /
    heldout 26,661）；LR 审计 B5（causal+static+semantic）：
    rec@1FP dev 0.056→0.269、ho 0.169→0.180；persistent AUROC
    dev 0.651→0.847、ho 0.668→0.796；代价是 @1FP age0/1 novel 归零。
- 判定：B 路 PARTIAL（保留 ablation），A 路 OVTR smoke 通过；
    最终仍选 OVTR + C1-C3；未启动任何 full training。
- 2026-08-09 16:35 主训练启动：
  - P0（官方 OVTR 1 epoch）已在 GPU5 启动（单卡，ETA ~23h，
    `--ckpt_interval 10000`）。
  - 审计发现 OVTR `_track2json` 的 `score` 实际取的是 bbox y2
    （track_result 缺 detection score）；已修复 eval 输出，P0/P1/P2
    将使用真实 score 重算 Novel Recall@FP（旧的 OVTR recall 数字作废）。
  - P2 TCO 设计完成：`models/tco.py` + `hit_count`/`is_fp` + 轻量
    BCE loss；`lambda=1.0`、`alpha=0.5`；只调制 `hit_count>=1`，
    保护 first-appearance。
  - 机制审计结论：native FP injection 未设置 -2、未直接训练
    history-conditioned validity；TCO 不是重复背景分类。
- 2026-08-09 17:25 P2 pilot 500 迭代完成（GPU3，5m16s）：
  - total loss 27.71→17.30；`frame_1_loss_tco` 0.040→0.031；
  - 无 NaN，checkpoint 每 100 迭代保存；
  - pilot eval：novel recall@1FP dev 0.223 / ho 0.040；new-query score
    未坍塌（mean 0.49）；TETA Combined 20.52（仅 500 iter）。
  - pilot gate PASS；P2 完整 1 epoch 已在 GPU3 启动（19:45，ETA ~22h）。
- 2026-08-10 12:12 P0 完整 1 epoch 完成（19h36m32s，GPU5），
  final checkpoint 已保存；P0 eval 已启动；P2 仍在 GPU3 训练（~116k/140k）。
- 2026-08-10 P0/P1/P2 全部完成：
  - P0 TETA 24.14（dev rec@1FP 0.232 / ho 0.000；FP@r0.3 dev 27.5/ho 21.3）；
  - P1（confirmation）与 P0 等价，说明简单 age/hit 修正不够；
  - P2 TETA 24.67（Base 25.08 / Novel 21.62）；FP@r0.3 dev 20.6/ho 16.0
    （-25%）；persistent FP/frame dev 16.8/ho 11.1（-39~42%）；
    dev early age0 @1FP 0.484→0.571；ho rec@3/5FP 0.065→0.090、
    0.082→0.158；
  - TCO 表示审计：persistent valid-FP logit 差 dev +3.24 / ho +3.33；
    new-object 未被压低；
  - 判定 `TRAINABLE_TCO_PARTIAL` + `EARLY_NOVEL_PRESERVED`；
    C2 设计完成未训练；不进入 semantic re-entry。

- 2026-08-10 Phase 4Q 启动（DSCQ 主方法）：
  - Same-Support Semantic Audit：P0/P2 same-support novel ClsA 未退化
    （heldout 0.013→0.020），Novel ClsA aggregate 下降更偏向 composition
    （H2），非 same-support 语义退化（H1 未支持）。
  - Gradient Conflict Audit（修复 load_model 后）：`L_TCO` vs `L_cls`
    shared params 平均余弦 -0.169（decoder -0.140，CIP -0.006），
    确认 `PHYSICAL_SEMANTIC_GRADIENT_CONFLICT_CONFIRMED`。
  - P1+ control（score/age/hit/disappear 的 dev logistic calibration）：
    在 P0/P2 proposals 上指标与 raw 完全一致 → 简单 confirmation
    不足以解释 P2 的 persistent-FP 抑制。
  - DSCQ 实现完成：`models/dual_state.py`（E/S GRU + reliability gate +
    birth 非对称 boost + s2q 语义注入 + gradient isolation）。
  - Q0/Q1 长轨迹续训已启动（GPU1/2，epochs 1-7，15k iter/epoch，
    到达 4-frame curriculum）；Q2 pilot 500 iter 运行中（GPU3）。
  - 修复 `main.py` resume bug：`if not args.eval` 误跳过 optimizer /
    start_epoch 恢复（`--eval` 默认非空），否则 Q0/Q1 resume 会从 epoch0
    重复训练。已修复并验证 `set epoch: epoch 1`。

- 2026-08-11/12 Phase 4Q 完成（Q0/Q1/Q2 matched 长轨迹 + eval）：
  - Q0（OVTR baseline）/ Q1（OVTR+TCO）/ Q2（OVTR+DSCQ）均完成
    epochs 1-7（15k iter/epoch，到达 4-frame curriculum）；
  - Q2 dev Novel Recall @1FP 0.460（Q0 0.250 / Q1 0.265），
    heldout 0.083（Q0/Q1 0.000）；dev FP/frame@r0.3 0.28，
    persistent FP 8.56；early age0 0.588；
  - 机制（40 batch）：E sep +6.10、birth sep +1.43、
    s_rel sep +5.38，无 NaN；same-support 语义不退化；P1+ 不能解释；
  - 代价：Q2 proposal 总量 9,504 vs Q0 28,315，LocRe 26.6 vs 50.0，
    TETA 18.83 vs Q0 25.84 / Q1 27.58，Novel AssocA 17.23；
  - 判定：PHYSICAL_STATE_SUPPORTED / SEMANTIC_STATE_SUPPORTED /
    DUAL_STATE_HYPOTHESIS_SUPPORTED / DUAL_STATE_TRACKOCD_PARTIAL /
    NOT_YET_ICLR_LEVEL；下一步一次 score-balance repair。

- 2026-08-13 Phase 4R / Q3（Observation-Existence decision decoupling）：
  - 假设：Q2 的 belief 表征学得好，但 belief→decision coupling 错；
    O_t/E_t/S_t 三种不确定性不应再压缩成单一 scalar score；
    E 应控制 physical lifecycle，O 保留 detection 职责。
  - 实现：`RuntimeTrackerBase.use_existence_lifecycle` +
    `_lifecycle_step()`；`--decision_decouple` 在推理时跳过
    `_apply_dual_scores`，输出分数=原生 O；新增
    `--e_keep_thresh 0.5 --e_term_thresh 0.35`。
  - Q3 pilot（Q2 long ckpt + 1000 iter）state gate PASS：
    E sep +6.93（≥70% Q2）、birth sep +1.67、s_rel sep +5.65、无 NaN；
    persistent valid/FP corr(O,E)=0.611/0.732，new valid=-0.075，
    证明 O 与 E 不是同一变量。
  - Q3 pilot eval FAIL（硬 gate）：
    proposals 60,198 vs Q1 31,650 / Q2 9,504；
    persistent FP/frame 61.2 vs Q1 16.1；
    novel@1FP 0.096 vs Q1 0.265；age0@1FP 0.245 vs Q1 0.459；
    FP@r0.3 55.3；TETA 22.26（LocRe 51.4 ≈ Q1，AssocA 29.2）。
  - Q2-alpha control（同 ckpt，仅 alpha）：
    a010 TETA 28.22（>Q1 27.58）/ a025 27.32；novel@1FP 0.260/0.244；
    persistent FP/frame 23.1/23.1（>Q1 16.1）→ alpha 插值不能同时
    恢复 TETA 与 FP 抑制。
  - 原因判断（三处叠加）：
    1) Q2-lineage 原生 O 在 0.19 阈值下每帧 ~82 detections
       （vs Q0 ~39）：训练时 E/TCO 承担抑制职责，O 未按“独立检测器”
       校准；
    2) 训练时 `disappear_time` 恒为 0（只有 protect_track_preds
       改它），E-GRU 从未见过 disappear>0；Q3 推理中低 E 高 O 的
       persistent FP 因 `cancel_disappear` 先清零再 +1 → disappear=1
       （OOD 输入）→ E 误判升高 → e≥keep → disappear=0 正向锁死，
       表现为 prior_hits 重尾（5-62）；
    3) `_lifecycle_step` 的 high-O+low-E 终止分支被 update() 里的
       分数清零逻辑抵消，永远不会累积到 miss_tolerance。
  - 修改内容：待定（一次 repair：lifecycle transition /
    association reliability / O-E interaction），先用有界
    lineage diagnostic 确认 flood 是否来自 Q2-lineage 原生 O。
  - 有界 diagnostic（800 val frames，修好 emission 过滤后）：
    Q2 long ckpt + decouple = 81.9 proposals/frame（Q0 = 34.1），
    emitted E 均值 +6.6、89% ≥ keep（训练 persistent FP E = -3.0）
    → flood 是 lineage 原生 O + E 正向锁死，与 pilot 无关；
    prior_hits 重尾 5-62 证实锁死。
  - 确认训练/推理不一致：训练 disappear_time 恒为 0，推理喂入
    0/1（OOD）→ E 误判升高 → keep 锁死；且 lifecycle 终止分支被
    update() 的 score-reset 抵消。
  - 一次 repair（O-E interaction，不重训即可测）：
    (1) DualStateQuery.forward 将 E 分支 disappear evidence 固定
    为训练分布 0（miss counter 是下游决策变量，不回灌 E-GRU）；
    (2) RuntimeTrackerBase.update 在 lifecycle 模式取消 blanket
    score-reset，disappear_time 由 learned belief transition
    决定（高 E retain / 低 E 累积终止 / 中间带 O hysteresis）。
  - Repair smoke：81.9 → 38.6 proposals/frame（≈Q0，落 gate 区间），
    emitted E +6.61 → +3.81；Q3 pilot re-eval 进行中
    （eval_repair.log）。
  - Repair eval 完成（dev）：proposals 60,198 → 33,742；
    persistent FP/frame 61.24 → 22.85（仍 > Q1 16.06）；
    novel@1FP 0.096 → 0.224（< Q1 0.265）；age0@1FP 0.281
    （< Q1 0.459）；FP@r0.3 55.3 → 26.1；TETA 22.26 → 28.28
    （≥ Q1 27.58）；LocRe 51.88、AssocA 25.92、Novel TETA 26.05
    全部 ≥ Q1；heldout novel@1FP 0.000。
  - 判定：Q3 repair ≈ Q2-α0.1（TETA 28.28 vs 28.22、persFP 22.85
    vs 23.08），且 FP-ranking/early-novel 更差 →
    `DECISION_DECOUPLING_NOT_SUPPORTED` +
    `SIMPLE_SCORE_CALIBRATION_SUFFICIENT`（reviewer control 触发）；
    按任务规则停止 patch，输出
    `FACTORISED_BELIEF_TRACKING_PARTIAL` / `NOT_YET_ICLR_LEVEL`。
  - 完整报告：docs/iclr27_phase4r/PHASE4R_COMPLETE_COPYABLE_REPORT.md
    （内嵌 Q2_ALPHA / Q3_PILOT / O-E MECHANISM AUDIT /
    TRACK_LIFECYCLE_AUDIT / NEAREST_WORK / ICLR_READINESS 全部内容）。

## Phase 4S（2026-08-14）

- 假设：TrackOCD 的 semantic identity 决策应当是 track-prefix 的因果
  sequential belief（KNOWN/EXISTING/NEW/DEFER）+ reliability-gated
  dynamic novel memory；episodic pseudo-novel training 可以让
  test-unseen identity 具备可迁移 discovery 能力。
- 数据审计：train_known_mean = 2,196 tracks × 8 帧 × 768-d，恰好 48 个
  supported-known category（min 1 / median 5 / max 1,341 tracks per cat），
  meta-train 38 类 / meta-dev 10 类（已有确定性 split）。
- 前端审计：Q1（novel rows 441、persFP/frame 16.06）优于 Q2-α0.1
  （393、23.08）→ 冻结 Q1 为 semantic 主输入，Q2-α0.1 做 cross-frontend。
- 2025/2026 prior art：OCGCD(ECCV24, KHU-AGI/OCGCD@28b9384)、
  TALON(CVPR26, ynanwu/TALON@4091c2d) 已 clone/pin（均无 LICENSE 文件）；
  PACO(2604.11484)、DP-BOA(2607.13504, ECCV26) 无确认代码；
  另有 PRISM/TTD-HM/DAA/MCCL。结论 `NEAREST_PRIOR_ART_NO_ISOMORPHIC_METHOD`。
- 实现：src/iclr27_phase4s/{protocol,episodes,model,runtime,train,baselines,
  pilot,dev_eval,features_q1}.py；episode 内 pseudo-novel 类从 known
  vocabulary 移除（否则模型可用其 frozen prototype 作弊）。
- 训练修复（重要）：decision CE 符号曾写反（minimize log-prob →
  loss 爆炸到 -12k）；memory protos 的 inplace 版本冲突用 clone 修复；
  teacher forcing：同一 track 只在首个 eligible 步 target NEW，之后
  target EXISTING(own slot)，避免教模型 slot explosion。
- 结果：tiny/trend 训练 loss 正常下降（dec ~4 → 0.05-0.09，known ~0.0001）；
  full training 20 epochs × 256 episodes（GPU0，约 30-40 min）。
- 修复（重要）：known 矩阵曾只含 meta-train 38 类（meta-dev 10 类为零行，
  训练时亦然），导致 eval 期 known logits 全零、模型在 meta-dev 上
  崩塌（known_acc 0 / novel_first_new 0.02）。v2 用 48 类全集矩阵
  重训（40 epochs × 256 fresh eps，GPU8）。
- 修复（训练循环）：Dataset 曾 `eps_per_epoch*epochs` 且每 epoch 全量
  重扫 → 总前向按 epochs² 膨胀（20 epochs 需 ~10h）；改为每 epoch 新建
  `eps_per_epoch` 个 fresh episodes（总 = eps×epochs）。
- 修复（bug）：decision CE 符号反（曾 minimize log-prob → -12k）；
  memory protos inplace 版本冲突（read 前 clone）。
- Pilot（v1 checkpoint + 修复后矩阵）：B3 fp_commit 0.0 / fp_born 0 /
  overbirth 4，但 known_acc 0.28、novel_first_new 0.02——确认是矩阵
  train/eval 不一致造成，非方法失败；等 v2 复测。
- 修复（representation）：GRU belief 未归一化参与 cos（h-norm 漏洞，
  eval 期 logits 到 163+）；修复为 decision 内 F.normalize(h)、tau_n
  16→10；对比 loss 0.5→1.0；slot 分离 margin 0.45（new）/0.35（existing）。
  诊断：z-mean 跨类 margin 0.29（好），裸 GRU h_T 塌到 0.10（差）。
- 修复（representation 2）：belief 加 residual running-mean
  h_t=norm(LN(GRU)+m_t)，m_t=running mean(z)；margin 0.10→0.16。
- 修复（decision）：new_head 由 MLP(h) 改为
  MLP([h, max_slot_logit, log1p(K)])——birth 决策必须 relative to memory
  证据，才能跨 category family 迁移。
- 重要评分 bug（非方法失败）：pilot 用“最后一个 commit”判 novel-first，
  而模型正确行为是 age2 NEW 后继续 EXISTING(own slot)，被误判为
  wrong_reuse（novel_first_new 0.02 的假象）；改为用第一个 commit 判
  identity。in-domain 逐帧诊断：模型按 teacher 语义正确运转
  （t=1 NEW → t>=2 EXISTING(own)）。
- v6 full training（修复后架构，40 epochs × 256 eps）进行中。
- v6 结果：episodic pilot PASS——B3 fp_commit 0.0 / fp_born 0 /
  novel-first NEW 0.729 / reuse 0.481 / slots 2.78；retrieval margin
  0.227；train loss dec 0.027 / known 0.007。
- dev（Q1，97 mapped/98 GT）：B3 RN-Acc 0.000（r_phys 接口在 95%-FP
  流上失效：原始尺度全 defer，重标定后 FP commit）；最强 dev causal =
  B2（RN 0.545 / rr 0.909 / count err 1）与 B1（All 0.500）/
  B0-dev（All 0.429）。dev 单帧 DINO 几何：own-cat cos 0.398 / novel
  0.420 / FP 0.558——FP 最像 known，novel routing 结构性瓶颈。
- 判定：`TRACKOCD_SEMANTIC_CORE_PARTIAL` / `NOT_YET_ICLR_LEVEL`；
  机制 episodic 成立，dev RN-Acc 因果链未建立。三修复周期已用尽，
  按规则停止 patch。报告：
  docs/iclr27_phase4s/PHASE4S_COMPLETE_COPYABLE_REPORT.md。

## Phase 4T (2026-08-14)
- 假设 H1/H2：flat variable-K softmax 导致路由随 K 漂移；synthetic 流≠真实 tracker 流。
- 训练流：冻结 Q1 在 60 个 TAO-train 视频(48 known 全覆盖)上推理，111387 行。
  审计：train FP 91.4% / dev 94.7%；score mean 0.351 两侧一致；prior_hits 3.09/3.29；
  持久 FP(≥2) 16036/4940 → TRAIN 流不比 dev 干净，H2 数据基础成立。
- 失败现象 1（T1 pre-repair 30×128）：known_acc 0.023、novel_first_new 0.857、
  reuse 0.0、overbirth 333 → L1 塌到 NOVEL、L2 塌到 NEW。
- 原因判断：L1 头没有 known-class evidence（known/novel 不可辨识）；L2 的 NEW logit
  是与 variable-K slot logits 竞争的绝对标量，无法相对 memory 证据锚定。
- 修改（一次性科学修复）：L1 = 固定 3 路 [KNOWN/NOVEL/DEFER]，输入加 max_known +
  known_margin；L2 = 固定 3 路 [EXISTING(best memory)/NEW/DEFER]，EXISTING 直接锚定
  max_novel，槽位选择另走 slot CE。两级均 K-invariant。
- 结果：待重训验证。
- 修复2（同一科学修复的完成部分）：L1 头输入 h/known-evidence detach——路由 CE
  不再反向重塑共享 representation（此前把不同 novel 类压到同一 manifold，
  learned_diff cos 0.70-0.78）；memory create/update reliability 恢复 Phase4S
  的 r_phys 约定（此前固定 0.5，且 pilot flat 分支与训练不一致）。
- 诊断：B3 官方 reuse 0.481 经旧 harness 复现为 0.0 → 先修 harness 再用。
- 12-epoch 消融：flat margin 0.177 / hier 0.142（B3 full 0.234）→ 部分是
  undertraining；统一升到 40×256（B3 同尺度）重训 T1-T4。
- T2 30×128 real pilot：known 0.933 / first_new 0.0014 / fp_commit 1.0 → flat
  + real 流塌向 KNOWN（T2 是 data-only control，失败符合预期）。
- T3/T4（旧版 30×128）real pilot：T3 known 0.942/first_new 0.901/reuse 0.114/
  fp_commit 1.0；T4 known 0.858/first_new 0.748/reuse 0.003/fp_commit 0.494/
  fp_born 184 → hierarchy 修好 known+first birth；cross-track reuse 与 FP 抑制
  仍是瓶颈，待 40×256 复测。
- 修复3（同修复内完成）：L2 头去掉 h，输入仅 [q,age,log1p(K),max_novel,
  novel_margin]——birth 必须相对 memory 证据而非 appearance；否则头从塌缩的
  belief 学出常数高 NEW（~16-17）盖过 max_novel（~13-15），永远 over-birth。
- 诊断：first-vs-later 的 max_novel 分布几乎重叠（其它类 12.9-14.2 vs 本类
  13.6-14.7，margin 0.14-0.21），L2 existing/new 可分离信号极弱；learned 边际
  0.209（B3=0.234）仍不足以稳定 reuse。修复后 reuse 仍 0——判定为
  representation-level 瓶颈，不再 patch L2。
- 已锁定最终架构（fixed-dim 两级 + L2 memory-evidence-only），T1/T3/T4 40×256
  重训，随后 pilot + dev 评测，按真实数字写报告。
- 最终 T 矩阵（40×256）：
  T1(hier+synthetic): epis known .441/first .480/reuse 0/fp_commit .007；
    dev known .053/RN 0。
  T2(flat+real): epis known .958/first .004/fp_commit 1.0；dev known .276/RN 0。
  T3(hier+real forced): epis known .916/first .941/reuse .080/fp_commit 1.0；
    dev known .171/RN .5455/rr .636/count err 0/NMI .951/ARI .389/mem 5493。
  T4(+qphys+defer): epis known .873/first .786/reuse .120/fp_commit .502/
    fp_born 197；dev known .040/RN .182/mem 1591（FP defer 66%，但 over-defer）。
- 判定：REAL_TRACKER_STREAM_MATCHED / HIERARCHICAL_ROUTING_SUPPORTED(episodic,
  K 稳定性 0.615→0.760 vs flat novel→known 0.147→0.450) /
  REAL_STREAM_TRAINING_SUPPORTED / DEFERRED+QPHYS 对 FP 有效但 over-defer。
  T4 full 未超 B2（0.182<0.545），Known/RN-Acc Pareto 未突破 →
  TRACKOCD_SEMANTIC_CORE_RETHINK_REQUIRED / NOT_YET_ICLR_LEVEL。
  报告：docs/iclr27_phase4t/PHASE4T_COMPLETE_COPYABLE_REPORT.md。

## Phase 4U (2026-08-14)
- 假设：cross-physical-track semantic representation 是 reuse 瓶颈；仅靠 frozen
  DINO + T3 GRU belief 会把 margin 压到 0.209，reuse 0.080-0.120。
- 数据审计：真实 Q1 train stream known 9,047 行 / 356 物理实例 / 47 类；
  跨物理语义正样本极不平衡（805 占 99.6%，26/48 类零跨物理对）→
  CROSS_TRACK_SUPERVISION_LIMITED；raw crops 全部可重建。
- 发现 Phase4T harness bug：train/pilot 只处理每个 episode 前 8 个
  occurrence（MAX_OCC=24），导致 episodic 数字失真；dev 不受影响。
  T3 fixed-harness pilot：known 0.923 / first 0.922 / reuse 0.418 /
  overbirth 815 / slots 8.71（vs 原报告 reuse 0.080）。
- 实现 TSR（GRU/mean + q_phys gate + residual running mean）Stage A
  cross-track SupCon（real+episodic mixed, 3k steps）。
  geometry：real complete margin_avg 0.205→0.758；episodic 0.307→0.861；
  dev 0.324→0.345（p8 0.394）；held-out meta-dev：learned 0.40 vs frozen 0.53
  （泛化反而下降）。
- 消融：none(CE-only) dev proto 0.798 最好；same_physical 最差（dev 0.66）；
  real-only 在 episodic/dev 崩（proto 0.11/0.47）→ mixed 必要。
- Stage B（TSR 冻结 + 只训 L1/L2，40×256，v2 修正 24-occurrence harness）：
  episodic known 0.681 / reuse 0.353 / overbirth 425；
  dev Known 0.724（T3 0.171）↑↑、RN-Acc 0.182（T3 0.545）↓↓、
  memory 19（T3 5493）↓↓、count err 9 → Known/RN Pareto 未突破。
- Stage C（joint fine-tune, rep-lr 1e-4，v2 修正 harness）：
  episodic known 0.966 / first_new 0.849 / reuse 0.779 / overbirth 118 /
  wrong_reuse 330；dev Known 0.618、RN-Acc 0.364、rr 0.455、count err 5、
  memory 947、all_track_acc 0.561、harmonic 0.458（T3 0.260）。
  reuse/memory/known/聚合指标大幅改善，但 RN-Acc 仍低于 T3 0.545 →
  TRACKOCD_REPRESENTATION_CORE_PARTIAL / NOT_YET_ICLR_LEVEL。
  一次主要 repair 已用，停止 patch，写完整报告。

## Phase 4V (2026-08-15)
- 假设 H1/H2/H3：known recognition 与 novel discovery 应分属不同 semantic
  space；Known/Novel routing 不应等价于 nearest-known similarity；novel
  anti-absorption 需 episodic open-world 训练学 routing boundary。
- 冻结分支：Known branch = R3 TSR（r3_mixed_gru）+ 48-way 线性头（dev Known
  可到 0.72）；Novel branch = Stage C TSR（d2_joint_v2）+ L2 head + 动态
  NovelMemory。Router = 15→19 维 evidence MLP/logistic，只训 router。
- Evidence 审计：episodic 400 ep 末步样本上 novel 分支 evidence 线性可分
  （dual+q AUROC 0.948）；known-only evidence AUROC 0.49（不可用）——
  pseudo-novel 仍在 48 类 frozen 词表内，known evidence 无法模拟 OOV。
- 第一次 pilot（final-step 训练 + first-commit 推理不匹配）：dev Known
  0.671 / RN 0.045 / memory 6 —— router 全偏 KNOWN。
- 修复（协议对齐，同一次 repair 内）：router 改为逐帧样本训练
  （samples_400_perstep，39339×15，heldout AUROC 0.953），dev 仍 RN 0.0、
  Known 0.724、memory 4；诊断显示所有 role 的 novel_max=0（memory 从未建立），
  t=0 三 role 的 p_known 几乎相同（0.539/0.533/0.520）→ router 在首帧失效。
- 根因：episodic pseudo-novel 的 known classifier 仍可识别（在 48 类词表内），
  router 学会忽略 known evidence、只依赖 novel memory；dev 真 novel 是 OOV，
  known evidence 其实可分（full-48 top1 0.689 vs 0.451、entropy 1.32 vs 2.05），
  但首帧 novel memory 为空。
- 修复2（同一次 repair 的完成部分）：episodic 训练时对 known evidence 做
  episode-masked（只在 episode-known 类上算 top1/margin/entropy/energy），
  模拟 OOV；样本 39339×19，heldout AUROC 0.982（MLP）/0.975（logistic）。
  dev 仍 RN 0.0：masked/full 差异信号在 dev 不存在（dev 无 episode mask）。
- dev 逐帧诊断（only eval）：true novel 的 full-48 classifier energy AUROC
  ≈0.71、proto energy ≈0.77、q_run ≈0.70-0.81（随 t 提高），known top1/entropy
  方向相反 ≈0.27-0.33（取反后 0.67-0.73）→ dev evidence 有弱可分信号，但
  episodic→dev 分布偏移导致训练 router 不迁移。
- OOD 阈值 baseline（cls_energy, min-age 0）：τ=5.2 → Known 0.566 /
  RN-Acc 0.364 / RR 0.409 / mem 4797；τ=5.4 → Known 0.487 / RN-Acc
  0.455 / RR 0.682 / mem 7070；τ=5.6 → Known 0.408 / RN-Acc 0.545 /
  RR 0.864 / mem 8606。无任何 τ 同时满足 Known≥0.55 且 RN-Acc>0.545；
  proto_energy 阈值在 3.2-3.5 全部 known（Known 0.724 / RN 0），边界
  附近 memory 爆炸 → prototype-similarity 路由同样不破 Pareto。
- 最终 dev 矩阵：dual MLP/logistic（per-step/masked）Known 0.724 /
  RN-Acc 0.0 / RR 0.0 / absorption 21/21；final-step MLP Known 0.671 /
  RN 0.045 / absorption 20/21。Stage C reuse 0.779 优势全丢。
- 结论：DUAL_SPACE_EVIDENCE_SUPPORTED（episodic 内）/
  OPEN_WORLD_ROUTER_NOT_SUPPORTED / NOVEL_ABSORPTION_NOT_REDUCED /
  KNOWN_RN_PARETO_NOT_BROKEN / TRACKOCD_NOT_YET_ICLR_LEVEL /
  TRACKOCD_CORE_FORMULATION_RETHINK_REQUIRED。
  一次主要 repair（per-step + masked OOV 模拟）已用，停止 patch。
  报告：docs/iclr27_phase4v/PHASE4V_COMPLETE_COPYABLE_REPORT.md。

## Phase 4W (2026-08-15)
- 假设：Phase4V 失败源于 (1) pseudo-novel 非真 OOV；(2) cold-start
  K=0 无 memory evidence 但 router 依赖 teacher memory；(3) cold/warm
  混在一个 router。
- Genuine-OOV：每个 episode 的 active known universe 只含 episode-known
  prototype（R3 TSR 均值），pseudo-novel 彻底移除；leakage 200/200 通过。
- Category-disjoint meta split：meta-dev 10 类
  [133,139,211,229,235,237,347,382,579,1144]，meta-train 37 类；
  TRAIN-only；容量：29/48 类≥2 物理实例、12/48 类≥2 视频。
- ColdStartHead（42→128→3，active-known+s_k 投影+q）与
  WarmMemoryHead（46→128→4，+memory evidence）分别训练；
  min_commit_age 前目标=NO_COMMIT；主评测 model-in-the-loop。
- 样本：train400 v3（bank size 2-12，cold 4855 / warm 46737），
  meta-dev200（bank=4）；head meta-dev acc：cold 0.810 / warm 0.736。
- Meta-dev pilot（200 ep）：cold NEW recall 0.622、absorption 10.5%；
  warm existing 0.217、birth 0.472、overbirth 332、FP born 283。
- Simple energy+clustering baseline：NEW recall ≤0.13、existing ≤0.04
  → state-conditioned head 显著更强。
- Q1 dev（v3，一次冻结）：Known 0.342、RN-Acc 0.045、RR 0.091、
  absorption 0.286（6/21）、memory 42、unresolved known 43% /
  novel 64%。
- Fixed-48 对照（同构 head，fixed-48 evidence）：meta-dev cold acc
  0.718 / NEW rec 0.511；dev Known 0.171 / RN 0.0 → genuine-OOV 有真实
  增益但不足。
- 诊断：dev active-known evidence known-vs-novel AUROC ≤0.76（t0
  energy），bank size 48 vs 训练 ≤12 的分布偏移与冻结 representation
  上限是主要瓶颈。
- 结论：CATEGORY_DISJOINT_META_SPLIT_VALID / GENUINE_EPISODIC_OOV_VALID
  / COLD_START_NOVEL_DISCOVERY_PARTIAL / WARM_MEMORY_REUSE_PARTIAL /
  STATE_CONDITIONED_TRACKOCD_PARTIAL / KNOWN_RN_PARETO_NOT_BROKEN /
  GENUINE_OOV_FORMULATION_NOT_SUFFICIENT /
  TRACKOCD_SEMANTIC_FORMULATION_RETHINK_REQUIRED /
  TRACKOCD_NOT_YET_ICLR_LEVEL。
  报告：docs/iclr27_phase4w/PHASE4W_COMPLETE_COPYABLE_REPORT.md。

## Phase 4X (2026-08-15)
- 假设：旧 prototype-router paradigm 的结构性瓶颈是 semantic identity
  inference；TrackOCD 应建模为 sequential non-parametric posterior over
  growing hypothesis set（known anchors + born novel components + NEW +
  NOISE）。
- Oracle ceiling（Stage C @ Q1 dev）：O1 oracle routing → RN 0.045 /
  Known 0.566；O3 oracle route+identity → RN 0.955 / all_track 0.990；
  → 瓶颈在身份推断，不在路由/覆盖率。
- Geometry：Stage C TSR 末步 unit-norm；train 跨物理类内 cos 0.787 /
  类间 0.259；dev known best-anchor 0.817 vs novel 0.729（gap ~0.09）。
- X3（vMF posterior + NEW/NOISE null，frozen）：meta-dev 200ep
  known 0.658 / first 0.535 / reuse 0.424 / fp_born 135（κ8），首次
  同时改善 birth/reuse/FP（>Phase4W 与 online clustering control）。
- X4（CompatibilityNet 768→128→128→1，61k pairs，train AUROC 1.0）：
  meta-dev known 0.603 / first 0.568 / reuse 0.405。
- Q1 dev：X3 κ16 → NEW 主导（Known 0.026 / RN 0.227 / mem 212）；
  X3 κ8 → over-defer（Known 0.000 / unresolved 99%）；X4 → RN 0.318 /
  Known 0.066 / mem 70。Pareto 未破。
- 结论：PROBABILISTIC_COMPONENT_MODEL_VALID（meta-dev）/
  NEW_COMPONENT_INFERENCE_SUPPORTED（meta-dev）/
  SEQUENTIAL_POSTERIOR_PARTIAL / KNOWN_RN_PARETO_NOT_BROKEN /
  TRACKOCD_NONPARAMETRIC_CORE_PARTIAL /
  NONPARAMETRIC_SEMANTIC_INFERENCE_NOT_SUPPORTED（dev）/
  TRACKOCD_FORMULATION_RETHINK_REQUIRED / TRACKOCD_NOT_YET_ICLR_LEVEL。
  一次 repair（X4 learned compatibility）已用，停止 patch。
  报告：docs/iclr27_phase4x/PHASE4X_COMPLETE_COPYABLE_REPORT.md。

## Phase 4Y (2026-08-15)
- Oracle Consistency Audit：发现 Phase4X O1 实现 bug（oracle route 对
  模型 DEFER 的轨迹未生效）。修正 O1c：Known 0.566 / RN-Acc 0.591 /
  RR 0.955 → oracle routing 下旧 memory 已进入 Pareto 目标区；
  Phase4X “身份是唯一瓶颈”结论修正为“routing 是主要瓶颈”。
- ADSSI：ObservationEncoder + StateAttention（permutation invariant）+
  trajectory-conditioned NEW proposal + learned gated transition +
  model-in-the-loop episodic 训练（warmup 10 + FP entropy 0.3 +
  birth margin 0.5）。
- 单元测试：empty memory / K=20 / permutation invariance 全部通过。
- 训练：150 eps × 20 epochs；teacher epochs births=0，rollout epochs
  births ~540/150eps（exposure bias 显著）。
- meta-dev pilot：known 0.571 / first 0.300 / reuse 0.000 / fp_born
  104；loose threshold first 0.410 / reuse 0.023。
- Q1 dev：Known 0.224 / RN-Acc 0.0 / RR 0.0 / absorption 0.09 /
  unresolved novel 0.91 / memory 2。
- 结论：ORACLE_CEILING_AUDIT_CORRECTED /
  SET_CONDITIONED_SEMANTIC_INFERENCE_PARTIAL /
  NEW_STATE_INSTANTIATION_PARTIAL /
  LEARNED_STATE_TRANSITION_NOT_SUPPORTED /
  MODEL_IN_LOOP_STATE_INFERENCE_NOT_SUPPORTED /
  KNOWN_RN_PARETO_NOT_BROKEN /
  LEARNED_DYNAMIC_STATE_INFERENCE_NOT_SUPPORTED /
  TRACKOCD_SEMANTIC_SUPERVISION_OR_TASK_RETHINK_REQUIRED /
  TRACKOCD_NOT_YET_ICLR_LEVEL。一次 repair 已用，停止 patch。
  报告：docs/iclr27_phase4y/PHASE4Y_COMPLETE_COPYABLE_REPORT.md。

## Phase 5A (2026-08-16)

### 简短研究计划

1. 关闭 Phase4Z KNOWN/NOVEL router 主范式，改为 unified strict-causal
   assign-or-create：每帧立即 KNOWN(c) / EXISTING_NOVEL(k) / NEW_NOVEL，
   过去标签不可变，predict -> freeze -> update。
2. 先验审计（OCD/GCD/C-GCD/Video-GCD/OCD 2025-2026）+ strict oracle
   ceiling + 4 个 pilot gate；通过才 full training。
3. Q1 strict/legacy 双评估 + ablations + multiseed + 一次 repair。

### 日志

- 2026-08-16 01:xx: Strict oracle（d2_joint_v2 TSR + 48 known head，Q1 dev
  chronological，704 aligned occurrences）：perfect causal action ceiling
  1.0，frozen known occurrence acc 0.733，novel reuse opportunity 0.908。
  修复了 precompute 中 state 不随 step 推进的 bug。
- 2026-08-16: 先验核验（web + GitHub API）：OCGCD ECCV2024（KU-VGI/OCGCD）、
  Happy-CGCD NeurIPS2024（MIT）、VB-CGCD ICML2025、Virtual-Category C-GCD
  ECCV2026、MCCL arXiv2509.06306（ICCV claim 未确认）、TALON CVPR2026
  （HF MIT）、LTC、DiffGRE ICCV2025、BURST VLDB2025、PACO（dataset）。
- 2026-08-16: Pilot episodes（150 train / 80 meta-dev，real Q1 TRAIN
  tracklets，bank {4,6} 与 {4,6,12,24}）。Gates：
  - Gate1 PASS（mechanism）：traj threshold meta-dev known 0.954 / first
    0.908 / reuse 0.869；learned head PARTIAL（0.885/0.708/0.642）。
  - Gate2 PASS in-episode：online vs static reuse 0.869 vs 0.821；Q1 反转为
    known 0.351 vs 0.669（online drift）。
  - Gate3 PASS：traj vs frame known 0.954 vs 0.465 / reuse 0.869 vs 0.196。
  - Gate4 PASS in-episode：cross-physical reuse 0.848；Q1 为 0.0。
- 2026-08-16: Q1 strict/legacy：
  - full stream threshold：known occ 0.151 / first 0.308 / reuse 0.209 /
    515 global births；legacy first Known 0.079 / RR 0.955 / RN 0.636。
  - aligned-only diagnostic：online known 0.351 / first 0.615 / reuse 0.419；
    static known 0.669 / reuse 0.465；learned head first 0.0；frame known
    0.0。
  - 一次 repair（physical birth gate age>=2, score>=0.35, prior>=1）：births
    515->134，Known 仍 0.158，未控制 FP pollution；停止 patch。
- 2026-08-16: Multiseed（3 seeds）稳定：traj online known 0.938-0.954 /
  cross 0.825-0.848。
- 结论：STRICT_CAUSAL_PROTOCOL_VALID / STRICT_CAUSAL_ORACLE_HIGH /
  IMMEDIATE_ASSIGN_CREATE_PARTIAL /
  TRACK_TRAJECTORY_SELF_SUPERVISION_SUPPORTED /
  ONLINE_CATEGORY_DISCOVERY_PARTIAL /
  CROSS_PHYSICAL_SEMANTIC_REUSE_PARTIAL /
  KNOWN_FORGETTING_NOT_CONTROLLED / PSEUDO_OOV_TRANSFER_GAP_NOT_REDUCED /
  KNOWN_RN_PARETO_NOT_BROKEN / STRICT_CAUSAL_CATEGORY_DISCOVERY_NOT_SUPPORTED /
  SUPPORTED_ONLY_SEMANTIC_SUPERVISION_INSUFFICIENT /
  TRACKOCD_NOT_YET_ICLR_LEVEL。未跑 full training（gate/诊断表明失败是
  结构性的：FP-dominated stream + true-OOV transfer gap）。
  报告：docs/iclr27_phase5a/PHASE5A_COMPLETE_COPYABLE_REPORT.md。

## Phase 5B (2026-08-16)

### 简短研究计划

1. Forensic audit Q1 physical stream：31,650 rows / 13,468 tracks /
   97 aligned / 13,371 unaligned 究竟从哪来、是什么。
2. 复现 counts；溯源 OVTR/TCO -> tao_track.json -> proposals CSV ->
   alignment；检查 raw query vs public output、duplicate、track-id、
   lifecycle。
3. Category-agnostic vs class-aware geometry audit、fragmentation、
   duplicate active tracks、TAO annotation coverage、retention frontier、
   visual contact sheets、counterfactual replay（frozen Phase5A）。

### 日志

- 2026-08-16: Counts 精确复现：31,650 rows / 20 videos / 732 annotated
  images / 13,468 tracks / 97 aligned（76 known + 21 novel）/ 13,371
  unaligned；mean len 2.35、median 1；exact duplicate rows = 0；
  near-duplicate（同帧不同 track，IoU>=0.5）2,252 rows。
- 2026-08-16: Lineage resolved：proposals CSV 直接来自 OVTR official
  public output `tao_track.json`（score>0.19, disappear_time==0,
  max 160/frame），再过滤到 732 个 dev annotated frames；不是 raw
  queries。OVTR 无 min-hits confirmation：query 出生当帧即 public。
- 2026-08-16: Geometry：frame IoU>=0.5 matched rows 1,669/31,650 (5.3%)；
  93/98 GT tracks 有 0.5 覆盖；4 个 greedy-aligned 无 0.5 帧。
  Unaligned 分解：geometry_unmatched 9,680 (72.4%)、partial overlap
  3,328 (24.9%)、fragment/duplicate 363 (2.7%)。median 17 preds/GT，
  p90 109，max 269；37 GT 有 duplicate active frames；66 preds 跨多 GT。
- 2026-08-16: TAO coverage：732 dev images 中 689 有标注（43 帧无 GT）；
  official 为 federated annotation（每视频至多 10 条 track + category
  exhaustive 子集）；per-video neg/not-exhaustive metadata 只 resolve
  4/13,371 unaligned tracks，13,367 unverified → unaligned != FP。
- 2026-08-16: Retention frontier：th=0.19 全部 admitted；th=0.5 known
  cov 0.171 / novel 0.048 / unaligned 0.007；novel 天然低分
  （first mean 0.274 vs known 0.371）→ NOVEL_RETENTION_BIAS_CONFIRMED、
  CAUSAL_PHYSICAL_SEPARABILITY_LOW。
- 2026-08-16: Counterfactual frozen Phase5A replay：S3 dedup Known 0.145 /
  RN 0.591；S4 frag-norm Known 0.118 / RN 0.591；S2 geometry-oracle
  Known 0.329 / RN 0.545；S0 0.118 / 0.636。→ SEMANTIC_TRUE_OOV_GAP_REMAINS，
  interface bug 不是主因（LEGAL_INTERFACE_FIX_NOT_FOUND）。
- 2026-08-16: Contact sheets 300 张已生成；环境不支持可靠视觉检查 →
  MANUAL_VISUAL_AUDIT_REQUIRED（未伪造人工标签）。
- 2026-08-16: 结论：PHASE5A_COUNTS_REPRODUCED /
  PHYSICAL_STREAM_DATA_LINEAGE_RESOLVED /
  PUBLIC_OUTPUT_VS_INTERNAL_QUERY_RESOLVED /
  UNALIGNED_CAUSE_DECOMPOSED / CATEGORY_AGNOSTIC_ALIGNMENT_COMPLETED /
  TAO_ANNOTATION_COVERAGE_MATTERS / PHYSICAL_STREAM_INTERFACE_CORRECT /
  UNALIGNED_NOT_EQUIVALENT_TO_FP / MIXED_PHYSICAL_STREAM_FAILURE /
  PHYSICAL_FRONTEND_FP_BOTTLENECK_CONFIRMED /
  CAUSAL_PHYSICAL_SEPARABILITY_LOW / NOVEL_RETENTION_BIAS_CONFIRMED /
  SEMANTIC_TRUE_OOV_GAP_REMAINS / LEGAL_INTERFACE_FIX_NOT_FOUND /
  NEW_TRACKER_PILOT_JUSTIFIED / TRACKOCD_NOT_YET_ICLR_LEVEL。
  报告：docs/iclr27_phase5b/PHASE5B_COMPLETE_COPYABLE_REPORT.md。

### Phase 6A 日志

- 2026-08-16: Phase 6A 启动：Joint End-to-End TrackOCD（JCDQ 工作名）。
  阅读 Phase 4Q runner / 4P stream builder / OVTR 源码；建立
  `src/iclr27_phase6a/`、`docs/iclr27_phase6a/`、
  `outputs/iclr27_phase6a/`。
- 2026-08-16: 构建 partial-label 训练文件 `lvis_known48_partial.json`
  （保留 48 supported-known 标注 141,373 条，其余 1,564,493 条进入
  unlabeled stream；保留全部 1203 类别以保证 OVTR class logits 可用）。
- 2026-08-16: 在 OVTR fork 实现 JCDQ：JointStateHead（objectness +
  semantic state + physical embedding）、SemanticMemory（48 known anchors
  + 动态 novel prototypes + assign/create head）、semantic->query
  feedback；训练损失：nnPU objectness / known CE / pseudo-novel disc /
  temporal consistency / physical separation。
- 2026-08-16: 修复三处训练阻塞：(1) `criterion.joint_model = model`
  造成 `.to(device)` 递归 → 改为共享 JointQuery；(2) resume 时 optimizer
  参数组不匹配 → 回退重建 optimizer；(3) 成对 IoU 用错工具 →
  `box_ops.box_iou`。30-iter smoke 通过（loss 有限、无 NaN）。
- 2026-08-16: known anchors 从 CLIP text embeddings 初始化
  （`init_known_anchors_from_text`）。
- 2026-08-16: 2-GPU DDP 两次尝试均在首个 iteration 停滞（非死锁：
  GPU 100%、rank0 计算、rank1 futex 等待），决定主训练改用单卡
  （GPU 0 全 epoch，GPU 9 消融），DDP 改动保留但非主协议。
- 2026-08-16: 主训练启动：GPU 0，1 epoch=41,421 iters，ckpt 每 5k；
  消融 A1–A5 各 5k iters 由 blocking runner 在 GPU 9 顺序执行。
- 2026-08-16: 修复 eval 链路多个问题：`forward_assign_create` CPU/GPU
  设备不匹配、`--video_ids` 过滤时机太晚（dataset 构建后才设置）、
  `ovtr_main_eval.py` heldout 空集崩溃、strict evaluator 需 torch 环境。
- 2026-08-16: 语义记忆改为推理期 stream-global（不再每视频 reset），
  否则 novel slot id 跨视频冲突且无法跨视频复用；训练仍每 sequence
  reset。5k checkpoint 2 视频冒烟：236 rows/49 tracks，全部行带
  sem_action/sem_sid；causal contract 全部通过（no-future/no-relabel/
  memory-legality/dual-identity/first-frame/objectness invariance）。
- 2026-08-16: 启动 autopilot：等消融完成 → 全模型 Q1 eval →
  等主训练完成 → final eval → 重建 58 节完整报告。
- 2026-08-16: 发现并修复 public-score gate bug：`TrackerPostProcess` 在
  joint objectness 赋值后用 max known-class confidence 覆盖 scores，
  public 准入实际仍是 known-conf。修复后 10k checkpoint 2 视频冒烟：
  novel first-score 0.485 > known 0.245（Phase5B 的 novel 抑制被反转）；
  objectness 与 known-conf 相关仅 0.15；契约测试全过。语义决策仍偏
  “全部 existing→slot0”（10k），待 41k 观察。
- 2026-08-16: 15k 2 视频冒烟：116 rows，1 new / 108 existing / 7 known；
  objectness 与 known-conf 相关升至 0.52（子集内 admitted 多为高 known
  conf），但 known 动作行 base known-conf 仅 0.33 而 joint objectness
  0.66，说明语义记忆与类置信度已解耦；最终以 20 视频全量审计为准。
  A1–A4 消融完成，A5 运行中。
- 2026-08-16: **消融协议 bug**：`run_ablation` 未传 `--resume`，触发
  `train_tracking_only` 全冻结逻辑（resume 为 None 时才 unfreeze 白名单），
  joint 模块（obj_head/sem/phys/assign/anchor）5k 迭代全部保持初始零权重，
  A1–A5 结果无效（A1–A5 全 0 行）。修复：消融统一 resume Phase4Q
  checkpoint；删除旧 `.done`/`.launched`，GPU9 重训 5×5k（预计 ~5h）。
- 2026-08-16: 时间线重排：主训练 GPU0 至 ~22:00；主完成后 GPU0 跑
  FULL_VAL 权威 Q1 eval（与 Phase5B 同协议）；GPU9 重训消融后跑 filtered
  eval（内部可比，报告明确标注协议差异）。
- 2026-08-17: 主模型 41k FULL-VAL 权威结果：2650 rows/196 tracks
  （vs 基线 31650/13468），len1 2%，frag cats 10（vs 21），duplicate
  0（vs 37 GT tracks），novel first-score 0.342 vs known 0.368（novel
  抑制反转），objectness corr 0.08；但语义坍缩：known 动作几乎为 0、
  全部路由到 novel slot 0（known-to-existing 0.939，RN-Acc 0.09）。
  物理/objectness 强，语义弱。
- 2026-08-17: Major Repair 1（有明确根因）：semantic head 欠训练 +
  assign/create 退化为“总是 existing”。修复：known CE ×3 + margin、
  disc loss ×2、held-out 增至 6，从 41k 续训 8k（1h25m）。正在跑
  repair1 的 FULL-VAL eval。
- 2026-08-17: Repair1 FULL-VAL：RN-Acc 0.091→0.136，known-to-existing
  0.939→0.764，但 known occurrence acc 仍 0.0（known 动作坍缩到
  428/35/34），first-novel birth 0，语义判别未解决。同一根因出现两次，
  按策略停止修复并诚实报告：物理/objectness 强成功（fragmentation、
  duplicates、novel 抑制全部反转），语义 true-OOV gap 未缩小 →
  JOINT_TRACKOCD_CORE_PARTIAL / TRACKOCD_NOT_YET_ICLR_LEVEL。
- 2026-08-17: 修正 physical_eval 的 semantic 统计 bug（known 类别 sid
  被误计入 novel slots）；重建 58 节完整报告与全部 topic docs。

### Phase 6B 日志

- 2026-08-17: Phase 6B 启动：Semantic Recovery + DSCT re-architecture。
  完成 8 项有界 correctness audit（无接线/映射/泄漏 bug；legal known
  held-out acc 0.88；action accounting 显示 existing logits 5.61/-7.07、
  max_novel 0.94 vs max_known 0.47）→ `JCDQ_SEMANTIC_FORMULATION_FAILED`
  → `DUAL_STATE_REARCHITECTURE_TRIGGERED`。
- 2026-08-17: 核验 2025/2026 prior art（DiffGRE、TTD、TALON、Video-GCD
  MCCL、NC-GCD、Dual-Path MOT decoder、DTME-MTL），写
  `REARCHITECTURE_DECISION.md` 与 `2025_2026_SEMANTIC_REARCH_PRIOR_ART.md`。
- 2026-08-17: 实现 DSCT-TrackOCD：`models/dsct.py`（physical state +
  semantic state + category memory + 3-way calibrated decision + gated
  P<->S interaction）；接入 OVTR（ovtr.py/main.py/eval.py）；Stage A-D
  训练开关与梯度分 branch 记录。单元/契约测试全过（K_0=0、birth
  legality、NEW path、dual identity、objectness independence、
  first-frame immediacy）。
- 2026-08-17: 30-iter smoke 通过（loss 有限）；2 视频 eval 全链路通过
  （254 rows，causal contract 全过）。修复两个集成 bug：
  (1) novel memory in-place 更新破坏 autograd → 改为 detach+reassign；
  (2) anchor_init device 不匹配 → build 后手动 to(device)。
- 2026-08-17: 发现决策头数值爆炸根因：memory 为空时 max_novel 特征为
  -1e9，被决策 MLP 放大到 ~1e8 logit → 全部 KNOWN。修复：空 novel
  时 max_novel=0 + has_novel=0，特征 clamp [-100,100]。
- 2026-08-17: 提高伪 novel 频次：held-out k 6->12（仍为合法
  supported-known，只作为 pseudo-novel mechanics 训练）；pilot C/D
  加长至 5k/2k；pilot gate 改为 known CE 收敛判据（<0.05）。
  Pilot 重跑中（GPU9），Stage A 已完成（GPU0）。
- 2026-08-17: 两次 pilot gate FAIL 的根因（关键修复）：
  (1) `sem_head.out` 零初始化 -> `sem≈0` -> `F.normalize` 梯度放大
  ~1e23 -> 总 grad_norm=inf -> `clip_grad_norm_` 把所有梯度清零，
  pilot B/C/D 全程空转（loss 下降只是 batch 方差）；修复为 xavier 初始化
  + resume 时检测零权重并重建；
  (2) `named_parameters(remove_duplicate=True)` 只返回
  `criterion.dsct.*`，freeze/branch-norm 的 `startswith('dsct')`
  前缀不匹配，stage 冻结从未生效；改为 `'dsct.' in name`。
- 2026-08-17: 修复后 5-iter smoke：grad_norm 有限（~100），known CE
  真实下降 0.15->0.01，decision 头开始更新。pilot 第四次重跑中。
- 2026-08-17: 进一步修复：(1) 决策头改为“校准 base + 零初始 residual”
  （KNOWN~max_known、EXISTING~max_novel、NEW~-max_known 当 memory 空），
  避免稀有 NEW target 无法移动任意初始化 MLP；(2) known CE 增加绝对
  anchor attraction（correct sim -> 1），使 max_known_sim 成为真正的新颖性
  信号（此前 CE 只要求相对最大，所有类别 sim 仍为负）；(3) novel-like
  coef 1->2。Pilot gate 通过：Q1 pilot 流 680 KNOWN / 1713 EXISTING /
  1 NEW（首次合法 birth 后 EXISTING 复用），objectness Pearson 0.13，
  causal contract 全过。正式 Stage B（41,421 iters）已在 GPU0 启动。
- 2026-08-18: 正式四阶段训练完成（Stage A 1k / B 41.4k / C 8k / D 8k）。
  最终 Q1 全量评估（36,375 帧，134,981 rows）：
  - 物理：373 tracks、len1 13.1%、frag 8 类、dup-active 2、
    novel first-score 0.367 > known 0.323、objectness Pearson 0.188；
    causal contract 全过（dual identity、first-frame、no-future/relabel）。
  - 语义：2 次合法 NEW birth + 122,701 EXISTING 复用（novel memory
    mechanics 恢复）；Novel NMI 0.437 / ARI 0.224（vs JCDQ 0.262/0.080）；
    conditional novel acc 0.375。但 aligned Known acc 仍 0、RN-Acc 0、
    cross-physical reuse 0 ->
    DUAL_STATE_SEMANTIC_RECOVERY_PARTIAL /
    TRUE_OOV_TRANSFER_GAP_NOT_REDUCED / TRACKOCD_NOT_YET_ICLR_LEVEL。
  - 根因判断：LVIS TRAIN 的 anchor attraction 未迁移到 TAO Q1 域
    （inference max_known_sim 均值 -3.68），aligned known 被决策层判为
    novel-like。按 Phase 6B 硬规则不再堆 head/调权重。

### Phase 6C 日志

- 2026-08-18: Phase 6C 启动：Open-World Trajectory Semantic Representation
  Reset。冻结结论：DSCT physical/objectness + strict-causal 保留；彻底更换
  semantic branch；禁止 novel GT 训练；最多一次架构级切换。
- 2026-08-18: 核验 prior art（TRACT/TraCLIP ICCV25、GET CVPR25、
  Prior-Constrained Association AAAI25），复用 Phase6B 已核验列表与内部
  V0/DINOv3 bakeoff 结论（DINOv2 保留，DINOv3 无明确增益）。
- 2026-08-18: 构建轨迹级数据集：2196 known tracks（48 类，DINOv2 逐帧缓存）
  + 11,377 条 Phase4T TRAIN unlabeled tracks（>=3 rows，GT 忽略）+ PCA
  （128d，explained 0.857）。
- 2026-08-18: 实现 TSE（PCA 初始化 base + zero-init residual MLP + 48
  anchors；known CE + anchor attraction + same-track temporal InfoNCE +
  cross-track MNN + anchor-preservation drift）。修复两个训练 bug：
  category_id→anchor index 映射；PCA buffer device 迁移。
- 2026-08-18: V0 非参数基线在 Q1 严格流：trajectory EMA tau=0.45 →
  Known=0.277、first-birth=0.333、reuse=0.333、cross=0.0、born=14、
  NMI=0.892/ARI=0.819（vs Phase6B Known=0/RN-Acc=0）；frame 基线
  Known=0.150/born=32 → trajectory > frame 确认。阈值校准也选 0.45。
- 2026-08-18: 正式 TSE 训练（60 epochs，~10s/epoch）：初版 MNN 固定
  min_sim=0.4 导致损失恒 0；改为互邻均值吸引（无固定阈值）+ unlabeled
  batch 32→64。主训练 GPU0、消融 no_mnn GPU7 / no_frame GPU9 并行。
- 2026-08-18: 主 TSE 结果（Q1 严格流，calibrated tau=0.65）：Known=0.671、
  first-birth=0.222、reuse=0.196、cross=0、born=10、count err=1、
  NMI=0.820/ARI=0.620；legacy all-track=0.235（Phase6B=0.031）；causal
  contract 全过。V0 基线 trajectory=0.277 > frame=0.150；TSE frame=0.504。
  消融：no_mnn≈main（0.677）、no_frame Known=0.813 但 novel 塌缩
  （birth=0.111/NMI=0.526）、no_pres≈main（0.700）。
- 2026-08-18: 失败根因定位：Q1 aligned 只有 11 条 novel 轨迹/9 类，其中仅
  cat 611/817 有两条物理轨迹（最多 21 条 cross 行）；TSE 空间中 known/
  novel 首帧 max-known-sim 分布重叠（known median 0.740 vs novel 0.623），
  FP 提前 birth 的 slot 会吸收真实 novel；cat 817 跨轨迹 cosine 仅 0.474
  （首帧 0.329）。双阈值+margin 诊断可达 cross=0.058-0.094，但 Known 降到
  0.33-0.43 且 born 61-108 —— 明确属于以 known 换 novel，不作为主结果。
- 2026-08-18: 架构级切换 TSE-v2（open-space repulsion + GT-identity 无标签
  池 + 强 MNN）：Known=0.720 提升，但 birth/reuse/NMI 反而下降
  （0.111/0.108/0.734）。结论：
  `TRAJECTORY_SEMANTIC_RECOVERY_PARTIAL` /
  `TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING` /
  `TRACKOCD_NOT_YET_ICLR_LEVEL`。完整报告：
  `docs/iclr27_phase6c/PHASE6C_COMPLETE_COPYABLE_REPORT.md`。

### Phase 6D 日志

- 2026-08-18/19: Phase 6D 启动：Full-TAO Cross-Physical Novel Category
  Discovery。核验 prior art：OCGCD/DEAN（ECCV24，官方 repo KHU-AGI/OCGCD）、
  MCCL（Video-GCD 25）、Beyond Known Clusters（24）。
- 2026-08-19: 构建 full TAO TRAIN 合法轨迹池：500 videos / 18,274 frames /
  2,647 tracks（2,196 known / 451 unlabeled，168 unlabeled categories，
  1,219 novel same-category pairs）；DINOv2 特征 2,644/2,647（99.9%）。
  已知轨迹缓存与 train.json 逐条核验一致（2196/2196），仅补提 novel 轨迹。
- 2026-08-19: 架构 1 GMNA（momentum teacher + global memory bank +
  cross-video mutual-neighbor aggregation）：Q1 Known=0.813 但 novel 全面
  坍缩（birth=0.111、reuse=0、NMI=0.271）——83% known 的 bank 把 novel
  吸进 known。
- 2026-08-19: 架构 2 OpenGMNA（唯一一次架构级切换；known-filtered bank
  θ=0.75 + open-space repulsion）：Known=0.732、birth=0、NMI=0.448，
  open 分离未迁移到 Q1。
- 2026-08-19: 消融：no_discovery Known=0.755/birth=0.111；no_teacher
  Known=0.810/birth=0（全坍缩）；small_pool（Phase6C 池 + 同一 GMNA）：
  Known=0.769、birth=0.222、reuse=0.265、cross=0.014（cat 611 一条正确
  跨物理复用行）、NMI=0.627 —— 唯一非零 cross，但 RN-Acc 低于 6C
  （route 0.091 vs 0.273），不稳健。
- 2026-08-19: TRAIN 侧后验诊断（448 novel tracks/167 cats，KMeans）：
  raw NMI=0.817/ARI=0.201；GMNA NMI=0.818/ARI=0.251；OpenGMNA
  NMI=0.814/ARI=0.252 —— 学习目标没有超越冻结 foundation 的 true-OOV
  结构。
- 2026-08-19: 最终结论：
  `CROSS_PHYSICAL_TRUE_OOV_REUSE_NOT_SOLVED` /
  `TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING` /
  `TRACKOCD_NOT_YET_ICLR_LEVEL`。完整报告：
  `docs/iclr27_phase6d/PHASE6D_COMPLETE_COPYABLE_REPORT.md`。
## Phase 7A 日志

- 2026-08-19: 核验 2025/2026 prior art（VB-CGCD/ICML25、DiffGRE/ICCV25、NCD-DLT/WACV25、RCA/ICML26、FSNCD/IJCAI25、TALON/CVPR26），写 PRIOR_ART.md。
- 2026-08-19: 实现 RACC-Memory（reliability-weighted causal memory + attach-or-create head）与 proxy-episode 训练/评估链路；修复决策 mask bug（NEW 被永久 mask）、内存数组拼接瓶颈（预分配）、track-EMA 语义表示。
- 2026-08-19: **发现并修复 Phase4T stream_data.py bbox bug**：第二遍循环用过期变量 `bb` 把最后一行 bbox 覆盖到所有行，导致 111,387 行的 bbox_xyxy 全错、原始 DINOv2 feats 为垃圾（TSE max-ksim ~0.08 vs Q1 ~0.64）。修复为 `json.dumps(r["bbox_xyxy"])`，重新生成 stream 并重新提取 DINOv2 特征（39,047 行，3 GPU 并行）。
- 2026-08-19: 早期（错误特征）训练结论：hybrid known-tau=0.65 可保住 Known≈0.66，但 dev cross=0；heldout cross≈0.04 与 EMA 基线相同，不构成方法增益。
- 2026-08-19: 正确 DINOv2 特征下完成 Phase 7A 主训练与评估：
  - RACC-v1（hybrid，冻结 known-tau=0.65 + 学习 attach/create 头）：DEV Known 0.657 / first 0.111 / reuse 0.176 / cross 0 / NMI 0.827 / ARI 0.644；heldout Known 0.411 / reuse 0.220 / cross 0.034（EMA 基线 heldout cross 0.040）。
  - 唯一架构级切换 RACC-v2（规则化 evidence memory，proxy-val 校准）：DEV cross 0 / born 0，heldout cross 0，不优于 RACC-v1。
  - 消融：no-maturity 破坏 birth/reuse；no-cross-track 破坏 first（0）且 NMI 掉到 0.712；no-rel/sem-only 反而提升 first/reuse（proxy 与 Q1 的 objectness 相关性方向相反）。
  - 根因：DEV 唯一可跨物理的 novel 类（611/817，以及 224）track-EMA max-ksim 首行为 0.65–0.85，被冻结 0.65 known 门吸收；冻结协议下无法在不降 tau/改 embedding 的情况下解决。
  - 最终结论：CAUSAL_NOVEL_CATEGORY_MEMORY_NOT_SOLVED / CROSS_TRACK_NOVEL_REUSE_NOT_SOLVED / TRACKOCD_NOT_YET_ICLR_LEVEL。

## Phase 7B 日志

- 2026-08-19: Phase 7B 启动：Trajectory-Level Open-World Semantic Explainability。冻结 7A 结论（RACC 停止、tau 冻结、physical 不进 semantic 主评分、novel memory 用 simple EMA）。
- 2026-08-19: 核验 2025/2026 prior art（GHOST/AAAI25、Known Meets Unknown/arXiv25、ProtoDCS/TCSVT26、Prior2Former/ICCV25、OWOBJ/CVPR25、Semiparametric Label Shift/arXiv25），写 PROTOCOL/PRIOR_ART。
- 2026-08-19: 实现 TOSE：corrected Phase4T 轨迹级 class-conditional Gaussian（MAP-shrunk diagonal）+ 统一 evidence 竞争头（KNOWN/EXISTING/NEW，无冻结 tau，无 objectness 特征）；proxy-OOD episodes 训练；修复 7A 继承的 first_acc 计数 bug（known 行误计 first）。
- 2026-08-19: TOSE-MLP 主训练 30 epochs（代理 val 选择 epoch 21）：DEV Known 0.501 / first 0.111 / reuse 0.147 / cross 0 / absorption 0.396（EMA 0.613）；heldout Known 0.372 / reuse 0.269 / cross 0.036 / absorption 0.204。known→existing 0.366 成为主要错误。
- 2026-08-19: 诊断 epoch 5 checkpoint 首次在项目内取得 DEV CT-Reuse>0（cat 611，track 842→843，3 行正确；heldout cross 0.163/20 行），但 Known 掉到 0.326/0.251——边界偏移过大。
- 2026-08-19: 唯一架构级切换 TOSE-Lin（线性 evidence 头，GHOST 风格；修复 LayerNorm 后完整重训）：DEV Known 0.0、born 96、cnt 87，彻底失败。
- 2026-08-19: 消融（MLP）：frame-level Known 0.438；no-dist Known 0.0；no-proxy first 0/born 0；classifier-conf Known 0.003/cnt 57/cross 0.167；均证明 distribution 核心、轨迹级证据、proxy-OOD、semantic explainability 各自必要。
- 2026-08-19: 最终结论：
  OPEN_WORLD_SEMANTIC_EXPLAINABILITY_NOT_SOLVED /
  NOVEL_TO_KNOWN_ABSORPTION_REMAINS /
  CROSS_TRACK_NOVEL_REUSE_NOT_SOLVED /
  TRACKOCD_NOT_YET_ICLR_LEVEL。报告：
  docs/iclr27_phase7b/PHASE7B_COMPLETE_COPYABLE_REPORT.md。

## Phase 7C 日志

- 2026-08-20: Phase 7C 启动：Known-Preserving Open-World Calibration。
  诊断 7B epoch-5：218 个 known→existing 中 195 个（89%）attach 到 FP 行
  诞生的 slot；吸收的 novel（190/224/817）为单 anchor 极稳定/高 ksim，
  位于 known 语义胞内；可复用的 611 的 class-flip rate=0.75（同 ksim 段
  known ≤0.5）。
- 2026-08-20: 核验 prior art（DEUS/CVPR26、REL/25、GeoEnergy/NeurIPS25、
  Conformal Open-Set/2510.13037），写 PROTOCOL/PRIOR_ART。
- 2026-08-20: 构建合法 class-level hard meta-val：train_visible 20 /
  hidden_train 14 / hidden_val 13（完全 hold out，避免 7B meta-val 记忆化
  缺陷）；meta-val visible-known 7849 行（7B 仅 95）。发现并修复 7C 训练
  bug：target 原按固定 split 而非 chunk 级 visible/novel 集合，导致
  episodic-hard 监督静默失效。
- 2026-08-20: 筛选：margin 目标不稳定（w=10 压垮 novel，w=1 压垮 known）；
  16-dim 轨迹特征（flip rate/anchor entropy）加入；logit 近似 frontier
  无效（假设首现后 slot 必存在），全部改用真实 replay。
- 2026-08-20: 完整训练 kpoc_main（hard，30 epochs）与 kpoc_abl_random
  （random）。meta-val 真实前沿选择 offset=+0.75（Known 0.871 / reuse
  0.941）。
- 2026-08-20: Q1 DEV（冻结 offset +0.75）：Known 0.225 / first 0.111 /
  reuse 0.167 / cross 0 / absorption 0.144（项目最低）/ known→existing
  0.767；heldout Known 0.194 / first 0 / cross 0。DEV 后验 offset 扫描
  无任何点满足 Known≥0.60 且 novel>0；frontier 未外移，反而劣化。
- 2026-08-20: 最终结论：
  KNOWN_NOVEL_PARETO_NOT_SOLVED /
  TRACKOCD_JOINT_OPERATING_POINT_NOT_SOLVED /
  TRACKOCD_NOT_YET_ICLR_LEVEL。报告：
  docs/iclr27_phase7c/PHASE7C_COMPLETE_COPYABLE_REPORT.md。

## Phase 10 日志

- 2026-08-21: Phase 10 重新读取磁盘上的 AGENTS、research/progress 文档、
  Phase 8A/9A 报告与当前资源状态；冻结 Phase 8A/9A 结论，不继续 RACC、
  TOSE、KPOC 或 semantic-memory lifecycle 工程。
- 2026-08-21: 在同一 corrected Q1 DEV stream 上完成 DINOv2 bbox、冻结 TSE、
  Phase-8A B causal embedding 的 novel geometry 诊断。11 条 aligned novel
  tracks / 9 类中只有 2 个 same-category pairs；B 的 same/inter cosine
  distance=0.1627/0.3380、NN=0.1818、cross-video NN=0.0909，说明存在弱
  结构但没有足够的 cross-track invariant signal，CT=0 不是单纯的 threshold
  bug。
- 2026-08-21: 比较 video-language foundation、unlabeled trajectory
  contrastive/self-supervision、hybrid 三方向；选择 Direction C 作为最小
  representation-only prototype，保留冻结 DSCT/B decision process，禁止
  physical ID 进入 semantic feature。
- 2026-08-21: 实现 `src/iclr27_phase10/model/hybrid.py` 与训练/回放/诊断
  链路。C 使用 frozen TSE frame projection、causal GRU、delta、gated
  residual、known CE、unlabeled two-view consistency 与 prefix smoothness，
  不读 novel GT。5 epochs x 40 steps，GPU5 单卡完成，未启动 4 GPU。
- 2026-08-21: C Q1 DEV strict：Known=0.763689、First=0.222222、
  reuse=0.029412、CT-Reuse=0、NMI/ARI=0.730753/0.500421；Known 提升但
  joint gate 失败。no-consistency 对照：Known=0.783862、First=0.333333、
  reuse=0.098039、CT-Reuse=0、NMI/ARI=0.664961/0.416260；同样失败。
  两个 CSV 的 causal contract 全通过。
- 2026-08-21: no-consistency 首次启动因未设置 `PYTHONPATH=.` 且日志目录
  未创建而在训练前失败；修复运行路径后用相同 seed/config 重跑成功。无 OOM、
  无 near-OOM、无其他任务进程被终止。
- 2026-08-21: 最终结论：
  `PHASE10_SMALL_PROTOTYPE_FAILED_STOPPED` /
  `CT_REUSE_NOT_RECOVERED_NO_4GPU_TRAINING` /
  `TRACKOCD_NOT_YET_ICLR_LEVEL`。报告：
  `docs/iclr27_phase10/PHASE10_COMPLETE_COPYABLE_REPORT.md`。

## Phase 11（2026-08-21，最终）

- 先完成 Q1 Stage-1 opportunity audit，未在诊断前启动训练。完整 Q1 私有
  GT novel population 为 22 tracks / 13 categories / 13 same-category
  pairs / 5 cross-video pairs；corrected DSCT strict-visible subset 仅 11
  tracks / 9 categories / 2 pairs / 1 cross-video pair。novel-track/category/
  pair/cross-video coverage 分别为 0.50 / 0.6923 / 0.1538 / 0.20。结论为
  表示不足与 proposal opportunity 稀疏共同存在，不把 CT=0 归因于单一因素。
- 完成 DINOv2、TSE、Phase-8A B、OpenAI CLIP ViT-B/32 的 post-hoc GT-track
  geometry；full-population CLIP same/different distance=0.0632/0.1272、
  cross-video NN=0.0455，说明 language-aligned image feature 有弱语义结构但
  没有稳定 cross-instance invariant signal。
- 核验 InternVideo2、MoSiC、TrackVerse、VESSA、Beyond the Static World、
  ViLAMP 及 OpenAI CLIP 的官方论文/仓库，写入
  `docs/iclr27_phase11/PRIOR_ART.md`；不宣称已有方法满足 TrackOCD causal
  birth/reuse contract。
- 新增 `src/iclr27_phase11/`：Q1 CLIP crop extraction、causal 128-D GRU /
  gated residual trajectory adapter、known+unlabeled small training、冻结
  Phase-8A B replay。训练 5 epochs x 40 steps，使用 2,196 train-known tracks
  和 1,797 unlabeled predicted tracks；无 Q1 label、future、physical-ID。
- CLIP prototype Q1 DEV strict：Known=0.288184、First=0、Reuse=0、
  CT-Reuse=0、NMI/ARI=0.563454/0.235276、Known→Existing=0；2,947 rows
  replay 动作全部为 `known`，frozen B create/state 与新表示分布失配。因果
  contract 全通过，失败不是 protocol violation。
- 训练前 nvidia-smi 显示 GPU0=4 MiB/0%，host available≈101 GiB；仅 GPU0
  单卡小训练，GPU1/6–9 的其他任务未触碰，无 OOM/near-OOM。首次 extraction
  的 tee 日志目录错误在 Python 输出完成后修复，NPZ shape=2947x512 已验证。
- 最终状态：`PHASE11_CLIP_REPRESENTATION_FAILED_STOPPED` /
  `CT_REUSE_NOT_RECOVERED_STOP_ARCHITECTURE_TUNING` /
  `TRACKOCD_NOT_YET_ICLR_LEVEL`。停止 semantic-memory/lifecycle/threshold/
  calibration tuning，不启动 4-GPU 或 large training。报告：
  `docs/iclr27_phase11/PHASE11_COMPLETE_REPORT.md`。

## Phase 12（2026-08-21，最终）

- 先复核 Phase 8A–11 报告、当前代码/产物与资源状态；没有在诊断前启动
  大训练。GPU0–4、6–9 有其他任务，GPU5 初始空闲但随后被其他进程占用；
  未终止任何其他任务。Phase 12 全程没有 4-GPU 或 large training。
- Experiment 1 完成严格上界 replay：冻结 Phase-8A B decision/state、同一
  corrected Q1 DSCT stream/evaluator，仅用 strict proposal-to-private-GT
  mapping 为 45 条对齐 tracks 注入 category-consistent novel direction。
  这是显式 `oracle_label_used=true` 的 upper bound，不是合法方法；FP 行仍
  使用 ordinary causal B embedding。458 aligned occurrences 上 Known=1.0、
  first birth=1.0、novel reuse=1.0、CT-Reuse=1.0，9/9 novel categories
  完整复用，NMI/ARI=1.0；causal contract 全通过。早期 prototype/ID 对齐尝试
  未计入结果，最终 mapping/role 修复后仅使用
  `oracle_ideal_final_strict/summary.json`。
- Experiment 3 构造合法 synthetic OOD episodes，仅使用公开
  `known_tracks.npz` 与 class split：hidden-train 10 类/61 tracks，
  hidden-val 10 类，support 28 / query 31，训练无 Q1 label、private GT、
  future frame、physical-ID feature。小型 causal GRU + CE/supcon 的
  prototype accuracy=0.967742、nearest-support=1.0；frozen TSE mean
  baseline 同为 1.0。结果说明控制分布内的 cross-instance correspondence
  可学，但不证明 Q1 transfer。
- Experiment 2 核验官方代码/论文：InternVideo、MoSiC、TrackVerse、VESSA、
  OFCL、TraceAnything、SlotCurri、SSync、Beyond the Static World；没有把
  未核验的方法写成事实，详情见 `docs/iclr27_phase12/PRIOR_ART.md`。
- 资源/修复：首次 synthetic 命令只因 tee 日志目录不存在而在 Python 前
  失败，创建目录后用相同配置完成；oracle smoke test 修复了 state device
  参数、tensor centroid 转换、proposal ID 类型和 CSV role/GT mapping
  误用。无 OOM/near-OOM，最终脚本 py_compile、JSON/NPZ/checkpoint/CSV
  assertion 均通过。
- 最终判断：oracle 成功，TrackOCD 不是结构上不可行；当前 CT=0 主要来自
  cross-instance semantic representation/supervision/domain transfer，且
  strict DSCT 只有 1 个 cross-video same-category pair（full Q1 仅 5 个），
  benchmark sparsity 独立存在。继续仅限合法 representation/data coverage
  研究；不再添加 memory/lifecycle/threshold module，不启动 4-GPU。报告：
  `docs/iclr27_phase12/PHASE12_FEASIBILITY_REPORT.md`。

## Phase 13（2026-08-21，最终）

- 复核 Phase 8A–12 报告、代码、数据和资源；GPU0–2、5–9 有其他任务，GPU3
  空闲，最终仅使用 GPU3 单卡小训练。没有修改 semantic memory、assign/create、
  reliability、threshold 或 evaluator，也没有终止其他进程。
- Stage 0 核验 InternVideo2/2.5、StreamFormer、MoSiC、TrackVerse、VESSA、
  BYOV、SRL、Trace Anything、TRACT、OFCL 的官方代码/论文与 causal/future
  边界，写入 `docs/iclr27_phase13/PRIOR_ART.md`。InternVideo/StreamFormer
  未下载大 checkpoint，避免把 foundation 依赖与表示目标混淆。
- Stage 1 使用 TAO TRAIN 真实视频（不是 MOTSynth）：复用 Phase6D 已验证的
  DINOv2 bbox crop cache 和公开 boxes，构建 2,196 tracks / 483 videos / 48
  categories 的 8-frame appearance + 4-D causal box-motion asset；无 Q1
  labels/private GT/future/physical-ID。产物：
  `outputs/iclr27_phase13/dataset/real_tao_tracks.npz`。
- Stage 2–3 实现 appearance+motion causal GRU `TrackSemanticEncoder`，用
  public TAO train labels 做 legal known CE 和 held-out support/query episode
  simulation（25 train categories/2033 tracks，10 held-out categories，33
  support/37 query）；full loss=CE+temporal consistency+same-category
  alignment+episodic support/query。训练 6 epochs × 80 steps，四个 variants：
  full、no_semantic_alignment、no_temporal、no_episodic_unknown。
- 为直接测试“main method 不需要 category labels”，额外运行
  `self_supervised`（仅 temporal consistency，无 CE/alignment/episode，metadata
  的 `public_train_category_labels_used=false`；类别 ID 仅用于 held-out split）。Real-TAO held-out 上
  prototype/nearest=0.729730/0.594595，但 same/different distance 几乎相同
  （0.000203/0.000289）。
- Controlled real-TAO diagnostic：frozen DINOv2 mean prototype/nearest=
  0.837838/0.864865；full learned encoder=0.459459/0.540541，表示学习反而
  损伤 foundation geometry。没有把该离线诊断冒充 Q1 成功。
- Frozen Phase8A B Q1 strict replay（2,947 rows/20 videos）结果：full
  Known=0.495677 / first=0.111111 / reuse=0 / CT-Reuse=0 / NMI/ARI=
  0.702915/0.538270；no-semantic=0.475504/0.111111/0/0；no-temporal=
  0.469741/0/0/0；no-episodic=0.414986/0.222222/0/0；self-supervised=
  0.152738/0/0/0。五个 CSV 的
  no-future/no-relabel、memory legality、dual identity、immediate action、
  objectness contract 全部通过。未达到 Known>=0.60 且 CT>0。
- 资源：训练前执行 nvidia-smi/free/process，GPU3 单卡，五个小变体，replay/evaluator CPU，
  host available>67GiB，无 OOM/near-OOM，未启动 4-GPU/large training。第一轮
  replay shell 提前返回时 no-temporal/no-episodic 仍在运行，按阻塞策略等待至
  完成，产物完整；不是实验失败。
- 最终判断：`PHASE13_FAILED_NEED_NEW_DIRECTION`。真实视频轨迹和显式 motion
  流可合法构建，但该 GRU/temporal/alignment/episode 组合未恢复 CT-Reuse，
  停止继续调这条架构；下一步只能另行注册更强 video/object-centric foundation
  或更好的 cross-video novel benchmark coverage。报告：
  `docs/iclr27_phase13/PHASE13_COMPLETE_REPORT.md`。

## Phase 14（2026-08-23，最终）

- 先复核 Phase 8A–13 报告、代码、产物和资源；冻结 Phase 8A B decision/state、
  semantic memory、lifecycle/reliability、threshold、causal evaluator 与
  physical stream。训练前执行 `nvidia-smi`、`free -h` 和进程检查；只在
  GPU2 空闲时做一次单卡冻结 DINOv3 特征提取，没有启动训练、四卡任务或终止
  其他进程。
- Q1 audit：完整 novel GT 为 22 tracks / 13 categories / 13 same-category
  pairs / 5 cross-video pairs；修正 DSCT strict 子集为 11 / 9 / 2 / 1，
  覆盖率 0.50 / 0.6923 / 0.1538 / 0.20。结论是 benchmark opportunity
  稀疏与跨实例 representation 不足同时存在，不能把 CT-Reuse=0 单独归因
  于 evaluator。
- DINOv3 W4 冻结 Q1 crop 特征产出 2,947x768，`failed_rows=[]` 且
  `q1_labels_used/private_gt_used/future_used/physical_id_used=false`。
  仅替换表示、保留 B：Known=0.005764、CT-Reuse=0；causal/memory/identity/
  objectness contracts 全部通过。DINOv2 baseline replay Known=0.412104，
  与 Phase 8A 一致。
- 完成 DINOv2、TSE、Phase 8A B、CLIP、DINOv3 CLS/pooled 的 feature-only
  geometry 与 Q1 frozen-B benchmark；结果和可复现实验入口见
  `outputs/iclr27_phase14/eval/feature_benchmark.json` 与
  `docs/iclr27_phase14/PROTOCOL.md`。
- 核验 2025/2026 官方代码、checkpoint、license、数据/类别监督与 future
  边界，写入 `docs/iclr27_phase14/PRIOR_ART.md`。没有候选同时达到
  Known>=0.60 且 CT-Reuse>0；登记 object-centric entity representation/
  track-level correspondence 为唯一合理下一方向，但 Phase 14 不训练它。
完整报告：`docs/iclr27_phase14/PHASE14_COMPLETE_REPORT.md`。

## Phase 14B（2026-08-23，最终）

- 预注册 Q1 quarantine、TAO TRAIN annotation-only split、GT-box diagnostic /
  primary proposal 双视图、prefix `{1,2,4,8,16}`、400 组 bootstrap、冻结
  Phase-8A B gate 与不调阈值协议；Q1 未用于选择或 replay。
- 从公开 TAO TRAIN 固定选出 20 个 novel 类别、193 tracks / 130 videos，
  1,044 same-category physical pairs / 941 cross-video pairs。原始 Phase6D
  cache 缺失的 `282_1802` 已从当前帧 causal 重抽，DINOv2/CLIP/DINOv3 三个
  canonical feature cache 均覆盖 193/193；primary proposal stream 仍未完成，
  未用合成 proposal 替代。
- 冻结 foundation benchmark：prefix16 cross-video R@1 DINOv2 0.6891、
  DINOv3 0.6736、CLIP 0.5026；DINOv2 mAP 0.5271，category-grouped
  95% CI [0.5918, 0.7851]。只报告 GT-box representation diagnostic。
- DEV+ oracle（非法 hidden category）20 births / 173 reuse / 158 cross-video
  reuse，三项 novel/reuse accuracy=1.0；instance-only temporal-delta R@1=
  0.2902；public-TRAIN-only PCA+LDA supervised diagnostic R@1=0.5959，明确
  标记 `category_label_used=true`，不冒充合法方法。
- OVTR 官方权重单卡 build/load smoke 成功，forward 暴露官方 evaluator-owned
  `model.ious_thresh` runtime contract；COVTrack 两个已存在环境分别被 MMCV2
  `mmcv.parallel` 与 MMCV1 缺少 `clip` 阻塞。无 OOM/near-OOM、无训练、无其他
  进程终止；所有失败已记录到 Phase14B logs。
- 决策：E stop。primary proposal view 未达到机会目标，未运行 strict B/TrackOCD
  gate，不宣称 ICLR-ready。条件 Phase15 只登记 explicit cross-instance
  correspondence projection；不再添加 memory/lifecycle/reliability/threshold。
完整报告：`docs/iclr27_phase14b/PHASE14B_COMPLETE_REPORT.md`。

## Phase 14C（2026-08-24，进行中）

- 先完成新的冻结协议、资源 ledger 和混合 sidecar。保留 Phase14B 的
  20 novel 类别/130 DEV+ 视频；同视频已有 487 supported-known tracks / 29
  known 类，因此不添加 KNOWN+。model-facing OVTR annotation 只有 4,596 张
  图像、空 `annotations`/`tracks`；GT 标签只写入 evaluator sidecar。
- 使用官方 Phase6B DSCT wrapper 与真实 TAO TRAIN frame symlink。第一次 smoke
  因 calibration smoke 视频不在 DEV+ filtered annotation，合法地得到 0 samples；
  修复为独立无标签 calibration annotation/config。第二次 model/official
  tracker 完整处理 2 videos / 72 frames、输出 362 rows / 79 physical tracks；
converter 初次误用了 DEV+ annotation（image-ID 域不同），修正后已通过
two-video schema/serialization validation。另修正 smoke assertion 的
`all(bool)` 小错误。没有 OOM、没有停止其他进程，Q1 未读取。

### Phase 14C completion (2026-08-24)

- Official DSCT proposal runner completed all 130 DEV+ videos / 4,596 frames
  on idle GPU5 in one blocking wait.  The physical-only stream has 16,616
  rows / 2,056 tracks / 130 videos; alignment gives 464 GT tracks (311 known,
  153 novel), 657 cross-physical and 580 cross-video same-category pairs.
  `proposal_opportunity_audit.json` therefore passes the preregistered
  100/30 opportunity gate.  Proposal/feature key-order integrity is exact.
- Raw proposal DINOv2 retains prefix-16 cross-video R@1=0.503268, R@5=0.732026,
  mAP=0.435157, prototype=0.575163, gap=0.173643 over 153 queries; frozen
  TSE/B fall to R@1=0.359477/0.281046 and gap=0.165007/0.072563.  This is the
  registered condition for one residual projection pilot.
- Exact and TRAIN-normalized frozen Phase-8A B replay are identical:
  Known=0.294591, CT-Reuse=0.007024 (8/1,139), gate fail.  The rank-32
  DINOv2->TSE residual main (public supported-known TRAIN only; 1,546 train
  tracks, 163 calibration tracks; seed 20260824; AdamW 1e-3; early stop at
  step 600) gives Known=0.310402 and CT-Reuse=0.002523 (3/1,189).  The
  no-cross-instance control gives Known=0.390569 and CT-Reuse=0.  Neither
  satisfies Known>=0.60 AND CT-Reuse>0; no final lock or Q1 run is allowed.
- DSCT objectness audit passes: Pearson(base-known, joint-objectness)=0.182721,
  Spearman=0.133005, 2,623/16,616 admitted rows have base-known<0.3.  Strict
  oracle/wrong-label controls are 1.0/0.0; causal contract is PASS (immediate
  action, no future/relabel, physical ID != semantic ID, no Q1/private GT).
- Three bounded repairs were recorded: separate label-free smoke annotation,
  smoke converter annotation-domain fix (plus assertion), and semantic CSV
  writer compatibility.  No OOM/near-OOM, storage recovery, or other-user
  process termination occurred.  Large `dscq_stats.json` (~1.21 GB) remains
  auditable.  Final status is
  `C_PROPOSAL_AND_RAW_SIGNAL_PASS_BUT_SYSTEM_GATE_FAILED`; report:
  `docs/iclr27_phase14c/PHASE14C_COMPLETE_REPORT.md`.

## Phase 15 (2026-08-24, registered plan before new results)

- Hypothesis: raw DINOv2 geometry that survives proposal crops may support
  category-disjoint cross-instance correspondence, but a learned relation and
  explicit causal KNOWN/EXISTING/NEW interface are needed to avoid Phase14C's
  duplicate-state explosion.  First run a cheap raw cosine/exemplar and
  relation-verifier ceiling/localization probe; do not assume Phase8A B is
  frozen and do not start a large run before this evidence.
- Data plan: public TAO TRAIN `full_tao_tracks.npz` only.  Use the existing
  `class_split_hard.json` category groups with priority-disjoint videos:
  train-visible categories for representation training, hidden-train
  categories for calibration, and hidden-val categories for category-disjoint
  meta-validation.  The fixed Phase14C proposal-aligned 130-video DEV+ stream
  remains evaluation-only; Q1 remains quarantined.
- Phase15A comparisons: raw cosine nearest-neighbor/prototype/exemplar bank,
  one small pairwise verifier with hard negatives, a temporal-only control, and
  one causal linker with calibration-only thresholds.  Report offline pair
  ROC/PR/retrieval/calibration and strict online actions; choose Branch A/B/C/D
  from category-disjoint evidence before any Phase15B training.
- Resource plan: one idle GPU5 at most for the cheap probe; no four-GPU job
  unless a branch is justified.  Before each GPU job run nvidia-smi/free/process
  count/df and wait with one blocking shell command.  Preserve all Phase14C
  symlinks and artifacts.

### Phase 15A completion (2026-08-24)

- Preregistered public TAO TRAIN roles are video/category disjoint:
  representation train 1,140 tracks / 240 videos / 14 present categories,
  calibration 65 / 38 / 11, and meta-validation 113 / 81 / 13.  The leakage
  audit passes with zero role overlaps; DEV+ and Q1 labels were not used for
  fitting or calibration.
- Verified official prior art before selection: Meta V-JEPA2/2.1 (2025/2026),
  Meta DINOv3 (2025), and ICML-2026 Grounded Correspondence/Rethinking OCL.
  No external weights were downloaded; Phase15A uses the existing raw 768-D
  DINOv2 cache.
- Phase15A compared raw cosine, leave-one-out category prototypes, a bounded
  four-exemplar diagnostic, a two-seed relation MLP, and a within-track
  temporal-only control.  The formal relation fit used public prefix 8,
  balanced cross-video positives plus random/hard negatives, 600 AdamW steps,
  seeds 20260824/20260825, and calibration-only 100-point thresholds.
- Category-disjoint meta prefix-8 results: raw R@1/R@5/mAP/AUC
  `0.9074/0.9722/0.6410/0.7447`; temporal-only
  `0.8056/0.9352/0.5179/0.6673`; relation seeds averaged
  `0.1389/0.3565/0.2620/0.4688`.  Relation transfer is therefore weak and
  below raw on retrieval, mAP, pair AUC, and probability gap; category
  prototypes are an offline label-assisted diagnostic, not a legal causal
  method.
- Strict DEV+ causal replay over the fixed 16,616-row / 2,056-track stream:
  raw cosine Known `0.228849`, CT-Reuse `0.027596` (`38/1,377` eligible
  cross-video occurrences; one category across two videos); relation seed 20260824 Known `0.322330`, CT `0`;
  relation seed 20260825 Known `0.297365`, CT `0`; temporal-only Known
  `0.472677`, CT `0`.  No candidate reaches the final
  `Known >= 0.60 AND CT-Reuse > 0` gate.  Relation births were 630/739 versus
  raw 70; relation novel reuse was zero; all causal/leakage/identity contracts
  pass.
- The first formal implementation attempt was stopped after its CPU-bound
  per-occurrence MLP calls remained unacceptably slow.  The smallest justified
  repair was global top-eight raw prefiltering plus causal physical-track
  carry-forward; a GPU smoke passed and the single formal GPU5 run completed
  in 79.3 seconds.  No OOM/near-OOM occurred and no other-user process was
  terminated; only two task-owned smoke PIDs were stopped.
- Preregistered Branch D is selected: category-disjoint relation transfer is
  weak.  The one bounded non-causal max-frame-pair tube diagnostic is worse
  than mean prefix-8 (R@1 `0.6481`, mAP `0.5333` versus `0.9074`, `0.6410`),
  so no Phase15B training, foundation replacement, or Q1 replay is justified.
- The initial runner's branch field reported C because its helper checked the
  Known-only condition before the explicitly weak-transfer condition.  A
  post-run audit recomputed the branch from the immutable summaries, applied
  the protocol-consistent D precedence, and changed no metric, seed, threshold,
  or data; the correction is recorded in the formal log.
- Final status: `PHASE15_D_STOP_RAW_FOUNDATION_CORRESPONDENCE_NOT_TRANSFERRED`;
  report: `docs/iclr27_phase15/PHASE15_COMPLETE_REPORT.md`.

## Phase 15R (2026-08-24, preregistered before repaired DEV+ replay)

- Hypothesis: Phase15 online conclusions may be partly caused by four concrete
  audit defects (prefix-1 carry-forward, repeated NEW accounting, one shared
  threshold, and GT-box/proposal bank mismatch).  Repair the state machine and
  rerun frozen raw DINOv2 before considering any new foundation.
- Created new-only Phase15R protocol/ledger/preregistration and symlinked all
  historical inputs.  The project has no Git metadata; source hashes are the
  revision record.  Current preflight leaves 199G on /data1 and 87G available
  RAM; GPUs 0/1/4/5 are idle while 2/3/6–9 are occupied by other tasks.
- Exact public TRAIN DSCT proposal features were not present on disk.  The
  existing Phase4T TRAIN-only tracker-induced proposal cache is registered as
  a crop-domain diagnostic only, not silently treated as DSCT.  Q1 remains
  closed and no foundation checkpoint is downloaded before the branch decision.

### Phase 15R completion (2026-08-24)

- Repaired the four registered defects with new-only code: cumulative prefix
  updates before every action, one legal NEW birth per semantic state,
  separate `tau_known`/`tau_existing`/`delta_new`, and an explicit
  proposal-aligned bank.  The independent transition validator and focused
  tests pass; every final candidate has NEW-action count = unique births =
  internal state count, and prefix/truncation audits pass.
- Built a legal exact DSCT TRAIN subset after the historical cache search found
  no public DSCT TRAIN artifact.  The subset is 18 public TRAIN videos / 641
  frames, covers every representation-train known category in the annotation,
  and uses the frozen Phase-6B stage-D checkpoint.  DSCT produced 1,900 rows:
  398 known matches, 83 novel matches, and 1,419 false proposals.  After
  proposal matching, only categories 41/99/429/805 yielded supported-known
  bank rows (149 rows, 38 physical tracks), so the bank coverage limitation is
  reported rather than attributed to the semantic representation.
- One GPU6 DSCT inference and one GPU6 frozen DINOv2 extraction (1,900 x 768,
  context 0.10, no failures) completed.  A first Python-3.8 DINOv2 load failed
  on modern union annotations; a batch-64 attempt reached about 35 GiB during
  transfer and was stopped by this task before OOM.  Adding the explicit
  PyTorch-1.10 attention fallback and rerunning at batch 16 passed.  No swap,
  host OOM, or other-user process termination occurred; all incidents are in
  the Phase15R resource summary.
- Final repaired raw-DINOv2 strict audit: cumulative exact-DSCT bank
  Known=`0.00638`, CT=`80/164=0.48780`; four-exemplar exact bank
  Known=`0`, CT=`98/126=0.77778`; historical GT and proxy controls also fail
  Known>=0.60.  CT spans multiple categories/videos in the eligible stream,
  but duplicate creation remains 1.0 and the known-bank coverage is only four
  categories.  The formal gate is false for every candidate.
- The registered bounded R-C ROI patch-token diagnostic kept context=0.10 and
  froze DINOv2.  On the exact public TRAIN proposal subset it improved track
  correspondence over CLS (R@1 `0.7833 -> 0.8833`, ROC-AUC
  `0.6275 -> 0.7126`, PR-AUC `0.5233 -> 0.5869`, positive-negative gap
  `0.0476 -> 0.1003`).  This is a useful crop/object formation signal, but it
  has no repaired DEV+ online evaluation and therefore does not satisfy the
  registered stable category-disjoint plus online-relevance condition for
  Phase16B training.
- Decision: `R-C` (proposal/crop domain coverage is the dominant unresolved
  loss), `foundation_audit_opened=false`, `phase16_training_authorized=false`,
  and `q1_opened=false`.  No DINOv3/V-JEPA checkpoint was downloaded.  Final
  report: `docs/iclr27_phase15r/PHASE15R_COMPLETE_REPORT.md`.

## Phase 15S/16 (2026-08-24, preregistration and confound audit)

- Hypothesis: the Phase15R gate is not identifiable until its DSCT bank
  coverage, self-state/known competition, pairwise calibration, and
  prediction-conditioned CT denominator are separated.  The registered plan
  is to construct video-disjoint public-TRAIN known roles, expand a
  programmatic exact DSCT bank to a ceiling above 0.60 (preferably 0.80), use
  frozen matched DINOv2 CLS/ROI features, and evaluate fixed-denominator CT.
  Q1 remains quarantined; Phase16 foundation/training is conditional only on
  the registered evidence branch.
- Independent read-only reproduction before edits: the frozen DEV+ aligned
  stream has 3,605 supported-known rows across 28 categories; the historical
  exact Phase15R subset has 18 videos (only 8 representation-train videos),
  1,900 proposals/398 known matches, and the exact bank retains 149 rows on
  38 proposal tracks.  The proposal summary has 22 matched categories (not
  four).  Exact-bank categories {41,99,429,805} overlap DEV+ Known only at
  {41,805}, covering 2,117/3,605 (ceiling 0.5872399445).  The cumulative
  exact replay sends 3,532/3,605 known rows to `existing`.  Historical
  `tau_known`, `tau_existing`, and `delta_new` are pair means/midpoint/std;
  CT eligibility is assembled from predicted states/births, not a fixed GT
  denominator.  These values match the immutable Phase15R artifacts.
- Plan: create only `docs/iclr27_phase15s/`, `outputs/iclr27_phase15s/`,
  `src/iclr27_phase15s/`, and `data/iclr27_phase15s/`; preregister fixed CT,
  legal roles, coverage expansion, controller hierarchy, chronological
  calibration, matched CLS/ROI candidates, branch rules, and resource/Q1
  constraints before any new DEV+ result.  Reuse large historical inputs via
  symlinks; do not modify Phase15/15R artifacts.
- First bounded DSCT launch failed before inference because the frozen OVTR
  loader selects its TAO parser from a `validation`/`test` annotation-path
  token, while the new file was named `dsct_public_roles.json` and lacked
  `coco_url` fields expected by the alternate LVIS parser.  The smallest
  protocol-preserving repair is a new `validation_public_roles.json` output
  and config path; the failed unit produced no proposal output and no memory
  incident.  Its `.launched` marker is cleared only for this repair, followed
  by a smoke parser check and one resumed blocking run.

### Phase 15S/16 completion (2026-08-24)

- The repaired public-role DSCT expansion exhausted 370 legal TAO TRAIN
  videos (320 bank / 40 calibration / 10 audit), selected 12,000 frames, and
  produced 43,423 rows (3,781 known, 1,642 novel, 38,000 FP).  The bank has
  33 matched supported-known categories and covers 3,442/3,605 frozen DEV+
  Known rows, giving a legal ceiling of `0.9547850208044383`; S-F is therefore
  rejected.  No DEV+ frequency or GT crop was used.
- Matched frozen DINOv2 ROI patch means improve the public bank diagnostic
  over CLS (R@1 `0.821872` vs `0.782933`, ROC-AUC `0.755851` vs `0.623811`,
  PR-AUC `0.728003` vs `0.604456`).  Public calibration ROI Known is
  `0.808511`; the registered chronological grid used 2,000 rows, four prefix
  observations per track, and seeds 20260824/20260825.  The audit role has no
  matched known DSCT rows and is reported as an unavailable opportunity, not
  filled with GT.
- Fixed-CT controls and tests pass: the GT-only denominator is 1,228 for every
  candidate; oracle/wrong-label/all-new/all-one-state are `1.0/0/0/90/1228`.
  The decoupled controller evaluates known first, excludes self-state from
  cross-physical evidence, requires another video for cross reuse, and treats
  overflow as invalid.  Both CLS and ROI transition contracts are valid.
- Frozen DEV+ strict replay fails the joint gate for both candidates.  CLS is
  Known `0.363384`, fixed CT `2/1228 = 0.001629` (legacy diagnostic `2/2`,
  explicitly not recall); ROI is Known `0.399168`, fixed CT `0/1228`.  Public
  strength therefore does not transfer online to DEV+.
- The one preregistered bounded S-D input/proposal diagnostic finds mean
  matched alignment IoU `0.700538` on public bank known rows versus `0.261925`
  on DEV+ supported-known rows (all-proposal means `0.175383` vs `0.090822`),
  with different source-family composition and public/DEV+ mean ROI-vector
  cosine `0.234388`.  It is diagnostic-only: no threshold/model fitting,
  foundation swap, training, or Q1 was opened.  Final branch is
  `S-D_PROPOSAL_DOMAIN_SHIFT`; TrackOCD remains not ICLR-level.
- Resource incidents: the frozen DSCT parser required a validation-named
  annotation path (repaired before consuming output), and the first slow CPU
  calibration implementation was interrupted by this task and replaced by
  vectorized scoring with unchanged data/grid/seeds.  DSCT and feature jobs
  used idle GPU2 only; there was no OOM, near-OOM, swap use, or termination of
  another task.  A post-run recursive JSON sanity check accidentally included
  the large raw DSCT JSON; its two task-owned validation PIDs were stopped at
  about 20GB transient RSS, before the host safety floor was crossed.  No
  experiment artifact was affected.  Final report:
  `docs/iclr27_phase15s/PHASE15S_COMPLETE_REPORT.md`.

## Phase 17 (2026-08-25, registered proposal-shift identifiability repair)

- Hypothesis: the Phase15S S-D proposal/input explanation is not yet
  identifiable because public `gt_iou` is frame-local and threshold-truncated,
  DEV+ `gt_temporal_iou` is a track-level temporal average, the reported
  domain-mean feature statistic is an unnormalized dot product, and the public
  calibration/novel CT roles are too small.  Phase17 will first recompute common
  frame/temporal alignment and opportunity strata, then run paired public
  crop-quality diagnostics before authorizing any representation pilot.
- Independent reproduction before edits: public DSCT has 43,423 rows
  (3,781 known / 1,642 novel / 38,000 historical FP); all 3,781 known rows
  have frame-local `gt_iou >= .5` and mean `.700538`.  DEV+ has 3,605
  supported-known rows with copied track-level `gt_temporal_iou` mean
  `.261925`.  Public calibration has 69 known rows in four categories
  (`805:43, 99:19, 211:6, 95:1`); the historical 2,000-row prefix retains
  47 known rows and one fixed-CT-eligible novel row.  The known audit role has
  zero known rows and zero CT denominator.  The frozen DEV+ denominator remains
  `1,228`, with oracle/wrong/all-new/all-one-state controls unchanged.
- Phase17 preregistration will use only new paths, keep Phase15/15R/15S files
  immutable, and prohibit DEV+/Q1 selection.  Public role construction will
  seek >=300 known calibration rows/10 categories/20 videos, >=200 known audit
  rows/8 categories/10 videos, >=100 novel-calibration CT rows/5 categories/10
  pairs, and >=50 novel-audit CT rows/3 held-out categories/5 pairs.  If legal
  public DSCT rows cannot satisfy these gates, the terminal branch is
  `P17-F_CALIBRATION_OPPORTUNITY_BLOCKED`; no threshold will be called
  calibrated and no PQIR training will start.

### Phase 17 completion (2026-08-25)

- Common alignment rebuilt public and DEV+ with the same positive mean exact
  source-frame IoU greedy one-to-one assignment. Assigned rows below .5 were
  retained; unmatched rows remain FP. Public common assigned supported-known
  mean frame IoU is `.187398` (11,205 rows) and DEV+ is `.219366` (3,605 rows).
  The old `.700538` public frame-filter mean versus `.261925` DEV+ temporal
  mean is therefore only `PARTIALLY_SUPPORTED`, not an apples-to-apples domain
  statistic. Normalized ROI centroid cosine is `.971433`, balanced RBF-MMD²
  `.012585`, energy distance `.027862`, and video-grouped domain AUC `.705864`;
  the old unnormalized mean-vector cosine is retired.
- Public roles are video-disjoint across all 370 legal TRAIN videos: 300 bank,
  20 known calibration, 10 known audit, 20 novel correspondence train, 14
  novel calibration, and 6 novel audit. Gates pass: bank DEV+ Known row
  coverage `.954230` (3,440/3,605), known calibration 1,109 rows/28 cats/20
  videos, known audit 317/11/10, novel calibration 237 fixed-CT rows/5 cats/15
  cross-video track pairs, and novel audit 77/3 cats/5 pairs. Every online role
  contains FP and competing assigned known/novel states.
- Paired public crops (1,000 occurrences, 11 views) and a 20-occurrence
  diagnostic-only DEV+ watermark ran with frozen DINOv2 and local DINOv3. At
  the recorded 224px paired DINOv2 resolution, GT-to-raw cross-video R@1 drop
  is `.160494`; local DINOv3 dense drop is `.115600`. Both video-bootstrap CIs
  exclude zero, cover >=3 categories and >=2 videos, and have the same DEV+
  direction: registered branch G1 proposal sensitivity passes.
- Conditional one-GPU PQIR P1/P2 pilots (600 steps each, frozen DINOv3) improve
  low-quality R@1 by `.164223`/`.184751` with clean-view drop `-.046016`/`-.050505`;
  both pilot gates pass. No tracker, memory, semantic lifecycle, or online
  controller was retrained; GT was teacher-only and no Q1/DEV+ labels entered
  fit.
- Complete public calibration used all 4,325 role rows and three video orders
  at the single historical operating point (`tau_known=.15`,
  `tau_cross=.15`). Fixed-CT denominator is 237, but CT is `0` in every
  order, so `reuse_threshold_unidentified=true`; no tie-break is evidence.
  Public final audit is Known `.35`, fixed CT `0/77`, predicted-existing
  precision `0`; transition contract is valid. The public gate fails, so no
  DEV+ replay and no Q1 run is legal.
- Terminal decision: `P17-E_CONTROLLER_CALIBRATION_LIMIT`. The evidence
  separates a genuine proposal-quality bottleneck (paired diagnostic/PQIR)
  from an unidentifiable unchanged causal controller. No further memory,
  lifecycle, tracker, or threshold tuning is authorized.
- Resource incidents were bounded CPU-time repairs only: two task-owned paired
  extraction processes and one full controller sweep were stopped before any
  artifact was written, then rerun with recorded smaller diagnostics/single
  operating point. No OOM, near-OOM, swap use, or other-user process
  termination occurred. Final artifacts are under `docs/iclr27_phase17/` and
  `outputs/iclr27_phase17/`.
## Phase 17R training-first (2026-08-25)

- Hypothesis: Phase17's training-contaminated 1,000-row PQIR result did not
  justify rejection; corrected full-population observability-gated semantic
  training could restore Known and nonzero CT without changing tracker/memory.
- Validity failure reproduced: `.15/.15/0` only, all 1,903 audit actions KNOWN,
  episode orders re-sorted away, P1/P2 same path over 950 fit + 50 audit rows,
  raw=temporal, full track length and 640×480 geometry used, wrong bootstrap
  label, and 5,513/11,205 assigned-known rows at exact IoU zero.
- Repair: rebuilt 43,423 rows using actual dimensions, causal prefix/stability/
  smoothing and three immutable event ranks; frozen split is 37,195 train /
  4,325 calibration / 1,903 untouched audit with zero row/video leakage.
- Features: full 43,423×6×768 DINOv3 cache extracted frame-once on GPU4/5/6;
  37,077 temporal features differ from raw. FP32 backbone forward was required
  because local torch1.12 AMP was non-finite; artifacts are validated float16.
- Training: T0 completed 6,000 total updates (step-3,000 model resumed in BF16
  after repeated FP16 loss-scale overflow, optimizer state unavailable). M1
  completed uninterrupted 12,000 BF16/DDP updates on GPU4/5/6/8, 10.324 full
  unique-row passes; calibration selected step 1,000.
- Result: M1 audit known closed top-1 `.665217`, online Known `.226087`,
  observability AUROC/AUPRC `.633089/.194632`, fixed CT `0/30` for all three
  orders, predicted-existing precision `.115175`. T0 closed `.669565` but
  online Known `.158696`, CT `0/30`.
- Cause: every genuine-order CT denominator has 30 rows across 3 cats/4 videos
  but zero rows at exact IoU>=.5. Oracle known + oracle observability remains
  CT `0/30`; semantic-label oracle is `30/30`. The reliable-update immediate
  action contract is therefore information-limited on the frozen denominator.
- Decision: reject additional retraining/architecture fallback under the
  preregistered oracle stop; `P17R-T8_IMMEDIATE_ACTION_CONTRACT_INFORMATION_LIMIT`.
  Public gate fails, so DEV+ and Q1 remain closed.
- Required causal-prefix curves were computed only after the frozen full
  decisions. Across cumulative minimum prefix counts 1/3/5/9/17, Known is
  `.226087/.239362/.254125/.290323/.213115` while every nonempty fixed-CT
  subset remains zero; the curves did not select a checkpoint or threshold.
- Resources/incidents: no OOM/near-OOM/swap/process termination. Unrelated GPU
  occupancy caused explicit device changes; one short T0 smoke lacked the
  later hard-abort despite a transient occupancy print and is disclosed.
Full details: `docs/iclr27_phase17r/PHASE17R_TRAINING_FIRST_COMPLETE_REPORT.md`.

## Phase 18 deferred semantic memory feasibility (2026-08-26)

- Hypothesis: the corrected causal cross-video CT task is identifiable, but a learned deferred semantic memory may fail because readiness and cross-instance correspondence are jointly difficult.  The frozen Phase18 census contains 43,423 rows, 11 eligible novel categories, 28 target tracklets, 41 positive events, 41 matched negatives, 435 post-prefix rows, and 15 positive events with unreliable prefixes.  Held-category/video-disjoint folds and fixed denominators were locked before fitting; DEV+/Q1 access remained prohibited.
- Phase17R terminal diagnosis was independently reproduced.  The three historical categories have no two-sided reliable CT opportunity; `oracle_both_routing` still uses a learned pair scorer and local-first routing.  This corrected the interpretation without editing Phase17R artifacts.
- Baselines: B1 causal DINOv2 tracklet prototype achieved Commit-CT `9/41`, post-prefix `80/435`, existing precision `.6923`, false merge `.0488`, Known macro `.1390`.  B2 fold-local balanced logistic pair scoring exactly matched B1, showing that the legal pair population (only 3–8 positives per fold) is the limiting baseline opportunity rather than an untested scorer.  B0 historical replay has zero target-row overlap and is retained only as a compatibility/coverage control.
- Main DSTM seed 1801 completed all 80,000 cross-fit updates with finite gradients but collapsed to DEFER: Commit-CT `0/41`, post-prefix `0/435`, pre-prefix defer `1.0`, unresolved `.9024`, and 8 duplicate births.  The legal O1 semantic oracle remains `41/41`, so the task is not evaluator-impossible.
- The only authorized repair, R1 (two-stage readiness then identity, readiness-masked DEFER, exact-reliable identity CE, and calibrated recall-constrained threshold), was run for seeds 1801/1802/1803.  Commit-CT was `8/41`, `12/41`, and `9/41`; pooled `29/123`, existing precision about `.457`, false merge about `.228`, Known macro mean about `.054`, and correct category/video coverage only 1–3 / 3–6.  Recovery on the 15 unreliable-prefix events was 6/5/5 versus B3 zero, but the preregistered method and DEV+ gates still failed.
- Required ablations completed: B3/no-DEFER-no-merge `0/41`, no-merge `3/41` with precision `1.0` but `.7073` unresolved, and no-history `1/41` with `.8415` unresolved.  The first B3 attempt exposed an empty-anchor shape failure; the smallest protocol-preserving repair updated the intentionally contaminated current-observation anchor.  A 5-step smoke and transition/causality/alignment regression passed.  No OOM, near-OOM, swap use, or other-user termination occurred; one task-owned wait handle was reclaimed while children completed and marker checks prevented duplicate launches.
- Final decision: `P18-T3_PROTOCOL_IDENTIFIABLE_MODEL_FAILS`.  Stop Phase18 memory/lifecycle/decoder/threshold tuning; preserve the protocol, denominator, baselines, oracle contracts, official-method audit, and negative results.  Any next study must first establish stable cross-instance semantic correspondence under the same causal contract before proposing another online architecture.  Complete report: `docs/iclr27_phase18/PHASE18_DEFERRED_SEMANTIC_MEMORY_COMPLETE_REPORT.md`.

## Phase 19 closed-loop OCD (2026-08-26)

- Hypothesis: Phase18's teacher-built/full-track state summaries put the
  learner on a different state distribution from the causal runtime. A shared
  rollout state machine with state-relative targets, scheduled sampling, an
  identity-preserving raw DINOv2 path, and a separate readiness head was
  preregistered before full training. Strict inputs were supported TAO TRAIN
  labels, legal pseudo-novel episodes, frozen DINOv2 CLS+ROI, and causal
  proposal geometry; true-novel labels and public event membership were
  masked from the trainer.
- Source audit: `src/iclr27_phase18/training/data.py:92-123,160-187` builds
  category-indexed, full-track state summaries and inserts the correct source
  into teacher candidates; `src/iclr27_phase18/evaluation/dstm_runtime.py:133-185`
  creates/updates states only after model-predicted actions. This confirms a
  training/runtime exposure mismatch rather than a missing cosine feature.
- Legal data: four deterministic folds and video-disjoint category splits
  were built without the 41 public events. The manifest SHA is
  `e4b17a4f4ba34e41b0edb2eb7ebcda069b090cfe32e2bb21db5ef0c3b64eab7b`.
  Held sets are F0=`805,347,229`, F1=`211,235,95`, F2=`382,133,579`, and
  F3=`41,429,81`; fit rows are 14,257/31,150/32,316/32,074 and held tracks
  are 194/30/16/13. Causal replay, illegal-label, no-future,
  physical-vs-semantic identity, and KNOWN/NEW/EXISTING/DEFER branch checks
  passed.
- Baselines under the repaired Phase19 internal evaluator were rerun after a
  malformed pre-repair Hungarian cell-count artifact was discarded. Frozen
  raw/AGE/TALON-style controllers are identical on this compact interface:
  mean L0 All/New/NMI=`.2797/.2328/.4390`, mean L2=`.2140/.1716/.3389`, with
  order sensitivity at most `.0224`; their raw public replay is kept as a
  newly implemented compatible baseline, while historical Phase18 B1/B2
  remain the exact comparison (`9/41`, precision `.6923`, false merge
  `.0488`).
- Main RA-OCD: all four folds completed the registered 40,000 updates with
  finite gradients (305,077 parameters), at least 60% zero-teacher scheduled
  on-policy budget, and 62.9-64.2 minute wall times. Internal means for
  L0/L1/L2 All accuracy were `.5026/.4708/.4899`; Novel NMI was
  `.0442/.0350/.0246`, ARI was negative, and discovery-count error was
  1.50/1.75/1.50. F0's high Hungarian number is a dominant single-cluster
  artifact and is not interpreted as semantic recovery. Main residual gates
  became negative (about `-.279` to `-.321`), while raw NMI was higher, so
  the preregistered internal trigger selected exactly Fallback-A before any
  public labels were joined.
- Selection audit: the trainer's per-checkpoint `best` file used the legal
  rollout-loss/state-existence proxy implemented in
  `train_rollout.py:200-214`, not the full weighted Hungarian/F1 formula
  written in the preregistration. The deviation is recorded in
  `outputs/iclr27_phase19/audit/selection_audit.json`; full held-known
  Hungarian replays and fallback choice were still performed before public
  label joins, so no true-novel selection leakage occurred.
- Fallback-A: four further full 40,000-update runs froze the residual gate at
  zero and trained 38,004 controller/prototype parameters (46.3-67.6 min per
  fold). Internal L0/L1/L2 means for All/New/NMI were
  `.4853/.4001/0`, `.4600/.5203/0`, and `.4652/.5690/0`; order sensitivity
  was zero but semantic NMI stayed zero. The all-supported-known final
  checkpoint also completed 40,000 updates (best step 16,000; 40.1 min).
- Freeze/evaluation: configs, fold manifest, fallback choice, checkpoint,
  code hashes, and raw predictions were frozen before evaluator-only label
  joins (`freeze_sha256=797459f29753671e925c66e4641690558153926a0527b9011168f55f7c842276`).
  On fixed 41 positive/41 negative development events, final Fallback-A got
  Commit-CT `17/41`, post-prefix `176/435`, category/video coverage `7/11`,
  existing precision `.2100`, negative false-merge `40/41=.9756`, 51
  duplicate births, new precision `.500`, and new event recall `16/41=.3902`.
  Known audit (317 rows, 31 tracks, 11 categories) was micro/macro `0/0`;
  main was `11/41` CT with precision `.1909` and false merge `.6585`. The
  raw/AGE/TALON-compatible controller was `1/41` with precision `1.0` and
  false merge `0`, but category coverage was only `1/1` and known
  micro/macro `.0631/.0150`.
- Public safety gate failed despite CT breadth: precision, false-merge, and
  known gates all miss by wide margins. DEV+ and Q1 remained sealed. The
  negative result rejects the rollout-alignment hypothesis and the raw-path
  controller as a TrackOCD solution; it does not show evaluator
  impossibility. No teacher-forced ablation or extra seeds were run because
  the preregistration requires those ablations only for a positive method.
- Incidents/resources: GPU0-3 were used only for the bounded four-fold runs
  (GPU4-9 external work was untouched); the final run used idle GPU3. Host
  memory retained at least 95 GiB available with no swap; no OOM, near-OOM,
  or other-user termination occurred. One post-freeze known-metric command
  initially used the base Python without torch and was rerun with the AVI
  environment; no model output changed. One earlier raw internal metric file
  had invalid >1 Hungarian values from the evaluator repair window and was
  overwritten by the corrected rerun.
- Terminal decision: `P19-T3_STRICT_ROLLOUT_AND_RAW_CONTROLLER_FAIL_PUBLIC_SAFETY_GATE`.
  Stop near-duplicate memory/lifecycle/adapter tuning. The next study must
  introduce independently justified cross-instance supervision or openly
  redefine the task, with known safety and false-merge gates fixed in
  advance. Complete report: `docs/iclr27_phase19/PHASE19_CLOSED_LOOP_OCD_COMPLETE_REPORT.md`.

## Phase 19R correctness repair (2026-08-26)

- Rebuilt the experiment in the independent `src/scripts/configs/outputs/iclr27_phase19r` namespace. Fixed E1--E10 with episode-conditioned known masks, balanced 24-step mixed streams, a single persistent state core, causal future-perturbation tests, frozen known-stage buffers, eligibility/video-disjoint folds, and official AGE/TALON/LTC source audits. The 10,000-episode-per-fold audit and T1--T7 passed; the synthetic T8 controller isolation overfit reached 1.0 on known, multi-state existing, and hard-negative NEW actions; Phase18 B1 exact parity was zero-delta.
- The first four-worker main launch was stopped explicitly after source review found the checkpoint selector still used a reduced event proxy rather than the preregistered seven-term macro score. This was a task-owned repair cycle (no external process was touched). The selector now consumes full persistent held-known replay with exact ExistingF1/NewF1/Known/Reuse/NMI/false-merge/fragmentation terms; an E8 two-step smoke and selection-field audit passed. Main training was relaunched only after that smoke.
- Before the second launch reached its first checkpoint, a final traceability audit added per-batch active/masked known-slot counts and an atomic temporary checkpoint around evaluator replay. The second launch was stopped explicitly (task-owned PIDs only) and the four-fold run restarted once more; this is the final allowed repair cycle. The corrected path has not changed the data, seed, architecture, thresholds or metric.

## Phase 19R corrective continuation (2026-08-27)

- The first corrected four-fold main run completed all 50,000 updates per fold with the exact persistent held-known selection evaluator. The internal gate failed and selected the preregistered Gaussian fallback; fallback and internal ladder evaluations were completed before any public label join.
- A final all-known run was then started on task-owned PID 20775/GPU3. After 09:32:34 wall time with no `final_rc_ms_latest.pt` or `final_rc_ms_best.pt` checkpoint, it was terminated explicitly with `SIGTERM` (no children; no external processes touched). The run had no recoverable progress; all logs/configuration were retained and the termination is not represented as a successful result.
- Root cause assessment: CPU-bound episode construction/state rollout dominated wall time; GPU was frequently idle. The supplementary Phase19R continuation therefore supersedes the earlier “complete final 50k” path: implement and validate semantically equivalent hard-pair/episode caching and event-aligned causal replay before considering another final run. Public new-model labels remain sealed.

## Phase 19R event-aligned stop (2026-08-28)

- The acceleration implementation was accepted for semantic equivalence: vectorized fold/split-local hard-pair caches, feature-free episode index shards, fast on-device state banks and fold-parallel workers preserved hard-pair choices, target actions, logits/loss and one AdamW step (all recorded deltas zero within the stated float tolerance). The old 500-update steady throughput was 0.877--0.907 updates/s; the cached/indexed path was 1.242 updates/s before the state fast path and 1.946 updates/s after it (1.592x over old, below the strict 2x target). First event-aligned four-fold runs reached 1.742--1.811 updates/s; the evidence-based commit-margin repair reached 1.538--1.701 updates/s because the extra loss term adds work. No OOM, swap, or memory-pressure incident occurred.
- Persistent-event mismatch audit: synthetic validation showed existing precision near 1.0 with existing recall around 0.35--0.42, while the persistent evaluator had zero existing precision/recall on the comparable mixed baseline. Event-level labels counted unresolved/over-defer 58, premature pre-prefix commits 48, duplicate target births 84, wrong existing state 24 and false merges 26 (events can receive multiple labels). These are observations; the possible causes “frozen DINOv2 lacks stable cross-instance separability” and “rollout/state evolution remains non-equivalent” remain hypotheses, not established facts.
- Legal pseudo-held event-aligned training used only supported TRAIN categories masked from the fold-known bank; no held/public/Q1 labels entered model inputs or selection. The first mixed+event run (4,000 updates/fold, 0.5 event ratio) produced aggregate persistent Commit-CT 2/76 (folds 0/12, 0/12, 0/24, 2/28), with no improvement over the authoritative mixed baseline 2/76. A single evidence-based repair targeting audited over-defer added `event_commit_margin` (weight 1.0); it produced 0/76 (0/12, 0/12, 0/24, 0/28), removed the two fold3 successes, and did not improve safety. Lower training loss is not treated as task success.
- Per the explicit stop rule, no 12k/16k extension, third tuning round, final 50k, final freeze or public evaluation was launched. The four repair workers and supervisor exited cleanly; their `.done` markers and prototype/checkpoint hashes are recorded in `outputs/iclr27_phase19r/audit/phase19r_corrective_decision.json`. Public new-model labels remain sealed because the internal gate failed. Decision code: `P19R_EVENT_ALIGNED_INTERNAL_GATE_FAILS_STOP_BEFORE_FINAL`.
- Next direction: establish a separately verifiable cross-instance semantic-correspondence/representation-learning baseline first, then design an online state controller. Do not continue threshold, lifecycle, memory or commit-weight tuning on this branch.

## Phase 20 proposal-aware correspondence (2026-08-28)

- Created the independent `docs/src/scripts/configs/outputs/iclr27_phase20`
  namespace.  Phase19R RC-MS-OCD, StateMemory, thresholds, action semantics,
  physical stream, and causal evaluator were read-only comparators.  Public
  TRAIN category/video metadata was the only label source; DEV+, Q1, and
  public new-model labels were not read and no public freeze/evaluation
  artifact was created.  The inherited four-fold manifest is a lightweight
  TRAIN-derived copy; proposal/features remain symlinked to existing caches.
- Stage 0 used all 76 positive and 76 negative pseudo-held events and every
  causal prefix in `{1,2,4,8,16}` on real frozen DSCT rows.  The reliable rule
  stayed `assigned=1 and row_iou>=0.5`, and the positive denominator stayed 76
  at every prefix.  Perfect-correspondence ceiling was 17/76, 22/76, 22/76,
  23/76, and 25/76 respectively; prefix16 has 27 source-no-reliable events
  and 24 target-no-reliable events.  The maximum 0.328947 ceiling fails the
  preregistered majority Gate O; all event-level failure reasons and IoUs are
  retained in `outputs/iclr27_phase20/audit/observability_events.{csv,json}`.
- Stage 1 was a no-training frozen DINOv2 audit with CLS/ROI causal mean,
  last, and max aggregation.  ROI mean prefix1 R@1 was `.3026` versus CLS
  `.2500`; pair PR-AUC was `.2567` versus `.2250` (full prefix and O strata
  are in `metrics/frozen_correspondence_baseline.json`).  This confirms some
  representation signal on visible proposals but does not overcome O and was
  not used to tune an online threshold.
- The single allowed Stage-H repair trained fold-local logistic proposal
  quality heads on causal score/geometry/stability fields from public TRAIN
  fit videos.  Its fixed-0.5 quality proxy raised apparent prefix16 coverage
  only to 31/76; true IoU coverage remained 25/76 and no proposal was
  created.  `proposal_quality_repair.json` therefore fails the repair gate.
- No Phase20 correspondence encoder, 4-GPU job, modern-backbone download,
  controller reconnect, final 50k run, or public evaluation was started.
  Preflight showed ten idle A100 40GB GPUs and 120G available RAM; CPU-only
  diagnostics completed with no OOM, swap, termination, or other-user impact.
  Terminal decision: `P20_GATE_O_FAIL_PROPOSAL_OBSERVABILITY_STOP_BEFORE_CORRESPONDENCE`.
- Root-cause evidence is limited deliberately: missing reliable proposal
  observations are established; DINOv2 semantic insufficiency and persistent
  rollout/controller mismatch remain hypotheses, not claims.  Next work must
  first improve proposal-domain/ROI observability and publish a verifiable
  cross-instance correspondence baseline before revisiting an online state
  controller.
- Integrity follow-up: after completion a duplicate task-owned Stage1 process
  PID 19198 (and the separately observed duplicate PID 19969) was found still
  running.  Both were stopped with explicit SIGTERM; no children or other-user
  processes were touched.  `stage1.done`, `frozen_correspondence_baseline.json`
  and `frozen_correspondence_queries.json` retained identical mtime, size and
  SHA-256 before/after shutdown.  No Phase20 process remains; see
  `outputs/iclr27_phase20/audit/stage1_duplicate_process_shutdown.json`.

## Phase 21 proposal/observability repair (2026-08-28)

- Created independent `docs/src/scripts/configs/outputs/iclr27_phase21`
  namespace.  Phase20's 76 positive pseudo-held events (plus 76 negatives for
  context), fixed folds, row keys, prefixes, reliability rule, and 25/76
  prefix16 ceiling were frozen.  Only public TRAIN category/video metadata
  was read; DEV+, Q1, and public new-model labels stayed sealed.  The source
  CSV and DINOv2 cache were reused through existing symlinks; `/data1` had
  about 50G free, so no large copy was made.
- Stage0 geometry/chronology audit used actual per-row image dimensions and
  recomputed xyxy IoU.  Across 43,423 rows there were zero invalid boxes,
  zero normalized-coordinate mismatches, zero stored-IoU mismatches, zero
  duplicate row keys, and zero non-monotone track chronologies.  The Phase20
  curve reproduced exactly (17/22/22/23/25 at prefixes 1/2/4/8/16).
- Stage1 fixed variants retained every event and row.  At prefix16,
  `causal_smoothed` reached 22/76 (source/target 45/37), fixed 10% expansion
  21/76 (44/38), and history/ROI-history/quality-rerank stayed 25/76
  (49/40).  No non-training variant improved both sides or reached 0.50;
  `GT-tight` was kept as a diagnostic only and reached 73/76.  Full
  variant/prefix/fold values and all failed event keys are in
  `outputs/iclr27_phase21/audit/stage1_*_events.json` and
  `metrics/stage1_proposal_variants.json`.
- Because Stage1 had no genuine proposal improvement, Stage2 class-agnostic
  refinement training, correspondence training, modern-backbone download,
  controller reconnect, final training, and public evaluation were not
  authorized.  Terminal decision:
  `P21_GATE_O_FAIL_PROPOSAL_OBSERVABILITY_STOP`.
- Resource preflight: ten idle A100-SXM4-40GB GPUs, 120G available RAM,
  swap disabled; diagnostics were CPU-only and produced no OOM, near-OOM,
  process termination, or other-user impact.  No Phase21 training process
  remains.  Next candidate is proposal-domain/ROI observation repair or task
  redefinition, not another threshold/memory/controller lottery.

## Phase 22 proposal source/detector repair (2026-08-28)

- Created the independent `docs/src/scripts/configs/outputs/iclr27_phase22`
  namespace.  Public TRAIN rows/GT/category/video metadata were the only
  supervision.  DEV+, Q1, public new-model labels, future frames, physical
  IDs and semantic text stayed sealed; prior evaluators/checkpoints were
  read-only comparators.
- Stage0 re-read the complete Phase21 76-event audit and retained every event
  and denominator.  Prefix16 reproduced the true-IoU perfect-correspondence
  ceiling **25/76** with a clean geometry/chronology audit.  The 51 failures
  were assigned-box IoU failures: target-only 24, source-only 15, both sides
  12.  No proposal-missing, wrong-frame/rank, assignment-only or coordinate
  error class was observed.  Cross-video, small-object and stability flags are
  explicitly correlational diagnostics, not causal claims; occlusion labels
  are unavailable.
- Stage1 confirmed the frozen Phase6B DSCT/OVTR source and built four fixed
  video/category-disjoint TRAIN folds (43,423 rows, 9,741 GT rows, 370
  videos, 153 categories).  Existing Phase21 non-training variants remained
  at or below 25/76, so one class-agnostic proposal refiner was registered.
- The refiner consumed frozen DINOv2 CLS+ROI and causal score/geometry/age/
  stability only; it predicted a bounded normalized box delta and quality
  logit with SmoothL1+BCE.  Four folds ran one worker per GPU (0--3), BF16,
  batch 256, 2,000 updates, checkpoints every 500, bounded supervisor and
  atomic `.launched`/`.done` markers.  Preflight retained about 115--121 GiB
  available RAM; no OOM, swap, near-OOM, duplicate launch or other-user
  termination occurred.
- The initial refiner moved usable boxes away from identity and reached only
  3/76 on the full true-IoU event evaluator.  One minimal repair zero-
  initialized the residual box-delta head (same split, loss, optimizer,
  steps, evaluator and no new architecture).  The repair also reached 3/76;
  source/target reliable event coverage fell to 10/7 versus raw 49/40.
  Fold ceilings were 0/12, 2/12, 1/24 and 0/28.  Training loss and row-level
  validation were not treated as task success.
- Gate P therefore **FAILS** with decision
  `P22_GATE_P_FAIL_STOP_BEFORE_CORRESPONDENCE` (required >=38/76, both-side
  and broad fold/category/video improvement).  No correspondence,
  controller/StateMemory/threshold tuning, modern-backbone download, final
  50k or public evaluation was started.  Next direction is a proposal
  source/detector or candidate-generation/objectness recall study, followed
  by correspondence only after a separately verified proposal ceiling.

## Phase 23 proposal source replacement/candidate generation (2026-08-28)

- Phase23 uses only its independent namespace and the frozen 76 positive
  pseudo-held event protocol.  DEV+, Q1, public new-model labels, future
  frames/tracks, physical IDs and semantic text remain sealed; Phase20--22
  artifacts are read-only.
- Stage0 found the first actionable Phase22 failure: the corrected CSV and
  frozen DINOv2 NPZ have the same five-field key set but **0/43,423 positional
  matches**.  The CSV row-0 key is `1231:0:0:3661:47589`, while the NPZ row-0
  key is `0:0:0:0:0`; all 76 event row sets and all four folds are affected.
  The old Phase22 loaders built a key map but indexed features positionally,
  pairing every proposal with another row's visual feature.  Geometry and
  chronology checks remain 0/0/0.  Phase23 records both source hashes and an
  in-memory permutation SHA256; no Phase22 file is modified.
- A two-step aligned-feature smoke and fold-0 targeted regression completed.
  Raw baseline remains exactly **25/76** (folds 8/12, 2/12, 10/24, 5/28),
  identity residual preserves that ceiling, and the frozen repair checkpoint
  can be evaluated with aligned features without invalid boxes.  The legacy
  positional and corrected-input row counts are retained in
  `outputs/iclr27_phase23/audit/alignment_smoke_targeted_regression.json`.
- Stage1 fixed candidate-pool oracle retained every raw row and added the
  preregistered 27 scale/center transforms for the current and up to four
  causal history boxes.  Prefix16 pool ceiling is **38/76** (raw 25/76), with
  source/target reliable event coverage 60/49 and four-fold pool ceilings
  8/12, 5/12, 15/24, 10/28.  TRAIN-only candidate recall at IoU .5 is
  0.4136 versus raw 0.2578.  This authorizes exactly the registered Stage2A
  quality/objectness ranker; no proposal source replacement, correspondence,
  controller or backbone branch is assumed yet.
- A Phase23 implementation audit found and fixed a metadata-ordering bug in
  the first candidate/ranker pass: boxes were concatenated transform-major
  while parent/assignment metadata was initially history-major.  That pass is
  retained but marked superseded; it is not used for any decision.  The fixed
  implementation uses transform-major boxes and matching parent/assignment
  order, then reruns Stage1 and the ranker smoke before the final four-fold
  training.  This is separate from the original Phase22 CSV/NPZ key-order
  bug; neither prior-phase artifact is modified.
- The first Stage3 diagnostic also briefly used integer `parent_idx` as the
  assigned mask, producing a superseded 23/76 oracle.  The evaluator was
  corrected to use the explicit boolean `parent_assigned` field and rerun;
  the valid fixed-pool oracle is **38/76**.  Both implementation incidents
  are retained in the Phase23 provenance and did not alter the 76-event
  denominator.
- Ordered Stage2A ranker smoke and four-fold 4,000-update training completed
  on GPUs 0--3 (BF16, batch 256, checkpoints every 500; GPU4's unrelated PID
  was untouched).  Validation top-5 candidate recall was 0.219/0.172/0.270/
  0.276 for folds 0--3.  On the frozen true-IoU evaluator the ranker top-5
  reaches only **21/76** (folds 4/12, 1/12, 11/24, 5/28), below raw 25/76;
  source/target event coverage is 44/36 versus raw 49/40.  Thus the fixed
  candidate pool has diagnostic coverage but the learned quality interface
  did not generalize to reliably select it.
- Phase23 Gate P2 is **FAIL** (`P23_GATE_P2_FAIL_STOP_BEFORE_CORRESPONDENCE`):
  the real ranker is below 38/76 and improves only one fold.  No source
  branch (not authorized after the >=38 pool condition), correspondence,
  controller/StateMemory/threshold tuning, modern backbone, final training or
  public evaluation was started.  The original Phase22 result is invalid as
  a visual-model conclusion because of its row-key bug; after the corrected
  Phase23 loader and candidate protocol, proposal selection remains the
  limiting result.

## Phase 24 proposal selection/source generalization (2026-08-28)

- Created the isolated `docs/src/scripts/configs/outputs/iclr27_phase24`
  namespace. Phase20--23 evaluators, row keys, 76-event denominator and DSCT
  stream were read-only; only public TRAIN rows/GT/category/video metadata
  were used. DEV+, Q1, public new-model labels, future frames, physical or
  semantic IDs and text stayed sealed.
- Stage0 reproduced corrected Phase23 alignment (43,423/43,423 key-set
  overlap, 0 positional matches, in-memory permutation SHA256
  `269b739a...fa9fa885a29`), raw prefix16 **25/76**, and fixed 27-transform
  causal candidate-pool oracle **38/76**. Raw prefix curves were 17/22/22/23/25
  and pool curves 31/33/34/36/38. Prefix16 taxonomy counts were pool-both-
  reliable 21, pool-reliable-but-MLP-missed 17, pool target missing/IoU<0.5
  22, pool source missing/IoU<0.5 11, and pool-both-unreliable 5; all events
  and hard denominator rows remained present.
- Stage1 unified diagnostics at prefix16 were fixed score/history 14/76;
  Phase23 MLP top1/top5/top10/top20 11/21/22/28; uncertainty/defer 28; pool
  oracle 38 (diagnostic only). Increasing K retained more candidates but did
  not recover the oracle, authorizing only the registered set-aware branch.
- The first set-aware smoke exposed `torch.flatnonzero`; the minimal
  `torch.nonzero` repair then exposed integer/string metric keys. A second
  minimal key-type repair passed the same smoke and a 10-step fold0 targeted
  regression; no data, seed or protocol changed.
- A first full launch (supervisor 20669, workers 20672--20675) was stopped
  with explicit per-PID SIGTERM when physical mapping was not auditable; no
  full checkpoint/marker existed and no external process was touched. The
  repaired supervisor used `env CUDA_VISIBLE_DEVICES=4,5,6,7` and an
  expected-physical-GPU assertion. Mapping smoke/targeted markers confirmed
  GPU4; the resumed four-fold run completed 4,000 updates/fold on GPUs4--7,
  BF16, batch32, checkpoints every500, with no OOM/swap/near-OOM or residual
  worker.
- Frozen set-aware top20 reached true prefix16 **32/76**, source/target
  reliable 56/46 versus raw 49/40, category/video 11/24, and fold ceilings
  7/1/15/9 versus raw 8/2/10/5 (only two folds improved). Gate P2 is
  **PARTIAL**, decision `P24_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE`.
- Because the real model is 30--37/76, no correspondence encoder,
  controller/StateMemory/threshold change, modern backbone, final 50k or
  public/Q1 evaluation was launched. Candidate-pool 38/76 and GT-tight 73/76
  remain diagnostics only. Final report:
  `docs/iclr27_phase24/PHASE24_PROPOSAL_SELECTION_SOURCE_GENERALIZATION_COMPLETE_REPORT.md`.

## Phase 25 MOT-preserving proposal set/source generalization (2026-08-29)

- Independent Phase25 namespace; public TRAIN only, DEV+/Q1/public new-model labels sealed. Phase24 regression is exact: raw 25/76, candidate-pool oracle 38/76 (diagnostic), set-aware top20 32/76.
- Stage0 failure taxonomy for 44 set-aware failures: source-side pool 11, target-side pool 22, no reliable pool on either side 5, candidate exists but source/target not retained 3/3. Stage1 confidence-calibrated top20 34/76 and history-consistent top20 33/76.
- New one-block two-head attention selector (key-aligned DINOv2 CLS/ROI + causal geometry, no IDs/GT/future) trained four video/category-disjoint folds, 4,000 updates, BF16, GPUs 4--7, no OOM. Stage3 best attention top27 is 30/76, source/target 52/47, folds 4/1/15/10; only 2/4 folds improve.
- Gate decision: **P25_GATE_P2_PARTIAL_STOP_BEFORE_CORRESPONDENCE**. No correspondence/controller/threshold/backbone/public evaluation. First smoke retained a marker because box_y2_norm was omitted (21 vs 22 dims); minimal repair rebuilt manifests, smoke and targeted regression passed. Stage1 had three code-only repairs; all final artifacts are atomic.
- Report: /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/docs/iclr27_phase25/PHASE25_MOT_PRESERVING_PROPOSAL_GENERALIZATION_COMPLETE_REPORT.md

## Phase 26 proposal source/candidate coverage (2026-08-29)

- Frozen Phase24/25 comparators: raw 25/76, fixed pool oracle 38/76, Phase24 set-aware 32/76, Phase25 attention 30/76.
- Stage0 source gaps: target 22, source 11, both 5; six pool candidates were missed by Phase25 selection. Stage1 broad causal pool oracle reached 56/76 (diagnostic).
- One class-agnostic source head (eight causal candidates, TRAIN-only GT supervision, BF16, four video/category-disjoint folds on GPUs4-7) reached real 41/76, source/target 67/48, folds [11, 5, 14, 11]; Gate **P26_GATE_P2_PASS_AUTHORIZE_CORRESPONDENCE**. No OOM, no sealed-data access.
- Duplicate Stage0 PIDs were audited; only PID25464 plus diagnostics were terminated, PID26424 exited naturally; hashes and atomic rewrite evidence are recorded.
- Proposal report: /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/docs/iclr27_phase26/PHASE26_PROPOSAL_SOURCE_CANDIDATE_COVERAGE_COMPLETE_REPORT.md

## Phase 27 correspondence and Phase 28 frozen compatibility (2026-08-29)

- Phase27 was an isolated, single-route causal GRU correspondence encoder over
  key-aligned DINOv2 CLS/ROI features.  The four video/category-disjoint folds
  completed 2,000 updates on GPUs 4--7 with BF16 and resumable checkpoints.
  Three bounded performance repairs were required (original validation
  hotspot, matrix similarity, then set-membership batching); task-owned PIDs
  2976, 7656 and 9871 were explicitly SIGTERM'ed, while retrieval benchmark
  PID13931 was run only diagnostically and then explicitly ended.  No OOM or
  external-process termination occurred.  The final Gate R result is
  **FAIL**: substantial/directional improvement was 0/4; aggregate R@1 fell
  0.8032 -> 0.7027 and mAP 0.7201 -> 0.6246.  The Phase27 report and decision
  are retained at
  `docs/iclr27_phase27/PHASE27_CORRESPONDENCE_CONTROLLER_COMPLETE_REPORT.md`
  and `outputs/iclr27_phase27/audit/phase27_decision.json`; controller,
  public/Q1 and modern-backbone branches were not started.
- Phase28 registered a no-training frozen-representation compatibility
  diagnostic.  Phase26 source proposal (41/76) and the unchanged Phase19R
  RC-MS-OCD controller/StateMemory/threshold/action semantics were symlinked
  read-only; original normalized fused DINOv2 CLS/ROI was evaluated on all 76
  positive events plus registered negatives and prefix diagnostics 1/2/4/8/16.
  The first run exposed a manifest field-name mismatch in compact diagnostics;
  the minimal fix used the evaluator-normalized `target_category`, then the
  same causal evaluation completed without training or threshold sweep.
  Frozen persistent Commit-CT is **3/76** versus the historical mixed **2/76**,
  but all three correct events are fold3/category81/source video575 (targets
  1814/1955); folds0--2 are zero.  Fold3 false merge worsens 0.3929 -> 0.4286
  and new recall 0.2857 -> 0.2500; per-fold safety/broad coverage therefore
  fails despite aggregate false-merge 0.2842, duplicate births 84 and known
  metrics not worsening.  Positive action outcomes are EXISTING 27 (only 3
  correct), NEW 20 and NONE 29; these are proxies, not an invented exact
  known/novel confusion matrix.  Gate C is
  **P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION**.  No public/Q1 labels,
  future information, training, calibration, threshold/memory/controller
  tuning or backbone download was used.  No residual Phase28 process remains;
  hashes, MOT invariants and integrity are recorded in the Phase28 audit.
- Phase28 report:
  `docs/iclr27_phase28/PHASE28_FROZEN_REPRESENTATION_COMPATIBILITY_COMPLETE_REPORT.md`.
  The result is a narrow fold/category/video compatibility signal, not sealed
  MOT+OCD success.  Next work must register one evidence-based
  cross-fold/domain representation route with Phase26 proposal and physical
  MOT frozen, and must beat the frozen baseline on disjoint validation and
  broad persistent safety before any modern backbone or controller work.

## Phase 29 cross-fold/domain representation alignment (2026-08-29)

- Registered one isolated `DomainAlignedResidualEncoder` route after the
  Phase28 Gate C failure.  Phase26 source proposal (41/76), physical MOT
  stream, Phase19R controller/StateMemory/thresholds/action semantics and the
  original 76-event causal protocol remained frozen.  The adapter consumed
  only causal key-aligned DINOv2 CLS/ROI mean/last/absolute-delta statistics;
  it had a zero-initialized residual, multi-positive cross-video InfoNCE,
  raw-DINO hard-negative ranking, prefix consistency and residual-norm penalty.
  No GRU, modern backbone, controller or sealed label was used.
- The first fold0 smoke (`domain_smoke_smoke_f0`) and its debug reproduction
  (`domain_smoke_debug_smoke_f0`) failed before update 1 due to a missing
  singleton broadcast in `[B,D] * [B,2,D]`; both launched markers and initial
  checkpoints are preserved as superseded/debug evidence with no `.done` or
  metrics.  The minimal `ea[:,None,:] * pos` repair passed `domain_smoke_fix1`
  and `domain_smoke_fix2`, then fold0 targeted runs.  A second code-only
  vectorized retrieval-mask repair preserved ranking semantics and removed the
  validation hotspot.  Formal four-fold training completed 2,000 updates/fold
  on GPUs4--7, BF16, bounded one-worker-per-fold, with no OOM/swap or external
  process termination.
- Disjoint validation at prefix16: fold0 identity 0.9624/0.9564, fold1
  0.9175/0.8709 -> 0.9588/0.8498, fold2 0.6866/0.5847 -> 0.7164/0.5461,
  fold3 identity 0.6462/0.4682 (R@1/mAP).  Mean R@1 improved 0.8032 ->
  0.8209, but mean mAP fell 0.7201 -> 0.7051; no fold improved both by the
  preregistered +0.02/+0.01 threshold.  Gate R is
  **P29_GATE_R_FAIL_STOP_BEFORE_CONTROLLER** (0/4 substantial, 0/4
  directional), so no persistent controller compatibility score is claimed.
  All 76 event cosine records, fold/category/video split and checkpoint hashes
  are retained for diagnosis only; Phase28's controller result remains 3/76
  narrow fold3/category81/source-video575 evidence.
- Phase29 report:
  `docs/iclr27_phase29/PHASE29_CROSS_FOLD_DOMAIN_ALIGNMENT_COMPLETE_REPORT.md`.
  Public/Q1/DEV+ labels remain sealed and the full MOT+OCD objective is not
  achieved.  Do not tune the controller, GRU, StateMemory or thresholds or
  download a backbone from this result; any next representation route must be
  independently justified and beat frozen DINOv2 broadly before one controller
  compatibility run is authorized.

## Phase24–29 integrated stopping decision (2026-08-29)

- Consolidated report written to
  `docs/iclr27_phase29/TRACKOCD_PHASE24_29_FINAL_MOT_OCD_REPORT.md` with
  machine-readable decision `outputs/iclr27_phase29/audit/final_integrated_decision.json`.
- Fixed-denominator proposal evidence is raw **25/76**, Phase24 set-aware
  **32/76**, Phase25 attention **30/76**, Phase26 learned source **41/76**;
  fixed/broad candidate-pool diagnostics are **38/76**/**56/76** (GT-tight
  diagnostic 73/76).  Phase26 is the only proposal Gate P2 PASS; MOT
  structural invariants remain continuity 1.000, duplicate tracks 0,
  fragmentation delta 0 and parent mismatch 0/26946.
- Both registered representation routes fail Gate R: Phase27 GRU mean R@1/mAP
  0.8032/0.7201 -> 0.7027/0.6246 (0/4 substantial), and Phase29 residual
  domain alignment 0.8032/0.7201 -> 0.8209/0.7051 (0/4 joint/directional).
  Phase28 frozen DINO/controller compatibility is 3/76 versus historical 2/76,
  but all three events are fold3/category81/source-video575 and fold3 false
  merge/new-recall safety regresses; Gate C FAIL.
- All Phase24–29 preregistered candidates are complete.  Final decision is
  `TRACKOCD_PHASE24_29_FINAL_STOP_CURRENT_PROTOCOL_INFEASIBLE`; full MOT+OCD
  success is not claimed.  No public/Q1/DEV+ labels, modern backbone,
  threshold/StateMemory/controller tuning or final-50k run was performed.
  Resource/process incidents, failed smoke evidence, hashes and symlink ledger
  are retained in the integrated audit; no OOM or external-process kill
  occurred and no Phase24–29 process remains.

## Phase30 interface redesign (2026-08-29)

- Read and froze `docs/iclr27_phase30/TRACKOCD_PHASE30_CROSS_INSTANCE_INTERFACE_REDESIGN_PROMPT.md`.
  Stage0 created an independent support/query episode contract from public
  TRAIN GT metadata only: 5,545 episodes across four video/category-disjoint
  folds, with exact held-event track/key exclusion and no category/video/ID/text
  values in model-facing fields.  Feature/CSV alignment remains 43,423 exact
  rows with the inherited in-memory permutation SHA256
  `269b739ab52e5c9b24b541c75de6039d7d721ca166f03f31f9901da9fa885a29`.
- The first Stage1 frozen retrieval implementation (task PID 18770) produced
  no atomic metrics after >2 minutes because of an avoidable Python candidate
  scan/repeated embedding hotspot.  It was explicitly SIGTERM'ed as a
  task-owned process; Stage0 artifacts were preserved.  The minimal repair
  vectorized candidate masks/support indexing and set Torch threads to one;
  no protocol, seed, denominator or data boundary changed.  The repaired
  Stage1 rerun completed and authorized the single preregistered Stage2 route.
## Phase30 Stage1/Stage2 completion (2026-08-29)
- Stage1 frozen diagnostics completed after one explicitly terminated task-owned PID 18770 (unoptimized candidate scan); vectorized support indexing and `torch.set_num_threads(1)` repaired the path. Aggregate frozen DINOv2 p16 R@1/mAP = 0.8932/0.8484; non-trained support-set max-over-support diagnostic mAP = 0.9415. No held/public/Q1 data was read.
- Stage2 smoke (`interface_smoke_smoke_f0`, 100 updates) and fold0 targeted (`interface_targeted_f0`, 500 updates) completed atomically with checkpoints, metrics and `.done` markers. Formal four-fold training (2000 updates/fold, BF16, GPUs 4,5,6,7) completed with best validation scores f0=0.7627, f1=0.9012, f2=0.9418, f3=0.9395; all checkpoints/metrics/done markers present.
- An accidental duplicate diagnostic supervisor tag `testtag` was started by a shell inspection command. Confirmed task-owned PIDs 31658/31664-31667 and sent SIGTERM only to those PIDs; preserved `.launched` markers as incomplete evidence, no external process touched.
- Frozen interface retrieval (`interface_retrieval.json`) at p16: learned track embeddings mean R@1=0.7472, mAP=0.7675, hard-negative gap=-0.0002, versus raw DINOv2 0.8932/0.8484/0.1896. Support-set scoring reaches R@1=1.0 on TRAIN validation episodes but is a diagnostic episode contract effect, not persistent OCD evidence. Gate R30 therefore FAILS (no broad multi-fold improvement); controller compatibility is not authorized.

## Phase31 raw-space reranker (2026-08-29)
- Stage0 reconciled evaluator contracts: Phase27/29 0.8032/0.7201 uses different filtered tracklet denominators than the Phase30/31 TRAIN-disjoint episode contract (837/82/39/30 validation tracklets); exact 43,423-row feature alignment and permutation hash are unchanged.
- Fixed temporal-stability diagnostic was a small +0.0025 mAP margin. The sole registered monotonic raw-cosine reranker passed smoke/targeted and four-fold BF16 training on GPUs 4–7. Frozen validation p16 mean R@1/mAP improved 0.8932/0.8484 -> 0.9463/0.9149; folds 1–3 met both preregistered margins, hard gap and coverage non-worse. Gate R31 PASS.
- Unchanged-controller compatibility was run only as a raw comparator (1/76 CT, safety metrics retained). Reranker pair scores have no legal hook in frozen RC-MS-OCD without changing action semantics, so Gate C31 is NOT RUN and no OCD gain is claimed. No public/Q1/DEV+ access, threshold/memory/controller/backbone tuning or protocol change occurred. Decision: `P31_GATE_R_PASS_C31_NOT_RUN_INTERFACE_CONSTRAINT`.

## Phase32 interface-compatible routing (2026-08-29)
- Stage0 contract audit PASS: routing restricted to candidate order; raw vectors, state bundle, physical MOT rows, known masks, thresholds and action semantics remain unchanged. Stage1 reproduced frozen ceilings raw 25/76, fixed pool 38/76, Phase26 source 41/76 and broad oracle 56/76; no oracle was treated as OCD success.
- Stage2 unchanged-controller compatibility yielded routed Commit-CT 1/76 (category/video coverage 1/1, duplicate births 87), not strictly above frozen Phase28 3/76. Because the physical stream has one selected raw vector per row, routing leaves the causal sequence unchanged. Gate C32 FAIL; no public/Q1 access, controller/threshold/memory/backbone tuning or protocol change. Luna remains open for separately authorized evidence-driven work.

## Phase33 query-conditioned feature interface (2026-08-29)
- Stage0 interface/causal audit PASS: adapter output is exactly the frozen 768-D controller input (`normalize(raw + alpha*delta)`), alpha initialized at zero, support summary causal/permutation-invariant, no IDs/text/future/held GT.
- Initial smoke failed only because the independent checkpoint directory was absent; minimal directory creation fixed it. New-tag smoke (100) and fold0 targeted (500) passed; four-fold 2k BF16 training completed on GPUs 4–7 with atomic checkpoints/markers.
- Frozen adapter retrieval p16 remained essentially identity: R@1 0.8932→0.8932, mAP 0.8484→0.8488, hard gap 0.1896→0.1900, 0/4 folds meeting registered gain. Gate R33 FAIL; Gate C33 NOT RUN. No public/Q1/DEV+ access or controller/protocol modification; final MOT+OCD remains unproven.

## Phase34 reranker-weighted prototype bridge (2026-08-29)
- Stage0 PASS: fixed K=5, temperature=0.1, beta=0.25 bridge preserves 768-D controller input and exact no-support raw identity; causal/sealed audits pass.
- Stage1 TRAIN retrieval improved p16 to R@1 0.9638/mAP 0.9482 (raw 0.8932/0.8484), but held-event tracks have no legal TRAIN support by construction. Stage2 unchanged-controller compatibility therefore falls back to raw vectors and yields Commit-CT 1/76, category/video coverage 1/1, duplicate births 87. Gate C34 FAIL; no controller/threshold/memory/backbone/public-label changes.

## Phase35 causal history interface (2026-08-29)
- Stage0 causal contract PASS for fixed K=4 current/previous-row history bridge; no future/support/ID/text/GT input and old controller interface unchanged.
- Stage1 history retrieval degraded p16 R@1/mAP to 0.8256/0.8169 (raw 0.8932/0.8484). Stage2 unchanged-controller replay produced Commit-CT 0/76, coverage 0/0, duplicate births 328; Gate C35 FAIL. No public/Q1/DEV+ access or controller/threshold/memory/backbone changes. Luna remains open.

## Phase36 reliability-gated causal fusion (2026-08-29)
- Stage0 input/scale/causal contract PASS; Phase35 failure reproduced. Fixed preregistered gate (`K=4`, alpha=0.25*sigmoid(4*(stability-0.7)), exact raw fallback) was evaluated without training.
- Stage1 reliability fusion degraded p16 retrieval to R@1/mAP 0.8045/0.7970 and hard gap 0.1373 versus raw 0.8932/0.8484/0.1896. Stage2 gate fitting and controller compatibility were not authorized. No public/Q1/DEV+ access or protocol/controller changes; Phase36 stops at Stage1 while Luna remains open.

## Phase37 observability/task-interface audit (2026-08-29)
- Complete 76-event audit confirms TRAIN-derived support availability is 0/76 at every prefix; prefix16 has 25 current-observable, 39 causal-history-only and 12 no-legal-support events. Raw/Phase26 ceilings remain 25/76 and 41/76.
- Universal absence of legal cross-video support satisfies the preregistered TASK-INTERFACE BLOCKED criterion. No Stage2 encoder, controller, threshold, memory or backbone experiment is authorized; only a future explicit support-definition authorization could unblock the interface. Public/Q1/DEV+ remain sealed and Luna stays open.

## Phase38 causal support-stream redesign (2026-08-29)
- Authorized prior-video support stream audited: `PRIOR_COMPLETED_TRACK` provides causal same-category diagnostic support for 76/76 held events (fold upper bounds 12/12/24/28; category coverage 19, video coverage 49) without future/ID/text/held-label leakage. Raw/Phase26 ceilings remain 25/41.
- Single support-conditioned correspondence route completed smoke, targeted and four-fold 2k BF16 training on GPUs 4–7. Frozen retrieval p16 mean R@1/mAP=0.7753/0.7733 vs raw 0.8932/0.8484, so Gate R38 FAIL and controller compatibility not run. No public/Q1/DEV+ access or controller/protocol changes; Luna remains open.

## Phase39 evaluator/model contract repair (2026-08-29)
- Stage0 confirmed first root cause: Phase38 evaluator ignored support-conditioned `forward/pair_scores` and used independent embedding dot products. Phase38 checkpoints/manifests remained frozen.
- Corrected forward replay still yielded p16 R@1/mAP 0.6893/0.8382 vs raw 0.8932/0.8484; loss already included positive pair supervision and hard negatives, so no loss repair was authorized. Gate R39 FAIL, C39 not run; no public/Q1/DEV+ or protocol/controller changes.

## Phase40 raw-preserving support score (2026-08-29)

- Stage0 confirmed the Phase39 evaluator contract and froze Phase38 checkpoints. The residual score is `raw cosine + bounded beta*delta`, zero-initialized with exact raw fallback; no category/ID/text/future/held-GT input.
- Initial smoke failed only because the independent Phase40 checkpoints directory was absent; minimal directory creation fixed it. New-tag 100-step smoke, fold0 500-step targeted, and BF16/AMP four 2k-fold runs completed on GPUs 4–7 with atomic markers/checkpoints.
- Global p16 retrieval improved raw 0.8932/0.8484 to residual 0.9506/0.8981 (R@1/mAP), but hard-negative gap regressed in folds 0 and 1 (0.2076→0.1642; 0.3520→0.3181). Gate R40 therefore FAILS the all-fold non-degradation rule; Gate C40 was not run because score-level residual has no legal row-vector hook into the frozen controller.
- A slow per-query evaluator was interrupted once (task-owned only) and replaced by a vectorized GPU evaluator; no OOM or external-process event. Public/Q1/DEV+ remain sealed, no proposal/controller/threshold/memory/backbone change occurred, and Luna remains open.

## Phase41 safety-constrained vector bridge (2026-08-29)

- Stage0 froze Phase40 residual checkpoints and verified the legal 768-D bridge contract. Stage1 fixed an alpha initialization bug (sigmoid(0) was 0.5); the corrected alpha=0 path is exactly raw.
- First gate smoke failed only from a missing Phase41 checkpoint directory. After minimal directory creation, fix2 smoke/targeted and four 2k BF16/AMP fold runs on GPUs 4–7 completed atomically. A zero-gradient `relu(tanh)` gate was then repaired once to differentiable bounded `clamp(alpha_max*sigmoid(g)*g,0,alpha_max)`; failed artifacts remain preserved.
- Trained bridge p16 aggregate R@1/mAP/hard-gap = 0.9184/0.8830/0.2047 versus matched raw 0.9000/0.8599/0.1944, but only fold3 met both +0.02/+0.01 margins and fold0 hard gap regressed. Gate R41 FAIL; C41 not run because the registered broad safety condition was not met. No controller/threshold/memory/proposal/backbone/public-label change; Luna remains open.

## Phase42 selective hard-negative gate (2026-08-29)

- Stage0 per-query flip audit over frozen Phase41 bridge found 0 unsafe flips under the fixed TRAIN-only selective rule (`support_quality>=0.2`, bridge margin >= raw+0.005); matched episode p16 improved 0.6099/0.8022/0.0485 to 0.7054/0.8424/0.0891, with three-fold margin potential.
- Unique logistic selective gate: first smoke/targeted failed only from bridge module CPU/CUDA placement; `.launched` evidence preserved, `.to(device)` was the minimal fix. Four 1k-step BF16/AMP fold runs completed on GPUs 4–7.
- Trained gate bridge-use rate was 0.0 in all folds, collapsing to raw with no R@1/mAP gain. Gate R42 FAIL; C42 not run. No proposal/controller/threshold/memory/backbone/public-label changes, no OOM or external process event, Luna remains open.

## Phase43 TRAIN-only policy-distilled selective gate (2026-08-29)

- Stage0 code audit confirmed Phase42 target/loss mismatch: fit max-margin teacher positives were 0.968–0.998 while all-positive-mean ranking positives were 0.026–0.129 at p16; fixed teacher-use rates were 0.731–0.839.
- Replaced the conflicting objective with BCE to the fixed TRAIN teacher plus raw-preserving unsafe-flip penalty. Smoke/targeted and four 1k BF16/AMP folds completed on GPUs 4–7 with atomic checkpoints.
- Learned gate improved matched p16 aggregate raw→learned R@1 0.6099→0.7331, mAP 0.8022→0.8539 and hard gap 0.0485→0.0989; teacher agreement 0.784–0.976. However bridge_use_rate was 1.0 in every fold, violating the registered non-unconditional selective criterion. Gate R43 FAIL; C43 not run. No public/Q1/DEV+, controller, threshold, memory, proposal or backbone change; Luna remains open.

## Phase44 calibrated conditional gate (2026-08-29)

- Stage0 reproduced Phase43 constant bridge: p16 predicted p<0.5 rate 0 in all folds, p means 0.773–0.863, matching teacher-majority rates 0.731–0.839. Fixed teacher condition and p>=0.5 inference threshold remained unchanged.
- Balanced BCE conditional gate training replaced constant-bias objective. Initial smoke hit an unsupported `pos_weight` argument to probability BCE; explicit weighted BCE was the minimal repair, followed by smoke/targeted and four 1k BF16/AMP folds on GPUs 4–7.
- Learned p16 matched replay improved raw→learned R@1 0.6099→0.7167, mAP 0.8022→0.8475 and hard gap 0.0485→0.0888, with 0 unsafe flips and teacher agreement 0.823–0.976. Fold1 bridge_use_rate remained 1.0, violating non-unconditional Gate R44; Gate R44 FAIL and C44 not run. No public/Q1/DEV+, controller, threshold, memory, proposal or backbone changes; Luna remains open.

## Phase46 causal support audit and final conditional gate (2026-08-29)

- Read-only Stage0 verified the redefined prior-completed-track support stream is causal for all 76 events and prefixes: strict prior-video ordering, 0 future rows, nonempty support 76/76, and no leakage. Frozen Phase41 bridge contract is valid 768-D with exact raw fallback for invalid support. Persistent controller was not run.
- Full TRAIN materialization (all fit records × prefixes) with a single balanced BCE-with-logits conditional gate and explicit support/margin penalties completed smoke, fold0 targeted and four 1k-step folds on GPUs 4–7. p16 aggregate raw→learned R@1/mAP/hard-gap = 0.6099/0.8022/0.0485 → 0.7111/0.8431/0.0893; all four folds improve, bridge use is 0.7297–0.9756 (non-unconditional), teacher agreement ≥0.894 and unsafe flips 0.
- Gate R46 PASS is retrieval/safety only; no Commit-CT or controller compatibility is claimed or run. A task-owned one-step `test` supervisor was accidentally started during process checking; all four tiny diagnostic units completed and are retained but excluded from selection. No OOM/external kill/public or sealed access; no old-stage modifications.
- Decision `P46_GATE_R_PASS_RETRIEVAL_CONTROLLER_PROHIBITED`; next work requires explicit unchanged-controller compatibility on the frozen gate. Luna remains open.

## Phase45 gate-loop exit and MOT+OCD readiness audit (2026-08-29)

- Read-only audit of Phase42–44 artifacts. The Phase42 registration already required non-unconditional bridge use per fold; this criterion was not introduced retroactively. Phase44 p16 fold1 uses bridge for 100%, while its TRAIN teacher-negative rate is 18.74% (teacher-use .8126), so all-bridge is evidence of conditional-gate calibration/constantization, not absent negatives. Folds 0/2/3 are non-constant and all folds have zero unsafe flips.
- Frozen contract smoke produced finite 768-D row vector and exact raw fallback for invalid support. Phase38's 76/76 prior-support figure is retrieval diagnostic availability, not persistent online Commit-CT support; Phase45 did not run controller or Commit-CT.
- Initial system-Python audit failed because torch was unavailable; rerunning the same script with the existing `/home/lwr/anaconda3/envs/locatemot/bin/python` completed. No long process, OOM, duplicate worker, sealed-data access, or old-stage modification occurred.
- Decision A: `P45_GATE_LOOP_AUDIT_CONSTANTIZATION_REPAIR_CANDIDATE`. Proposed (not executed) next minimal route is TRAIN-only conditional calibration with explicit negative coverage and unchanged p>=0.5/raw fallback. Luna remains open; no new gate training is authorized by Phase45.

## Phase46 C1/C2 and Phase47 (2026-08-29)
- Phase46 C0/C1 contract checks passed: prior-completed-track support is causal for 76/76 diagnostic events, the frozen Phase41 bridge emits finite normalized 768-D vectors with exact raw fallback for invalid support, and unchanged physical MOT/state ordering was preserved. No sealed labels were read.
- Phase46 C2 unchanged Phase19R controller replay was completed on all 76 positive events plus negatives. Learned gate produced 3/76 Commit-CT (fold3 only; category coverage 1, video coverage 2) versus 1/76 in this raw replay (historical mixed comparator remains 2/76). False-merge mean 0.3051, duplicate births 84, premature 0.2664, unresolved 0.4449; fold3 gain was narrow and known/novel safety/coverage were not broad. Gate C46 FAIL; no sealed evaluation.
- C2 first fold exposed a missing Phase41 checkpoint f-string path. The minimal path-format fix was smoke/regression tested; no protocol or evaluator change. All C2 workers used GPUs 4–7, RAM headroom stayed above the 25% floor, no OOM or external process was terminated.
- Phase47 was the single authorized class-agnostic correspondence/interface repair after C2 failure. It used frozen Phase46 causal vectors and the Phase27 disjoint TRAIN manifest, with LayerNorm-linear-GELU-linear 256-D L2-normalized embeddings and multi-positive alignment, hard-negative ranking and prefix consistency. Smoke (2 steps), fold0 targeted (500) and four 1k formal folds completed atomically on GPUs 4–7.
- Phase47 retrieval failed Gate R: aggregate raw→learned p16 R@1 0.5093→0.4974, mAP 0.5381→0.5341, hard-gap 0.0128→−0.0110; only fold3 had a small R@1 increase and no fold met both registered margins. No controller reconnect, threshold/memory/proposal/backbone change or public/Q1/DEV+ access occurred. Failed/superseded smoke evidence and checkpoints are retained.
- Decision: `P47_GATE_R_FAIL_STOP_AFTER_SINGLE_INTERFACE_REPAIR`. Under the fixed protocol the authorized representation/controller-interface routes are exhausted; future work requires a new support/supervision or task-definition authorization, not another gate/encoder lottery.

## Phase48 support/supervision contract redesign (2026-08-29)
- S0 read-only audit confirmed a legal causal prior-completed-track support stream for all 76 diagnostic events and 1,672 TRAIN video/category-disjoint multi-positive fit episodes with 1,672 hard negatives. Labels/IDs remained loss metadata only; Phase26 proposal, Phase46 bridge/gate, Phase19R controller/StateMemory and physical MOT were frozen.
- The sole support-conditioned 256-D encoder route passed 100-step smoke, fold0 500-step targeted and four 1,000-step formal folds on GPUs 4–7. An initial smoke failed at import because `PYTHONPATH` was omitted; adding the project path was the minimal repair and preserved the failed marker.
- Gate R48 FAIL: aggregate p16 raw→learned R@1 0.9000→0.7775 and mAP 0.8599→0.7753; only fold0 R@1 increased and mAP declined in all folds. No controller compatibility or sealed evaluation was run. Decision `P48_GATE_R_FAIL_STOP_BEFORE_CONTROLLER`; no further gate/encoder lottery is authorized without a new support/task-definition contract.

## Phase49 raw-preserving controller-aligned correspondence (2026-08-29)
- Stage0/1 contract and precheck passed: frozen Phase26 proposal, Phase41 bridge, Phase46 gate, physical MOT and Phase19R controller remained unchanged; raw fallback is exact 768-D normalized, causal and ID/text/future-free.
- The sole 768-D support residual route ran smoke (100), fold0 targeted (500) and four 1,000-step formal folds on GPUs 4–7. Initial all-raw output exposed a zero-gradient `sigmoid(g)*ReLU(g)`/zero residual initialization; two minimal same-route repairs (tanh alpha, then tiny residual-head init) were retained and regression-tested.
- Fix2 Gate R49 FAIL: p16 aggregate raw→learned R@1 .9000→.9117 (+.0117, below +.02), mAP .8599→.8643 (+.0044, below +.01), only fold2 R@1 improved, and raw-correct→learned-wrong unsafe flips were nonzero in all folds. Controller/sealed evaluation was not run; no second model or threshold/controller change authorized. Decision `P49_GATE_R_FAIL_STOP_BEFORE_CONTROLLER`.

## Phase50 end-to-end MOT+OCD causal architecture (2026-08-29)

- Stage0 official method audit completed without downloads. OVTR (ICLR25), ObjectRelator (ICCV25), C3Po (NeurIPS25), MOTIP-2 (CVPR25), MASA (CVPR24), MeMOTR and MOTR were checked at their official repository HEADs/licences. None satisfies all TrackOCD requirements (causal physical MOT, no text/ID input, cross-video novel correspondence, semantic state and Commit/Defer); no external checkpoint or sealed label was accessed.
- A new TrackOCD-native contract was registered under `docs/iclr27_phase50/`: class-agnostic proposal → physical association → persistent query → causal 768-D raw-preserving state → prior-support memory → correspondence → semantic state/controller. Phase26 proposal/physical rows remained frozen; semantic outputs cannot mutate physical IDs. TRAIN-only four video/category-disjoint manifests contain 1,672 multi-positive fit episodes, 4,620 positive links and 1,672 hard negatives. Leakage/row-key audit passed.
- Contract smoke (including no/invalid-support exact raw fallback) passed; fold0 targeted 500 steps passed; four formal 1,000-step workers completed on GPUs 4–7 with atomic checkpoints/markers. No OOM, duplicate supervisor or external-process event occurred; >25% RAM headroom was retained. Proposal/MOT modules were frozen passthrough and their losses were recorded as zero rather than fabricated detector supervision.
- Gate R50 FAIL: frozen TRAIN-disjoint p16 raw R@1/mAP/hard-gap `0.9000/0.8599/0.1944` fell to learned `0.7867/0.7873/0.0203`; unsafe flip rate `0.1246`, only fold0 had a small R@1 gain, and folds1–3 declined. Training loss reduction is not task success. C50 unchanged-controller compatibility and S50 sealed evaluation were not run because the registered R gate failed; public/Q1/DEV+ remain sealed.
- Decision `P50_GATE_R_FAIL_END_TO_END_ROUTE_STOP_BEFORE_CONTROLLER`. The single registered end-to-end route is complete as a negative evidence route; further work requires an explicitly approved support/task-definition or richer causal detector/track supervision change, not another gate/threshold/controller/backbone lottery. Full artifacts: `docs/iclr27_phase50/PHASE50_END_TO_END_TRAINING_REPORT.md` and `outputs/iclr27_phase50/audit/phase50_decision.json`.
- Phase50 finalization: added explicit prefix 1/2/4/8/16 aggregate and TRAIN fold-coverage tables to the report. Final read-only integrity check parsed all 19 Phase50 JSON files, verified four formal `.launched/.done` pairs, non-empty checkpoints and valid frozen fold-manifest symlink, and found no live Phase50 worker after excluding the inspector process. Report SHA256: `99def008562ecda4c957fc45450e27257b69e14cd50253c9e85824460844a832`.

## Phase51–56 unified end-to-end MOT+OCD route (2026-08-29)
- Phase51 official-method audit used verified repository records for OVTR, ObjectRelator, C3Po, MOTIP-2, MASA, MeMOTR, MOTR and historical OVTrack/COVTrack/VOVTrack. No external code/checkpoint was selected or downloaded: none satisfies the simultaneous causal physical-MOT, no-text/no-ID, cross-video semantic-support and Commit/Defer contract. Architecture contract and forbidden-input audit are in `outputs/iclr27_phase51/audit/{architecture_contract,leakage_audit}.json`.
- Phase53 TRAIN-only inventory passed exact key alignment and four video/category-disjoint folds: 43,423×768 feature rows; 19,379 proposal rows; 14,076 GT-box loss rows; 17,411/4,449 association positive/negative pairs; 4,867 cross-video positives; 3,573 hard negatives; 1,672 multi-positive episodes; 8,360 causal rollouts. No held/DEV+/Q1/public/sealed labels, future rows, text or ID features were used.
- Phase54A–D used one unified graph with trainable proposal/objectness, association/lifecycle, causal track query, raw-preserving 768-D semantic state, support aggregation and Commit/Defer/Reset controller. The selected curriculum was representation-initialized joint training, 4×1,000 steps on physical GPUs 4–7, BF16, atomic checkpoints/markers. Proposal and association losses were non-zero and group gradient norms were logged. Two initial smoke failures (ragged support arrays, BF16 BCE) and one supervisor `wait`/process-substitution failure were retained; minimal fixes passed same-path smoke/targeted regression. No OOM, swap, external PID termination or duplicate formal supervisor occurred.
- Phase56 C1 compatibility smoke passed exact no/invalid-support raw fallback, finite normalized 768-D vectors, causal prefix shape, action logits and inherited physical invariants. Frozen evaluator ran all 76 positive and 76 negative events with prefixes 1/2/4/8/16. A read-only safety-stat audit found raw top-1 comparison accidentally included the query itself; the pre-fix JSON (`phase56_full_evaluation.pre_safety_fix.json`, SHA256 `2775cccf183bf189712a2c1ded8a4a5ee6d535b6e155f60e49814ad178b39812`) was retained, the candidate-set exclusion fix was pycompiled and the identical evaluator rerun. Final full-evaluation SHA256: `66a270c0992c9cbe528d9aaeefc4ccf9d9f8cc5e3655430b561184dc5032a881`.
- Final Phase56 retrieval: raw→learned p16 R@1 `0.9000→0.9468`, mAP `0.8599→0.9441`, hard-gap `0.1944→0.1642`, corrected unsafe flips `0.0089`; Gate R56 **FAIL** because hard-gap non-inferiority and unsafe=0 fail despite 4/4 fold R@1/mAP direction. New causal controller produced 4/76 Commit-CT (folds 2/3 only; categories 3, videos 4), negative false merge `0.0395`, duplicate births `0`, premature `0.5197`, unresolved `0.9539`; Gate C56 **FAIL** on broad coverage and safety. Sealed evaluator was not run; DEV+, Q1 and public new-model labels remain sealed.
- Final decision: `P56_GATE_R56_FAIL_C56_FAIL_STOP_BEFORE_SEALED`. The Phase51–56 registered end-to-end candidate is a negative result; it does not prove universal visual-semantic impossibility. Any continuation requires a newly authorized causal image/video proposal+association and controller-aligned supervision contract, not another gate/threshold/memory/backbone lottery. Integrity check parsed all Phase56 JSON, verified checkpoints/markers, no suspicious public/sealed filenames and no residual Phase54/56 process.
- Final artifacts: Phase56 report SHA256 `a7457278bdd265781ca01b994bffe29be9f14d7697c813b764a817185927d977`, integrated Phase29 report SHA256 `c6e9dbf5f3e69bd3710389038d16f25106916579ddb81cbce0f8a93916282833`, Phase56 decision SHA256 `fefdddc1d3e9778ca627abc64560a7e1c9da6e100e0382ab126bff995c00aa2b`. The integrated decision is `TRACKOCD_PHASE24_56_FINAL_REGISTERED_ROUTE_NEGATIVE_STOP_BEFORE_SEALED`; no sealed/public evaluation was performed.

## Phase57–61 raw-frame pixel end-to-end route (2026-08-29)

- Phase57 official-method audit queried only official repository metadata/README/LICENSE at recorded remote HEADs (OVTR `500e72c19bf5f7f8717546911a5639fdc26bfee5`, MOTIP-2 `012856c1dc13b324064e79339ae71054518d1b5e`, ObjectRelator `59f79d5d0fa5cfc7169b6737fd414c25d1ed83a6`, C3Po `21254a078435451e99d2feabd5db9334c02d8483`, MASA `c5472b9c7615f35abdf1188cb1a0c5408fe50d66`, MeMOTR `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`, MOTR `8690da3392159635ca37c31975126acf40220724`).  No external code/checkpoint was downloaded.  OVTR is the closest persistent-query reference but uses CLIP text/category paths; the other audited projects lack the simultaneous causal, no-text/no-ID and cross-video Commit contract.  Decision `P57_AUDIT_COMPLETE_NO_EXTERNAL_METHOD_SELECTED`.
- Phase58 raw-frame contract uses only `data/raw/tao/annotations/train.json` (SHA256 `7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0`): 18,274 images, 54,639 boxes, 500 videos, 1,230 categories, 2,647 tracks, zero missing frames.  Four fixed video/category-disjoint manifests and leakage audit passed; model inputs contain RGB/causal geometry only, with labels/IDs loss metadata.
- Phase59 implemented an independent pixel causal graph: dense class-agnostic objectness/box/quality, physical association/lifecycle, causal track state, raw-preserving 768-D semantic state, prior-support residual and Commit/Defer/Reset.  `outputs/iclr27_phase58/audit/architecture_contract.json` and the Phase58 report are frozen.  Physical IDs are bookkeeping only; no semantic output mutates parent assignment.
- Phase60 smoke, fold0 targeted and four-fold formal runs used one bounded worker per GPU4–7 (FP32, batch4, two workers, seeds 575700–575703, 1000 updates, atomic 100-step checkpoints).  Preflights show >103 GiB available RAM and ~3.7 GiB RSS/worker; `/data1` 107–111 GiB free; no OOM, swap pressure or external PID kill.  Initial `formal` detector top20 true-IoU≥0.5 recalls were 0.10/0.80/0.00/0.20% (folds 0–3).
- Repair cycle 1 (`repair1`) fixed dense objectness imbalance (explicit positive/negative BCE means; prior `pos_weight=8` learned all-negative).  Smoke/target/formal completed; recalls 0.70/0.40/1.40/0.90%, still poor.  Repair cycle 2 (`repair2`) added normalized absolute `(x,y)` channels after identifying translation-equivariant box-head ambiguity; recalls 0.10/1.00/1.20/1.20%.  Repair cycle 3 (`repair3`, final allowed) supplied every legal TRAIN annotation in each image as objectness positives after identifying single-track false negatives; held categories/videos were filtered per fold; smoke/target/formal completed atomically, recalls 0.10/0.90/0.60/0.50%.  No fourth repair or backbone/controller lottery was run.
- Phase61 frozen evaluation ran all 76 positive and 76 negative events at prefixes 1/2/4/8/16.  Repair3 source reliable=8/76, target reliable=11/76, perfect event ceiling=0/76 at prefix16.  It emitted 76/76 positive commits but also 76/76 negative false commits, premature=1.0 and known/novel confusion=1.0; repair2 was 75/76 with the same negative failure.  The positive count is therefore a degenerate policy, not OCD success.  Standard HOTA/DetA/AssA/MOTA/IDF1 are null because no full-sequence TrackEval exporter exists; inherited physical counters are not claimed as MOT success.
- Phase57–61 gates: P50 **FAIL** (raw image-level proposal source insufficient), R50 **NOT PASS** (no valid learned track-candidate retrieval export), C50 **FAIL** (unsafe all-negative commits), S50 **NOT RUN** (sealed remains sealed).  Public, DEV+, Q1, future rows/tracks, held GT as model input, text and ID features were never accessed.  Reports: `docs/iclr27_phase60/PHASE60_PIXEL_END_TO_END_TRAINING_REPORT.md`, `docs/iclr27_phase61/PHASE61_MOT_OCD_SEALED_EVALUATION_REPORT.md`; machine decision `outputs/iclr27_phase61/final_decision.json`.  The raw-frame route is retained as a negative evidence candidate; a future continuation requires a validated pretrained class-agnostic detector and full-sequence MOT exporter before any semantic/controller claim.

## Phase67 OVTR asset lineage audit (2026-08-29)

- User-authorized route correction: do not train another small RGB detector.  The read-only audit recorded upstream OVTR commit `500e72c19bf5f7f8717546911a5639fdc26bfee5`, MIT license, local mirror status (historical Phase4P/4Q edits), and the official checkpoints `ovtr_det_pretrain.pth` and `ovtr_5_frame.pth` with SHA256 in `outputs/iclr27_phase67/audit/ovtr_assets.json`.
- Historical local lineage was preserved: Phase4Q Q0/Q1/Q2 checkpoints each completed epoch 7 of an 8-epoch, 15,000-iteration/epoch schedule.  Q0 is the selected Phase68 base (score-corrected full TAO stream); Q1/TCO and Q2/DSCQ remain historical ablations, with Q2's known proposal suppression preventing its use as a physical baseline.
- OVTR requires CLIP/category paths for its open-vocabulary head.  Phase68 class-agnostic evaluation must isolate text/category logits and pass only RGB/geometry/query state to physical tracking and later semantic modules.  MOTIP-2, ObjectRelator, C3Po, MASA, MeMOTR and MOTR were recorded as references with incompatible ID/text/static-view boundaries rather than claimed as solutions.
- Phase67 was read-only: no sealed/public/Q1 labels, no new training, no external process termination, and no old-stage files modified.  Next is Phase68 full-sequence OVTR baseline reproduction before any class-agnostic fine-tuning.
Phase69 manifest generation stopped: task-owned PIDs 13931(shell),13932(python); root cause was repeated 382MB source reload in compact() and no atomic output after 12m; no output files were produced, no GPU/RAM incident.
Phase69 manifest repair1 stopped: task-owned PIDs 29496/29497 spent 11m rereading/hashing 382MB lvis_clear_75_60.json with no atomic output; minimal repair is lightweight split manifest plus runtime category/video filtering. No GPU/OOM/external-process event.
Phase69 fold0 targeted failed (task-owned run, GPU4): after one successful iteration the pinned OVTR path raised a CUDA device-side index assertion in `OVTR._forward_single_image` while indexing `image_embeddings[:, select_id]`. Root cause is an inherited label-index contract mismatch: LVIS `cat2label` is zero-based (0..1202), while the local OVTR constructor builds `all_ids = 1..1203`; random padding can select invalid column 1203. The failure marker `fold0_targeted.launched` and log are retained; no `.done`, metrics, or checkpoint were written and no external process was touched. Minimal Phase69-only repair: normalize the wrapper's post-construction sampling IDs to valid zero-based embedding columns (N=1203), then repeat smoke and targeted regression before any formal folds.
Phase69 formal fold0--3 all failed (task-owned PIDs 19016/19020/19024/19028, GPUs4--7) after ~0.6--1.8k updates with no OOM; logs and `.launched/.failed` markers are retained. First formal root cause was the new TRAIN category filter leaving an image with zero valid bboxes, causing the legacy `get_min_wh()` reduction on an empty array. Repair3 adds a pre-filter over LVIS `img_ann_map` (handling both ID and dict forms) and was validated by smoke_fix4 and fold0_targeted_fix2 (nonzero objectness/DSCT losses and checkpoints). Formal repair1 will rerun the registered 7-epoch/15k-update-per-epoch schedule in fresh fold*_repair1 directories, using the same Q0 initialization and GPU4--7 mapping; no external process/OOM event occurred.
smoke_original failed: config pointed policy manifest lacking annotations; repair1 config uses immutable LVIS source + runtime filtering.

## Phase67–70 OVTR asset reuse and semantic integration (2026-08-31)
- Phase67 lineage audit completed read-only at upstream OVTR commit `500e72c19bf5f7f8717546911a5639fdc26bfee5` (MIT).  Official detection/5-frame checkpoint hashes, local Q0/Q1/Q2 lineage, score-field correction, CLIP/text isolation and reusable persistent-query boundaries are recorded in `outputs/iclr27_phase67/audit/ovtr_assets.json`.  No new download or sealed/public/Q1 access occurred.
- Phase68 reproduced the historical Q0 full-sequence stream from raw OVTR frames: 1,268,113 predictions, complete TrackEval artifacts and top-20 IoU>=0.5 recall `0.629993` (71,062/112,798).  A duplicate task-owned TrackEval wrapper was explicitly terminated earlier; the retained direct evaluator completed.  The vendored NumPy compatibility was process-local; no external process or source file was changed.
- Phase69 class-agnostic adaptation used Q0 initialization, video/category-disjoint folds, 7 epochs x 15,000 iterations, batch 1 on GPUs 4–7.  It completed without OOM, but full-sequence top-20 recall@0.5 averaged `0.06749` and macro HOTA `0.05031`, far below Q0; Gate M69 FAIL.  Checkpoints and exact hashes are retained under `outputs/iclr27_phase69/checkpoints/`.
- Phase70 reused the Phase69 DSCT semantic/state route (`semantic_b -> assign/create c -> joint d`) with frozen proposal/physical stream and no new backbone.  The first semantic smoke failed at a relative OVTR path; adding an explicit `chdir` was the minimal repair, and smoke/targeted runs then completed.  The first formal supervisor encountered `/data1` full; intermediate semantic_b/assign_c checkpoints were moved (not copied) to `/home/user/trackocd_phase70_archive/checkpoints/` and symlinked back.  A separate smoke typo (`--dsct_stage joint_d`) was retained as a failed marker; the valid `d` path passed smoke/targeted.  The fixed joint_d repair1 supervisor used only GPUs 4–7 with one worker/fold and completed f0/f1/f3 at 4,000 updates and f2 at 5,000 updates; no OOM or external-process kill occurred.
- Post-training full-sequence validation and single-process TrackEval completed for all four folds.  The atomic artifact `outputs/iclr27_phase70/validation/joint_d_repair1/step_5000_metrics.json` records top-20 recalls @IoU 0.3/0.5/0.7 (`0.11266/0.03924/0.00919` aggregate), macro HOTA/DetA/AssA/IDF1 (`0.02827/0.01260/0.09492/0.01806`), and per-fold prediction/checkpoint hashes.  Compared with Q0 top-20@0.5 `0.629993` and HOTA `0.844035`, this is a decisive proposal/MOT sanity failure; therefore the checkpoint is not eligible for longer training or semantic selection.
- Required stage validation sanity gate: **FAIL** (`STOP_PHASE70_JOINT_REPAIR1_BEFORE_CAUSAL_OCD`).  Retrieval, controller and sealed metrics are explicitly `NOT_RUN`/blocked by the proposal sanity gate; no loss decrease, DSCT gradient or inherited physical counter is treated as success.  No held/public/Q1/DEV+ labels were used as model inputs and no threshold, denominator, seed, evaluator or physical-ID semantics were changed.
- Resource/lineage events retained: `/data1` temporary 0-byte condition, archive symlink ledger, failed path/argument smoke markers, and one accidental duplicate analyzer PID 27529 explicitly SIGTERM-ed while PID 26618 was retained.  No broad kill, OOM or swap event.  The Phase70 route is stopped before OCD because its adapted physical stream is not a valid MOT baseline; future work must use a separately validated class-agnostic detector/association initialization rather than continue this semantic route.
- Final integrity hashes: stage validation `602d51b48d4f219e7d4034d284a1a89d0ddac16ac812a07fc2c6ea071a165e26`, stage marker `1008961d9de099509925946c1d03af341caa602420e4e3286e4a453c81ff0a49`, Phase70 decision `96d66a0fc3233a4615593a081b4f4d5cc5d998803ae94b466bd55c231cef6484`, integration report `948da3b9a502cd1994639a779d812aad019ac9e3500ca30077919493582687cb`, final evaluation report `08b97377de3e697a7ed4002b3d3de1e962b2475744f65d9baecef1893a5dabb6`.  All 25 Phase70 JSON files parsed; 48 checkpoint symlinks resolve; four training, four full-sequence and four TrackEval completion markers are present; no public/Q1/sealed-named output or residual Phase70 worker remained after the blocking validations.

## Active context consolidation and reversible storage cleanup (2026-08-31)
- Added `docs/ACTIVE_TRACKOCD_CONTEXT.md` as the sole active execution summary. It records the final MOT+OCD objective, fixed identity/monitoring contract, Q0 anchor, Phase69/70 negative decisions, unresolved contract risks, and the ordered pretrained physical-MOT → correspondence → controller → sealed plan. Historical reports remain evidence-only.
- Performed a read-only ownership/dependency/process-descriptor audit while `/data1` was ~100% full. Moved completed Phase69 checkpoint artifacts (38,496,264,372 bytes), Phase70 checkpoint artifacts (15,804,648,261 bytes), Phase70 smoke artifacts (2,807,786,054 bytes), and Phase70 targeted artifacts (2,807,770,557 bytes) to `/data2/usr_for_deadline/trackocd_archive/20260831/`. Original paths are symlinks; all 144 archived files and retained authority paths validated. No active descriptor, Luna session, code, evaluator, report, metric, manifest, or failure evidence was deleted. Full ledger: `docs/CLEANUP_20260831_MANIFEST.md`.

- The fixed target Luna session `01a01fb6-96f7-7132-a318-0833180c88d8` was resumed with the canonical long-run continuation prompt after its Phase70 `task_complete` marker. A separate new thread was not created, avoiding duplicate work. The session record emitted fresh task/reasoning events at 11:37; the remote app wait endpoint could not read the host, so this is recorded as an observability limitation rather than a completion or failure.
- The resumed session acknowledged `OPEN/RUNNING` and started the new Phase71 Q0 asset/interface audit, score-mode equivalence replay and preregistration. No new training route or gate claim has yet been made.

## Phase71 Q0-preserving physical route (2026-08-31)

- Stage A read-only audit completed in the independent `docs/iclr27_phase71/` and `outputs/iclr27_phase71/` namespace.  The immutable Phase4Q Q0 checkpoint (`809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738`) and full prediction stream (`1,268,113` records; SHA256 `112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac`) were checked.  Recomputed top-20 IoU>=0.5 recall is exactly `71062/112798=0.6299934395999929`, matching Phase68.  Five-field CSV key audit has 28,315 rows, zero malformed/duplicate keys, and ordered digest `2f204885776e655129ca893fe7bbdf9321ff0449f938c3e3544e787aaa263c0e`.
- Q0 score semantics are explicitly `score_mode=base`; the exported JSON has no pre-filter/DSCT/objectness channels.  Phase71 emits a compressed sidecar with `base_score=raw_score` and null unavailable channels rather than silently reusing them.  Legacy OVTR CLIP/text tensors remain constructor compatibility paths; they are frozen and not exposed to the new class-agnostic TCO input.  Physical IDs remain bookkeeping only.  Static text/ID/future findings are preserved in `outputs/iclr27_phase71/audit/leakage_audit.json`; no DEV+/Q1/public/sealed labels were accessed.
- Fixed TRAIN-only video/category-disjoint policy metadata were materialised as lightweight Phase71 manifests (source LVIS annotation SHA256 `ee9f4bc8253ac7502291e591f831f272ce3467ab6c8e0edf61a9ebf2ce7fe204`); annotation/HDF5 data were not copied.  Fold mapping is GPU4/5/6/7, with GPU0 occupied by an unrelated process and untouched.  Preflight recorded >115 GiB available RAM, no swap, `/data1` ~72 GiB free and `/data2` ~1.2 TiB free.
- The only registered B route is a Q0-initialized trajectory-conditioned objectness (TCO) quality/lifecycle adapter.  `train_tco_fold.py` loads Q0 and freezes every parameter except `tco_head`; base score, decoder/query, parent assignment and physical lifecycle remain the Q0 contract.  The bounded supervisor enforces one `.launched`/`.done` unit per fold and refuses duplicate launches.  Smoke (20 updates), fold0 targeted (100), then a seven-epoch/full-update four-fold run are required before any correspondence/controller route.  A physical MOT sanity failure blocks all semantic work; loss decrease alone is not a gate.
- First Phase71 smoke (`smoke1_f0.launched`) failed before dataset construction with `ValueError: dataset lvis not supported`; the wrapper omitted OVTR's registered `lvis_generated_img_seqs` dataset name.  Failure evidence and exit marker are retained; no checkpoint or metric was produced.  Minimal repair is to pass the Q0-compatible dataset name plus `with_box_refine/two_stage` and sampling arguments, then rerun under a fresh tag before targeted training.
## Phase71 Q0-preserving physical route (formal training/evaluation status, 2026-08-31)
- Phase71 TCO adapter smoke/targeted repairs completed, then the registered four-fold `formal1` run completed on GPUs 4–7 from the immutable Q0 checkpoint. Each fold ran 15,000 iterations (seven-epoch schedule), produced a non-empty 240,021,717-byte checkpoint and an atomic `.done`; no OOM or external-process termination occurred. Per-fold training wall times were approximately 13:21–13:57 and the no-gradient guard only counted legal all-new-query batches.
- The first full-sequence TCO validation attempt (`formal1_tco`) incorrectly launched four memory-heavy evaluators concurrently. Available RAM fell below the 25% safety floor (about 14 GiB available); task-owned supervisor/evaluator/DataLoader PIDs 11448, 11459/11463/11467/11471, 11461/11464/11469/11472 and 12712/12589/12449/12325 were explicitly SIGTERM-ed. Partial logs and `.launched` markers are retained; no `tao_track.json`, `.done` or metric was produced, so this attempt is not a result. External GPU0 PID10750 was not touched. The failed markers record exit 143 and `resource_memory_floor`.
- A serial evaluator (`scripts/iclr27_phase71/run_eval_serial.sh`, fresh tag `formal1_tco_serial`) was added as the minimal resource-safe repair: one worker on GPU4 at a time, the same Q0-compatible evaluator arguments, atomic markers and no protocol change. It completed all four folds and emitted valid predictions; the resulting physical metrics are recorded in the Phase71 report and decision.
- Valid serial validation metrics: top-20 IoU≥0.5 recall Q0 `0.6299934396` → TCO `0.6261081757` (folds `0.630419/0.613743/0.631501/0.628770`); macro HOTA `0.8440349013` → `0.7948236678`, DetA `0.6940826053` → `0.6442467188`, AssA `1.2822744737` → `1.2350673026`, IDF1 `0.8395552303` → `0.7603061760`. Physical Gate P71 therefore FAILS; correspondence/controller/sealed remain blocked/not run.

## Phase72 OCD metric audit and frozen baseline test (2026-09-01)
- Stage A streamed the Q0 TAO JSON and the four unique P71 serial TAO streams (TrackEval symlink duplicates deduplicated by real path). All contain only `bbox`, `category_id`, `image_id`, `score`, `track_id`, `video_id`; none has the TrackOCDEvaluator `prediction_type`/semantic/virtual/action or causal representation fields. The 76 positive and 76 negative Phase19R event manifests are complete (12/12/24/28 per fold and 152 total) but have zero source/target `(video_id, track_id)` intersections with Q0/P71. Decision: `q0_p71_ocd_status=NOT_RUN_INTERFACE_MISMATCH`, `p71_learned_ocd_status=NOT_RUN_BLOCKED`; this is not OCD=0.
- Stage A first parser was interrupted (exit 130) after a per-record full-buffer `lstrip`; the cursor-based streaming decoder was the minimal repair. No training/GPU job was started. The prescribed pytest command returned exit 5 because the legacy file is a `main()` script with no pytest-collected functions; direct execution of the same nine cases passed 9/9 and left the old test-report hash unchanged. The one-positive/one-negative causal smoke and fold1 targeted regression required only local checker fixes for legacy `tracklet_position`, JSON set serialization, and independent source/target position checks; both passed.
- Stage C ran the registered `scripts/iclr27_phase19r/run_internal_evaluation.py --candidate raw --device cpu` path serially over all four folds and all 152 events. Manifest/replay keys match exactly, no records are missing/duplicated, and source/target positions are monotonic. The frozen diagnostic aggregate is 1/76 correct persistent Commit-CT, post-prefix CT 3/502, existing precision/recall 3/593 and 3/502, negative false merge 26/76, negative false commit 47/76, premature 48/152, unresolved 58/152, duplicate births 87/152, NMI 0.268770 and ARI 0.035444; fold Commit-CT is `[0/12, 0/12, 0/24, 1/28]` with category/video coverage only in fold3. This result is `phase19r_native_frozen_baseline_diagnostic`, not P71/Q0 OCD or final MOT+OCD.
- Phase72 report and status: `docs/iclr27_phase72/PHASE72_OCD_METRIC_AUDIT_AND_FROZEN_TEST_REPORT.md`, `outputs/iclr27_phase72/status.json`. No DEV+/Q1/public-new/sealed labels, future rows/tracks, category text or ID features were accessed; no OOM, swap, external-process termination or duplicate worker occurred. TrackOCDEvaluator per-track metrics remain `NOT_APPLICABLE_INTERFACE_MISMATCH` (null, not zero). Next legal action is a separately registered, strictly audited per-track causal semantic/action exporter joining Q0/P71 physical lineage to the evaluator schema; no threshold/controller/StateMemory/backbone lottery is authorized by Phase72.

## Phase73 Q0 lineage reconciliation and causal exporter contract (2026-09-02)

- Registered and executed an isolated, CPU-only audit under `scripts/src/docs/outputs/iclr27_phase73/`; no training, controller replay, threshold search, sealed/DEV+/Q1/public-new access, or mutation of Q0/old evaluator occurred. The audit used one bounded process and closed its atomic `RUNNING.lock`; no residual Phase73 process, OOM, swap, or external-process termination was observed. `/data1` was 99% full (~37G free) but the job wrote only small audit/export artifacts; available RAM was ~110G and the cache target is `/data2/usr_for_deadline/trackocd_phase73_cache`.
- Input hashes and counts are stable: Q0 checkpoint SHA256 `809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738`, Q0 TAO stream SHA256 `112d185e1a7d94495491d919d59045f0e474b5e2df1ab1c0fb6317f64bbab2ac` with `1,268,113` records, positive event manifest SHA256 `6442d1a32cf6a0dfdd6bacc04b42e1ba41d9708b5aa8480079202b17dafdadd2` and negative manifest SHA256 `9673b928df45934080a5f9ed2c7aa0a31f585846e2ce5e66c8957c2baac829fc`. The 152-event denominator and fold counts (12/12/24/28 per polarity) are preserved.
- The explicit five-field event/CSV key is `video_id:frame_id:proposal_local_id:track_id:image_id`; event physical bookkeeping keys are `v<video>:p<track>`. Q0 TAO rows contain only `bbox, category_id, image_id, score, track_id, video_id` (TAO `xywh` boxes); they lack `frame_id`, `proposal_local_id`, `assigned`, semantic/action fields. Direct `(video_id,track_id)` event intersections are `0/152`; all `91` event videos and all `1,422` event-referenced image IDs have `0` Q0 rows in the frozen stream. Thus an evaluator-only temporal IoU mapping cannot be proven, rather than being silently guessed.
- Full alignment audit emitted `1,520` records (152 events × 2 roles × 5 prefixes) with every unmatched/null record retained. Across every prefix and both source/target roles, temporal mapping and reliable joined observations are `0`; event-side reliable-row counts are audit metadata only (source/target at p16 `524/424` rows), not Q0 performance. Mapping alternatives (deterministic ID remap, temporal bbox IoU, arbitrary tie/guess) were considered; the legal result is `UNAVAILABLE`/lineage unknown because no shared image/video evidence exists.
- Isolated model-facing plumbing emits a bounded 1,000-row `CONTRACT_NULL_POLICY` sample independent of event manifests: representation `null`, `prediction_type=unresolved`, `action=DEFER`, uncertainty `1.0`; evaluator adapter rows are post-hoc and metrics are `null` with `NOT_RUN_NO_SEMANTIC_MODEL`. Static/runtime leakage audits pass (no category/text/ID/future/GT fields in model tensors), causal positions are monotonic, and pytest collection is `3 passed`.
- Required metamorphic/protocol JSON tests (category shuffle, event-label swap, future append, physical-ID renumbering, Q0 hash preservation, evaluator protocol preservation, repeat determinism, atomic output) all pass. Q0 hash was unchanged before/after; old files are read-only and no historical file was modified.
- Decision: `PHASE73_BLOCKED_LINEAGE_UNKNOWN` (`outputs/iclr27_phase73/status.json`, marker `PHASE73_BLOCKED`). This is a plumbing/interface block, not `OCD=0` and not evidence against the Q0 MOT stream. Phase73 is intentionally stopped here; no Phase74, correspondence training, controller tuning, or sealed evaluation may start until Desktop ChatGPT reviews/provides an authorized support/lineage contract. Complete evidence is in `docs/iclr27_phase73/PHASE73_Q0_LINEAGE_AND_EXPORTER_CONTRACT_REPORT.md` and the adjacent audit/export/test artifacts.
- Phase73 source/test code was committed and pushed after execution as public commit `a3f789c` (`Add Phase73 Q0 lineage exporter audit`). The public repository received source only; reports, outputs, logs, data and checkpoints remained local/ignored.

## Phase74 repair, asset reconciliation and frozen-Q0 replay audit (2026-09-02)

- Implemented an isolated `src/iclr27_phase74/` / `scripts/iclr27_phase74/` / `tests/phase74/` audit. The registered positive/negative manifest order (76+76), source-before-target causal prefix contract, tracklet alignment, canonical asset identity, five-field lineage checks, leakage checks, and metamorphic tests were executed without training, semantic/controller replay, threshold search, or sealed/DEV+/Q1/public-new access. `26 passed` Phase74 tests and the legacy direct evaluator smoke exited 0.
- Reconciled the asset namespaces: all 1,422 Phase19R event images exist under TAO TRAIN (91 canonical videos), while the frozen Q0 stream/annotation is TAO validation (988 videos; 1,268,113 rows). Canonical event-to-Q0 mapping is therefore 0/1,422; integer IDs are namespace-local and were not used as a mapping. Q0 TAO rows do not carry `frame_id` or `proposal_local_id`, so the five-field physical lineage remains explicitly unavailable rather than guessed. Branch B is selected, but its required exact Q0 control replay was not run.
- Phase74 status is `PHASE74_BLOCKED_Q0_REPLAY_EQUIVALENCE`; no event full-video replay or OCD metric is claimed. Historical Q0 `25/76` is retained as a reference and marked not directly comparable until the exact control replay is independently reproduced. Static OVTR CLIP/text/category references remain `TEXT_CATEGORY_DEPENDENCY_UNKNOWN`; no semantic-stage qualification was made.
- A stale task-owned companion scanner (parent PID 39610, child 39611) was found after artifacts were complete. Only those explicit PIDs were sent SIGTERM (with bounded fallback); no external process was touched. Failure-taxonomy JSON remained complete and hash-stable (`ca3662…d185` / `7040d7…3fbd`). No OOM, swap, GPU worker, or duplicate supervisor event occurred. The event is recorded in `outputs/iclr27_phase74/audit/process_events.json` and `status.json`.
- Phase74 code/tests were committed and pushed to the public repository as commit `a40ab00` (`Add Phase74 asset reconciliation and Q0 replay audit`). Reports, outputs, logs, data, and checkpoints remain local/ignored. Required next action is Desktop ChatGPT review and authorization of one reproducible Q0 validation control replay; Phase75, correspondence, controller tuning, and sealed evaluation remain unstarted.

## Phase74R harness and asset revalidation (2026-09-02)

- Created the independent `src/iclr27_phase74r/`, `scripts/iclr27_phase74r/`, `tests/phase74r/`, `outputs/iclr27_phase74r/` and `docs/iclr27_phase74r/` namespace. The audit runner is CPU-only and does not invoke Q0, train a model, run a controller, or access DEV+/Q1/public-new/sealed labels. Phase74/Phase19R files remain read-only.
- Repair attempt r1 failed before audit artifacts with `NameError: EVENT_POS` in the runner preregistration. The minimal constant-name fix (`EVENT_POSITIVE`/`EVENT_NEGATIVE`) was applied; the failure and closed lock marker are retained. Repair attempt r2 then exposed an over-strict prefix source-text check that incorrectly required frame/image literals; the frozen runtime actually orders its index by `event_rank`. The checker was corrected to verify the real `idx.sort(event_rank)` expression plus the Phase74R projection tie-break. The report-renderer import failure (`ModuleNotFoundError: src` when invoked directly) was separately fixed and retained in `audit/repair_events.json`.
- Smoke and targeted tests after the repairs passed `9/9`. Final run `phase74r-final-20260902-r3` reconstructed the actual historical fallback model event order: 82 rows (41 positive + 41 negative), independently reproducible. The frozen evaluator metadata remains 152 rows (76 positive + 76 negative; fold totals 24/24/48/56), with **0 key matches**. `EVALUATOR_DENOMINATOR` is therefore the only failed mandatory gate; decision/status is `PHASE74R_BLOCKED_DENOMINATOR`.
- Asset identity is complete and duplicate-preserving but has no legal cross-namespace mapping: 36,375 Q0 validation records versus 1,422 event records, content-key intersection 0, all event records `NO_CONTENT_MATCH`. The synthetic Branch-A content-key fixture passes (`UNIQUE_MAPPING`), and overlap versus physical fragmentation is classified separately. This is evidence of a protocol/lineage mismatch, not a detector or OCD failure.
- Before a valid Q0 replay every observation field is null with `NOT_AVAILABLE_Q0_NOT_REPLAYED`; 1,520 records (152 events × 2 roles × 5 prefixes) were emitted and no zero/no-detection conclusion was made. Prefix, reliability, causality, no-leakage, metamorphic, artifact-format and resource-safety gates pass. No Phase75A/B replay, semantic/controller route, threshold tuning, or sealed evaluation is authorized until Desktop ChatGPT provides an exact hash-identified event-stream contract.
- Resource pre/postflight used one CPU process and zero GPUs; no external process was terminated. At final preflight, RAM was 125 GiB total/~118 GiB available, `/data1` ~36 GiB free and `/data2` ~1.2 TiB free; the Phase74R run left no active process. Source/tests/report-generator were pushed before report generation as public commits `deb2652` (`Add Phase74R harness and asset lineage revalidation`) and `5b082ba` (`Fix Phase74R report generator entrypoint`). Generated reports/outputs remain local/ignored.

## Phase75B autonomous repair (2026-09-02)

- The first full TRAIN event replay (`event_full_sequence`) failed before inference because the pinned `lvis_generated_img_seqs` path entered OVTR's LVIS loader and expected `coco_url`, which TAO TRAIN annotations do not contain. The minimal repair kept the old failure directory intact, added a fresh-tag runner, and exposed the immutable TRAIN JSON through an atomic `/data2/usr_for_deadline/trackocd_phase75b/train_validation.json` symlink so the existing TAO loader is selected without copying data. Code/config was pushed as `809ccdd`.
- Repair smoke then reached the TAO loader but failed because TRAIN images provide `frame_index` rather than OVTR's `frame_id`. The wrapper now assigns an in-memory contiguous causal `frame_id` per video after sorting by `(frame_index,image_id)`; the source annotation remains read-only. The repair was pushed as `5d5d981`; the failed `smoke_repair1` evidence remains unchanged.
- Fresh `smoke_repair2` completed on GPU0 with one TRAIN video (40 frame trace rows, 2,069 native records), no OOM or external-process event. The registered next action is a bounded targeted replay followed by the full 91-video replay; no event labels are joined before model inference and public/DEV+/Q1/sealed data remain sealed.

## Phase75B observability and Phase75C R registration (2026-09-02)

- The repaired Q0 TRAIN replay completed all 91 event videos (3,373 causal frame rows and 145,429 native physical records). The post-inference O audit found positive prefix16 reliable source/target/both coverage of 49/76, 40/76 and 25/76 respectively; fold both counts were [8,2,10,5]. The pre-registered O anchor therefore passes at 25/76 with nonzero evidence in all four folds. This is an observability result, not an OCD or controller result.
- Three exploratory CPU matching snippets were accidentally left running after their tool windows returned partial output. They were task-owned diagnostic PIDs 23159/24457-24458/25434-25435 and 29310-29311; each was explicitly SIGTERM-ed (with bounded SIGKILL fallback where needed). No external process, GPU job or historical output was touched. The snippets produced no artifacts and are not results; future matching code will be bounded and committed before execution.
- Phase75C is registered as one Grounded Correspondence route based on the official ICML 2026 Rethinking Temporal Consistency repository (commit 5d345268797425558b449337519af3ab24aeb6f1, MIT). Its central operation is frozen-feature, causal set/temporal correspondence with deterministic one-to-one matching; no text, category, ID, future or controller input is used. Q0 physical lineage, the 152-row model/evaluator contract and O artifacts remain frozen. A learned model will not be started until the route implementation, manifests and contract tests are committed and pushed.

## Phase75C grounded correspondence execution (2026-09-02)
- The single registered Grounded Correspondence route was committed/pushed before execution (`a0ca478`, followed by minimal metadata-index repair `cfd2d32`). The route is parameter-free: frozen DINOv2 CLS/ROI (0.8/0.2), causal consistency-weighted aggregation, normalized 768-D output; no checkpoint or learned weights were introduced.
- Contract smoke passed (finite 768-D output, exact prefix causality, forbidden-input metadata). The first retrieval attempt failed before evaluation with `TypeError: 'int' object is not subscriptable`: `by_track()` returns row indices, while the new metadata loader dereferenced an index as a row. Failure evidence is retained at `outputs/iclr27_phase75c/audit/repair_events.json`. The minimal `rows[ordered[-1]]` fix was pushed as `cfd2d32`; smoke, direct tests and fold0 targeted regression then passed. The OVTR environment does not include pytest, so the direct regression was used and recorded rather than treating missing pytest as a scientific failure.
- Full four-fold TRAIN-derived video/category-disjoint retrieval completed on the exact Phase30 validation manifests. At prefix16 raw→Grounded: R@1 `0.893219→0.893817` (+0.000597), mAP `0.848374→0.852533` (+0.004159), hard-negative gap `0.189559→0.196521`; there were 3 raw-correct→Grounded-wrong unsafe flips. Substantial registered folds (R@1 +0.02 and mAP +0.01) = `0/4`, directional folds = `1/4`, so decision is `P75C_GATE_R_FAIL_STOP_BEFORE_CONTROLLER`.
- No controller, StateMemory, threshold sweep, 152-event replay, sealed/public/Q1/DEV+ access or GPU training was performed for Phase75C. The complete report is `docs/iclr27_phase75c/PHASE75C_GROUNDED_CORRESPONDENCE_REPORT.md`; machine metrics and integrity hashes are under `outputs/iclr27_phase75c/`. This route is retained as a bounded negative representation result, not detector failure, OCD=0, or final-task completion.
- Final report rendering used pushed source commit `b0dd4ce` (verified literature URL/report provenance addition). Integrity artifacts record metrics SHA256 `598aabe0aca704b9608faa3b80acf75cbc898b30d657286a0a1d39eec14be5cb`, literature SHA256 `09de571b8bbaf5b25629ddb4eadaaf784f53d60c520e55fca6cf4cf26028188e`, and report SHA256 `fc8e15a105ccbd664f275fa830f25d3c0de91b64ab696cf83a6b2baf8e431b8f`.

## Phase75D pairwise trajectory correspondence and Phase75E adapter (2026-09-02)

- Phase75D used frozen Phase15S DINOv2 43,423-row/1,298-track five-field alignment and the Phase30 fit/validation manifests. It ran exact R-global and explicit causal R-legal Hungarian frame-set scoring for prefixes 1/2/4/8/16, with no learned parameters, threshold sweep, controller, StateMemory, sealed, DEV+, Q1 or public-new access. Raw parity passed 20/20 fold-prefix checks at tolerance 1e-7.
- Phase75D strict gates failed: global p16 fold-macro R@1 `0.913618` versus raw `0.963415` (Δ `-0.049797`), mAP `0.951780` versus `0.952443` (Δ `-0.000663`), unsafe `58/984`; legal p16 R@1 `0.483740` versus `0.333333` (Δ `+0.150407`), mAP `0.755448` versus `0.718383` (Δ `+0.037065`), unsafe `45/984`, and legal hard-gap non-worse was false. The registered teacher signal was nevertheless present: legal ΔmAP `+0.025376`, positive hard-gap folds `3/4`, global ΔmAP `+0.003885`, no global fold R1 drop below −0.02, so `P75D_PAIRWISE_SIGNAL_AUTHORIZE_P75E` was recorded without calling it a Gate pass.
- The Phase75D interpretation was narrowed: Phase75C's implementation was consistency-weighted temporal pooling with a fixed pair diagnostic, not the complete official Grounded Correspondence temporal identity method. Historical Phase75C files were not modified.
- Phase75E was registered as one rank-8 raw-preserving feature adapter (A 768→8, B 8→768, B=0, scale 2), fixed Adam 4e-5/seed42/15,000 steps, loss `0.5 rank + 1.0 raw reconstruction + 1.0 safety`, explicit Phase30 fit episodes only. Code commits `dbfdee9`, `12048a3`, `582ba3e`, `1fce31f`, `f8a88f2`, `f98033f`, `e0acc75`, `c5bee70`, and `667cbaf` were pushed to GitHub before final report generation; private reports/outputs/checkpoints remain ignored and local.
- Repair evidence is retained for four ordinary implementation failures: frozen legal builder import, query/support prefix cache collision, legacy torch `weights_only` incompatibility, and raw comparator field misuse in exact aggregation. Each kept its failure artifact/marker and used a minimal repair without changing seed, denominator, candidate universe or protocol. Fresh smoke and targeted runs completed after repairs (100 and 500 updates), followed by four 15,000-step workers on GPUs 4/5/6/7 under one bounded supervisor; no OOM or external process termination occurred.
- Exact Phase75E r3 is authoritative. Global p16 learned versus raw is R@1 `882/984=0.896341` versus `948/984=0.963415`, mAP `0.769969` versus `0.848374`, hard-gap `0.087382` versus `0.189559`, unsafe `75/984`; legal p16 learned versus raw is R@1 `599/984=0.608740` versus `328/984=0.333333`, mAP `0.795111` versus `0.718383`, hard-gap `0.031361` versus `-0.036633`, unsafe `36/984`. Global and legal strict Gate R therefore fail; controller/StateMemory/Commit-CT/sealed were not run.
- Formal best checkpoints were selected only by the registered lexicographic TRAIN-disjoint validation rule: f0 step1000, f1 step2500, f2 step14500, f3 step8000. Long-run drift was severe (last probe cosine f0 `-0.18299`, f1 `0.54471`, f2 `0.72123`, f3 `0.51790`), supporting primary root cause `RAW_GEOMETRY_DRIFT`, with `FOLD_IMBALANCE`, `HARD_NEGATIVE_FAILURE` and `PREFIX_INSTABILITY` as secondary evidence. The legal retrieval improvement is not a final OCD result and cannot authorize controller replay.
- Tests: Phase75D `7 passed`; Phase75E six contract functions passed directly in the pinned OVTR environment. The requested Phase75E pytest command was attempted but pytest is not installed in that environment; this is retained as a dependency caveat, not relabeled as a pytest pass. All Phase75D/E Python files and the report generator compile.
- Final Phase75D/E reports and machine decisions are generated from JSON artifacts by `scripts/iclr27_phase75d/generate_report.py`. Current stop is `P75E_GATE_R_FAIL_STOP_BEFORE_CONTROLLER`; no evidence-backed frame-quality/temporal aggregation bottleneck was isolated, so no optional TFA/TCR component, backbone lottery, threshold change, controller run or sealed evaluation was started. Next work requires a separately authorized single visual-only causal quality hypothesis, if evidence warrants it.

## Phase76R contract errata and Pareto audit (2026-09-02)

- Added an isolated Phase76R namespace. Contract tests passed 2/2; source/preregistration was pushed before the long audit as public commit `b4fb1c2`.
- Corrected the historical Phase75D teacher checker to apply the −0.02 R1 guard to global folds. The old legal-first result was `true`; corrected result is `false` because global folds 0 and 2 have ΔR1 below −0.02. Phase75D status and report remain unchanged.
- Recorded Phase75E errata: actual optimizer was constant Adam LR `4e-5` with no scheduler, formal seeds were `750500+fold`, and legal-first checkpoint selection omitted global unsafe/R1/hard-gap/raw-cosine constraints.
- Replayed all 120 retained Phase75E step checkpoints (30 per fold, 500–15000) with exact p16 global/legal metrics, drift quantiles and delta norm/raw. No diagnostic checkpoint met the registered safe window; decision is `PHASE76R_NO_SAFE_FEATURE_ADAPTER_WINDOW`. GPU count was zero, no OOM or external process event occurred, and `.launched/.done` markers are complete. The first raw-parity attempt differed by ≤1.79e−7 due to redundant normalization; the failed artifact is retained, the minimal raw-anchor fix passed all 20/20 checks at ≤1e−7.
- Phase76A is the next route: frozen raw global scorer plus bounded candidate-level relation residual; no controller/StateMemory/sealed evaluation is authorized unless its independent global/legal/memory gates pass.

## Phase76A anchored local relation reranker (2026-09-02)

- Revalidated official references with pinned commits: RethinkingOCL `5d345268797425558b449337519af3ab24aeb6f1`, SlotContrast `55ec66dc02eeade630805789ef4a6c5df06f21ff`, TRACT `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a`, and COVTrack `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`; references only, no external model code/weights used.
- Phase76A contract smoke passed (1536-D symmetric tokens, 13-D summary, zero-initialized delta/confidence, exact raw step-0). Initial raw parity had ≤1.79e−7 drift from redundant normalization; failed artifact retained and the operation-order fix passed all 20 fold×prefix checks at ≤1e−7.
- Deterministic Phase30 fit/val candidate banks and detached Hungarian index caches were prepared (fit banks `[31,539,536,566]`, val `[837,82,37,28]`, 97MB total). Preparation r1 failed from a prematurely closed temporary-file descriptor; r2 completed after minimal fix. No data/protocol change.
- One fold0 100-step smoke and 500-step targeted run completed on GPU4 with atomic checkpoints/validation. Four formal workers then ran 20,000 steps on GPUs4–7 under one supervisor, AdamW 1e−4→1e−5 (1000 warmup/cosine), validation/checkpoint every 500, deterministic visit imbalance ≤1, no OOM or external process termination.
- Exact independent TRAIN-disjoint val replay used selected checkpoints f0=1000, f1=12500, f2=8000, f3=17500. Raw structural parity passed, but local learned p16 aggregate R@1 `0.506730` vs raw `0.606246` (Δ `−0.099517`), mAP `0.560056` vs `0.627041` (Δ `−0.066985`), hard-gap Δ `−0.050207`, unsafe `94/984`; only fold0 improved and had 61 unsafe flips. All legal/prefix safety gates failed (`PHASE76A_GATE_R_FAIL_STOP_BEFORE_STATE_MEMORY`).
- Root classification: `LEGAL_OVERFIT`, `FOLD_IMBALANCE`, with `LOSS_CONFLICT`/`CONFIDENCE_FAILURE` evidenced by unsafe local reorderings. No evidence supports frame-quality/geometry repair, so no second encoder/component, StateMemory, controller, Commit-CT or sealed evaluation was started. Phase76B/C reports explicitly record blocked/not-run (not zero).

## Phase76AR selective relation correctness repair (2026-09-02)

- Registered an isolated raw-first route after auditing Phase76A. The erratum confirms the historical implementation had no true dual stream, one scalar quality weight instead of per-match weights, fold0 `banks[:128]` validation bias, no bank-aware confidence, unbounded delta, top8-only safety, and p16-only hard-negative construction. Phase76A files/status remain read-only.
- Built independent `MemoryMimicBank` and `LegalFitEpisode` loaders from frozen Phase30 TRAIN manifests. Memory banks use <=3 positives and a prefix-union (top4 per prefix, <=12) raw-hard-negative set; legal banks read `kind=multi_positive_cross_video` records directly. Four atomic 30 MB/fold pair caches include five per-match quality features and no forbidden inference fields. Stream counts are fit/val `[31,539,536,566]` / `[837,82,37,28]` for both streams; hashes are in `outputs/iclr27_phase76ar/audit/build_summary.json`.
- Implemented a bounded `0.10*tanh` residual, per-match quality MLP, bank-context gate (top1/top2 margins, spread, entropy, count) initialized near abstention, all-negative raw safety, and deterministic fold0 hash-stratified validation selection. Contract direct tests passed 4/4; the pinned OVTR environment has no pytest, so the pytest invocation was retained as a dependency caveat rather than counted as a pass. Code was pushed before execution (`55c8dd4`, `af03128`, `dad6135`, `4d92f31`, `efb963f`, `8d2ba42`).
- One smoke attempt failed before inference with a BCE shape mismatch because the query-level bank logit was repeated per candidate; failure marker was retained, the smallest reduction fix was applied, direct tests passed, and a fresh smoke plus fold0 targeted run completed on GPU4. The smoke/targeted policies stayed raw at step 100/500 with no unsafe flip. The first all-fold exact evaluator timed out at the bounded 300 s tool boundary (exit 143, no partial artifact); a resumable fold/stream evaluator was added and the event is recorded in `outputs/iclr27_phase76ar/audit/repair_events.json`.
- Four 10,000-step formal folds ran on GPUs4/5/6/7 under one supervisor, AdamW 1e-4→1e-5, 500-step checkpoint/validation, without OOM or external process termination. Exact full validation covered all prefixes and 984 queries per stream. Legal p16 raw→learned: R@1 `0.609892→0.609892`, mAP `0.802159→0.802159`, hard-gap `+0.000072`, unsafe `0`; memory p16: R@1 `0.606246→0.606246`, mAP `0.641163→0.641408`, hard-gap `+0.000143`, unsafe `0`. Teacher agreement was `0.922` legal / `0.945` memory, but intervention was zero legal and ~1.35% memory with no R1 gain.
- Decision is `PHASE76AR_GATE_R_FAIL_ROUTE_TO_PHASE76S_OR_PHASE76G`: all safety gates pass, but no stream reaches the registered +0.02 R1/+0.01 mAP or 3/4 substantial-fold requirement. This is an abstention/no-generalized-signal result, not a controller or StateMemory result; no controller, held event, DEV+, Q1, public-new, or sealed input was accessed. The pre-authorized next route is Phase76S selective/abstaining routing (then 76G/76X/O as evidence dictates).

## Phase76S selective/abstaining relation router (2026-09-02)

- Added an isolated three-way HELP/HARM/NEUTRAL router over frozen Phase76AR counterfactual diagnostics. The router input is limited to raw/relation margins, gate/delta summaries and bank context; HELP is the only action that can use the frozen relation scores, otherwise the exact raw scores are returned. COVTrack ICCV 2025 was audited as a confidence-routing reference only; its category-aware semantic cue is not used here.
- TRAIN/validation examples were generated from the frozen Phase76AR relation checkpoints. Validation label counts were all NEUTRAL for every fold (f0 4185, f1 410, f2 185, f3 140); fit HELP examples were only 28 total (f1=5, f2=23) and HARM was zero. This is direct evidence that the frozen relation did not provide a validation counterfactual improvement for the registered router to learn.
- A 100-step GPU4 smoke and 500-step fold0 targeted run completed with finite logits, exact raw fallback and zero unsafe flips. One exact-evaluation invocation failed at import because the direct script call omitted `PYTHONPATH=.`; no artifact was written, and the same path was rerun with the minimal environment fix before continuing.
- Four 2,000-step formal folds ran once under the bounded GPU4/5/6/7 supervisor without OOM or external-process intervention. Exact TRAIN-disjoint validation covered all 4,920 examples and every prefix. Aggregate p16 raw→router was R@1 `0.609892→0.609892`, mAP `0.802159→0.802159`, hard-gap `+0.000012`, unsafe `0`; router HELP rate `0.0331`, teacher agreement `0.9669`, and teacher-use rate on validation `0.0`. No registered +0.02 R1/+0.01 mAP or 3/4 substantial-fold gate was met.
- Decision: `PHASE76S_GATE_R_FAIL_ROUTE_TO_PHASE76G`. This route is a safe raw-preserving abstention result, not a controller/StateMemory result; no held/public/DEV+/Q1/sealed data was read. The next authorized route is Phase76G cross-category meta-holdout/group-robust training.

## Phase76G cross-category meta-holdout route (2026-09-02)

- Registered deterministic four-way groups inside each fold's TRAIN fit categories (no held validation category entered training) and a rotating 3-of-4 group objective: weighted cross-entropy plus 0.5 worst-group loss. Manifest counts/hashes are under `outputs/iclr27_phase76g/manifests/`.
- GPU4 smoke (100 steps) and fold0 targeted (500 steps) passed with exact raw fallback and zero unsafe flips. Four 2,000-step formal workers ran once on GPUs4/5/6/7 under one bounded supervisor; no OOM or external process termination occurred.
- Exact TRAIN-disjoint validation again produced p16 raw→learned R@1 `0.609892→0.609892`, mAP `0.802159→0.802159`, hard-gap `+0.000042`, unsafe `0`, HELP route rate `0.0608`; all four fold deltas were zero and worst-fold checks were non-negative but not positive. Decision: `PHASE76G_GATE_R_FAIL_ROUTE_TO_PHASE76X`.
- The negative result indicates that reweighting/meta-holdout training cannot create signal when the frozen Phase76AR relation has no validation counterfactual improvements. No controller, StateMemory or sealed/public/DEV+/Q1 data was accessed.

## Phase76X soft optimal-transport primitive (2026-09-02)

- Audited official correspondence references before implementation: ObjectRelator (ICCV 2025, `https://github.com/insait-institute/ObjectRelator`), C3Po (NeurIPS 2025, `https://github.com/c3po-correspondence/C3Po`), Grounded Correspondence (ICML 2026, `https://github.com/LiZhYun/ICML2026-RethinkingOCL`), and SlotContrast (CVPR 2025, `https://github.com/martius-lab/slotcontrast`). Only the correspondence primitive idea was used; no repo code/weights or text/ID inputs were imported.
- Implemented exactly one parameter-free symmetric uniform-marginal Sinkhorn matcher (temperature 0.07, 50 iterations) with a fixed 0.5 raw-score anchor. Contract smoke verified finite and symmetric scores; no GPU/training was used.
- Legal p16 raw→OT R@1 `0.609892→0.634898` (+0.025007), mAP `0.802159→0.817727` (+0.015568), but hard-gap `−0.003962`, unsafe `14/984`, and mAP substantial folds `2/4`. Memory p16 R@1 delta `+0.015779`, mAP delta `+0.018970`, hard-gap `−0.005834`, unsafe `15/984`; gates fail. Decision: `PHASE76X_GATE_R_FAIL_R_EXHAUSTED_UNDER_FROZEN_FEATURE_PROTOCOL`.
- The OT gain is therefore a retrieval diagnostic with unsafe flips, not a safe correspondence gate; controller/StateMemory/held/sealed/public evaluation remained unrun.

## Phase79O causal physical-observability route (2026-09-02)

- After all R routes failed, registered one O route using the frozen Q0 native lineage: strictly prior one/two physical-track boxes, constant-velocity projection (max gap 2), and retention of all raw candidates. This is causal trajectory aggregation only; no labels, IDs as features, future rows, or semantic/controller code are used.
- Official TRACT/trajectory-aware tracking and COVTrack confidence-aware tracking references were checked (`https://github.com/Nathan-Li123/TRACT`, `https://github.com/zekunqian/COVTrack`); their text/category cues are explicitly excluded. Code/config were pushed as `2ff3b66` before execution.
- The route generated 33,334 synthetic candidates for 1,378 of 1,422 event image keys, but prefix16 positive perfect-observation ceiling remained exactly `25/76`; fold counts stayed `[8,2,10,5]`. Decision: `PHASE79O_GATE_O_FAIL_R_EXHAUSTED_UNDER_FROZEN_PHYSICAL_STREAM`.
- This leaves the five pre-authorized substantive routes (AR/S/G/X/O) exhausted under the frozen feature/proposal protocol. No retrieval route passed with safe, multi-fold generalization; no controller or sealed evaluation is scientifically authorized.

## Phase80+ autonomous research window (2026-09-02 UTC)

- `research_start_utc=2026-09-02T18:27:32Z`; registered deadline `research_deadline_utc=2026-09-03T04:27:32Z` (10 hours). Starting HEAD: `3c8dd33844e80947c9cbbd5aef64c07411cbb917`; working tree clean and origin/main matched at start.
- Phase80 closes the frozen Phase15S 768-D relation-engineering family as historical evidence. The new family is visual-representation source renewal: audit existing dense/DINOv3 work, verify official 2025/2026 sources, then select one new dense/local visual evidence source before any training. DEV+/Q1/public-new/sealed labels remain sealed; no held outcome is used for selection.
- Resource preflight: 125 GiB RAM (102 GiB available), GPUs 0/1 externally occupied (4.1/29.9 GiB), GPUs 2–9 idle, `/data1` nearly full (34 GiB free), `/data2` has 1.2 TiB free. Any new cache/checkpoint will be placed on `/data2` with a project symlink; no external process will be touched.
- Extraction repair cycle 1: the first four-shard supervisor failed before writing shards 1–3 because 22 edge-touching boxes were narrower than the previous 4-pixel crop guard (shard counts 1/12/9); no cache was written and no GPU process remained. The minimal fix expands only these rows to a deterministic 4-pixel window, preserving all row keys and the denominator. Shard0 targeted extraction remained valid (10,855 rows); shards 1–3 retain their `.launched` and `.failed.json` evidence until explicitly archived for the repair rerun.
- Diagnostic repair cycle 1: the first full dense replay counted four Phase30 validation queries lacking either a positive or negative cross-video candidate, whereas the frozen Phase75D scorer excludes them. This produced a false raw-parity failure (988 vs 984). The minimal fix retains those keys in an `unevaluable` audit list and excludes them from all retrieval/rescue metrics, matching the frozen scorer contract; no data or denominator policy was changed.
- Phase80A dense-source diagnostic completed after the repair. Exact raw parity passed (`R@1=0.8932193826961726`, `mAP=0.8483743525266084`, 984 evaluable queries). At prefix16 the DINOv3 TIMM dense patch relation scored `R@1=0.835144`, `mAP=0.822818`, with 9 rescued raw-wrong and 17 harmed raw-correct queries (net `-8`); three of four folds had an R@1 drop below `-0.02`. The new global CLS scored `R@1=0.891365`, `mAP=0.838037` (net rescue `+3`) but did not meet the registered dense routing criterion. No Phase80B dense router or controller replay was run. The complete diagnostic and retained failed-cycle artifact are under `outputs/iclr27_phase80a/`; readable report: `docs/iclr27_phase80a/PHASE80A_DENSE_VISUAL_DIAGNOSTIC_REPORT.md`.
- Phase80A routing decision: close the visual-source sub-route and pivot to the preregistered Family B causal-memory-matched supervision route. This route will use the existing frozen DINOv2 visual source but train a sequential candidate-set evidence scorer over TRAIN-only memory-mimic banks; it does not change proposal, physical tracking, controller, thresholds, denominator, or sealed boundaries.
- Phase80B contract audit completed on the read-only Phase76AR memory-mimic banks. Fit/validation distributions are strongly imbalanced (for example p16 raw top-1 positive rates fit→val: fold0 `0.4839→0.2485`, fold1 `0.0779→0.9390`, fold2 `0.0448→0.5946`, fold3 `0.0512→0.6429`), supporting a causal-memory/sampling mismatch hypothesis. A duplicate audit invocation was accidentally started after a tool-window return; only the later task-owned PID 3170 and parent shell 3167 were SIGTERM-ed explicitly, while the original PID 2322 completed and wrote the atomic audit JSON. No external process, GPU job, or data artifact was touched. The event is retained for the Phase80B report and is not a scientific result.
- Family-B smoke (100 updates) and fold0 targeted (500 updates) completed with finite stateful outputs and exact raw step-zero fallback. Targeted fold0 p16 mAP changed by only `+0.000260`, R@1 was unchanged, and no intervention/unsafe flip occurred. The first exact replay attempt failed at checkpoint loading because the pinned torch lacked the `weights_only` keyword; no metrics were written. A compatibility fallback was applied and a four-row fold0 load/evaluation regression completed; the failed invocation remains recorded for the report.
- Family B formal four-fold training completed once under the bounded GPU4/5/6/7 supervisor (5,000-update target; best steps f0=500, f1=2,000, f2=5,000, f3=1,000). Exact TRAIN-disjoint replay after the loader compatibility repair produced aggregate p16 raw→learned R@1 `0.606246→0.581414`, mAP `0.641163→0.594240`, hard-gap `0.046725→0.032875`, unsafe flips `8/984`; fold R1 deltas were `[0.000000,-0.036585,-0.027027,-0.035714]` and mAP deltas `[+0.000260,-0.080686,-0.057493,-0.049772]`. The stateful memory route failed its registered safety/generalization gate and was not connected to the controller. The first exact replay failed only because the pinned torch rejected `weights_only`; the minimal try/except loader fix was committed and pushed as `374cb00`.
- Family C evaluator-only proposal/observability audit completed on the frozen Phase75B Q0 trace (760 prefix rows; 76 positive events, prefixes 1/2/4/8/16). At prefix16, Q0 candidate-pool IoU≥0.5 exists on source `72/76` and target `64/76`, but strict event-level reliable assignment is only source `49/76`, target `40/76`, both `25/76`; candidate-present assignment/temporal gaps are `23` source and `24` target. Per-fold both-reliable counts are `[8/12, 2/12, 10/24, 5/28]`. The quality audit is diagnostic only (no fitting or held selection): source pool-good rows have mean max score `0.5123` vs `0.3984` for pool-bad; target `0.5241` vs `0.4820`. This supports an assignment/temporal reliability-headroom hypothesis but does not authorize a new ranker or controller route. Outputs are under `outputs/iclr27_phase80c/audit/`.
- Family C temporal sub-audit separated pool coverage from causal lifecycle: candidate-present source gaps average temporal IoU `0.1699` with `11.96` fragmentation transitions, and target gaps `0.1045` with `5.00` transitions, while event-reliable source/target sides average `0.4863/0.6206`. This is evaluator-only evidence for a future physical-association repair, not a new representation result.
- Family C quality correlations are descriptive only: temporal IoU vs event reliability was `0.637` (source) and `0.736` (target), while maximum q0 score was `0.079/-0.155` and candidate count `-0.232/-0.147`. This reinforces that causal track completeness, not simple candidate score, is the actionable bottleneck.
- Family D trajectory route audit completed without downloading or executing TRACT. Local official checkout is commit `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` (ICCV 2025, Apache-2.0 MASA license). The release requires external detector proposals/MASA tracks and its TraCLIP path constructs CLIP text/class-name cues; it has no persistent TrackOCD semantic StateMemory/Commit-Defer or category-free cross-video correspondence implementation. It is therefore rejected as a primary Phase80 route; only trajectory aggregation ideas are retained as literature context. Artifact: `outputs/iclr27_phase80d/audit/tract_route_audit.json`.
- Family D modern-source search audited official TrajViT (ICCV 2025, main `8fe9949dd86435bebc2c35d8b23d77a019c487a2`) and Trace Anything (ICLR 2026, main `54677b5e7bf11510c2e8c917a509988ad379f8eb`). TrajViT exposes trajectory tokens but its released pretraining assumes 8 GPUs, external SAM2 trajectories and image/video caption metadata; no license file was exposed. Trace Anything exposes scene-level 4D trajectory fields (Apache-2.0 code, CC BY-NC 4.0 weights), not semantic track embeddings or causal cross-video correspondence, and its examples require ≥48 GB VRAM. Neither was downloaded or executed; both are retained as audit-only references. Artifact: `outputs/iclr27_phase80d/audit/modern_trajectory_audit.json`.
- Phase80+ finalization: all registered Family A–D work in this window is complete; no controller/Commit-CT/sealed/public evaluation was authorized after the failed representation gates. The window ended at approximately `2026-09-02T20:17:27Z` (1.83 h) because no compliant, reproducible next route remained under the frozen contract: Family C requires a new physical-association implementation, while audited Family-D implementations require external/text/caption dependencies or unavailable resources. This is an early route stop with negative evidence, not a claim that TrackOCD is globally impossible. Final report: `docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md`; ledger: `outputs/iclr27_phase80/validation_evidence_ledger.json`.
# Phase81P+ (2026-09-03, Q0-anchored physical association route)

- Registered `2026-09-03T08:40:36Z` through `18:40:36Z`, starting HEAD `75edd12e2ae58a6decde02f860949786af844598`; Q0 detector/proposals/base score, row keys, denominator and causal evaluator remain frozen. Outputs/checkpoints use `/data2/usr_for_deadline/trackocd_phase81p` via a project symlink; no DEV+/Q1/public-new/sealed input is accessed.
- Phase80C evidence is retained: p16 Q0 pool-good source/target `72/76`, `64/76`; strict both reliable `25/76`; 36 events have pool candidates but assignment/temporal gaps. Phase81P taxonomy is evaluator-only and does not use held events for training.
- Resource preflight at start: 125 GiB RAM, ~97 GiB available, GPUs 2–9 idle and GPU1 externally occupied; planned mapping GPUs 4–7, bounded workers and >=25% RAM headroom. `/data1` is full, so no large data is written there.
- Manifest attempt `build_train_manifest.py` was stopped at 2026-09-03T08:56Z after ~8 minutes with no output: an unbounded per-video history made candidate sorting quadratic. Only task-owned PIDs 25998/25999 were terminated; no partial manifest/data was present and no external process was touched. Minimal repair prunes history to the registered 8-frame causal window before candidate construction; smoke will rerun on a fresh tag.
- Formal event replay first run (supervisor PIDs 15776/15786) completed folds 1–3 but fold0 produced no artifact after ~10 minutes and its active-track matrix grew, indicating unbounded runtime memory under low match scores. Only those task-owned PIDs were SIGTERM-ed; no external process was touched. Fold1–3 artifacts are retained as superseded diagnostics. Minimal repair adds a fixed causal 256-track memory bound (evict highest miss/lowest support after assignment) and a fresh four-fold replay tag; proposal rows and denominator remain unchanged.


## Phase81P+ final association-family closure (2026-09-03)

- Registered window `2026-09-03T08:40:36Z`→`2026-09-03T18:40:36Z`, start HEAD `75edd12e2ae58a6decde02f860949786af844598`, ending HEAD `319fc3b91e8bb93398c9c6f54569d4990ba066d0`. Q0 proposal/boxes/base score, 76-event denominator, row keys and causal evaluator stayed frozen; DEV+/Q1/public-new/sealed remained sealed.
- Q0/Phase80C reference: source/target pool-good 72/76 and 64/76, strict source/target/both 49/76, 40/76, 25/76; 36 assignment-gap events (29 fragmentation, 7 missed continuation).
- Initial association training/replay exposed NEW-context calibration mismatch and physical collapse (mean 362 tracks; see route JSON). Masked candidate context repair (`f111b11`) retained evidence but still produced 2,895 mean tracks, 7,354 switches, 403 fragmentation, 1,197 merges, 6,645 duplicate births.
- Resolution-aware repair (`2946a92`) used actual current-frame dimensions (event rows include 1280×720, 1920×1080/1200, 640×480) and was TRAIN/event causal; physical mean 2,834 tracks, 7,343 switches, 403 fragmentation, 1,085 merges, 6,025 duplicate births.
- Final top-4 candidate-conditioned NEW context (`832ad95`) improved some single-video continuation counts but physical mean remained 2,714 tracks, 7,250 switches, 403 fragmentation, 1,028 merges, 5,606 duplicate births; fold3 had only five continuations. All three learned versions failed the pre-registered Q0 physical gate.
- Event O proxy means were initial 49.75/76, masked 60/76, resolution-aware 60/76, top-4 57/76. These are not MOT/OCD success and did not authorize controller/R/sealed evaluation.
- No OOM or external-process termination occurred. Large artifacts use `/data2/usr_for_deadline/trackocd_phase81p` via `outputs/iclr27_phase81p` symlink. All code changes were pushed before report generation; route decisions and hashes are in `outputs/iclr27_phase81p/audit/`.
- Decision: close this physical-association model family after three evidence-based versions; do not threshold-sweep. Next research should first build a TrackEval-native Q0-compatible TRAIN supervision/lifecycle contract, then register a new route only with independent causal validation.

## Phase82P+ Q0-preserving residual fragment repair (2026-09-03/04)

- Registered a ten-hour Q0-anchored physical-association route at `2026-09-03T16:39:00.669566Z` with deadline `2026-09-04T02:39:00.669566Z`. Q0 detector/proposals/base score, row keys, denominator and causal O evaluator are frozen; only TRAIN supervision may train the residual. GPUs 4--7 are reserved for this task; GPU1 remains an external process. Outputs/checkpoints use the `/data2/usr_for_deadline/trackocd_phase82p` target through the project symlink.
- Strict O parity completed before training: all 152 positive/negative event rows and p16 positive both-reliable `25/76` (prefix counts 17/22/22/23/25; p16 source 49/76, target 40/76). No DEV+/Q1/sealed/public-new labels were read.
- Q0 DINOv2 appearance extraction first failed on all four shards because the frame root was incorrectly resolved under `annotations/frames`; the failure logs and original `.launched` markers are retained. The minimal repair used the verified `TAO-Amodal/frames` root and a bounded four-row smoke, then a four-GPU `repair1` extraction completed 111,387/111,387 rows with zero failures. Shard hashes and merged feature hash are recorded in sidecars (`878451f2178f117c0919a3b2688bcb494077e1e63ea2ace3ab3b7b47163de902`).
- Manifest construction after extraction exposed a schema-only mismatch: the observation concatenation is 8 box/center/size + 4 causal scalars + 4 velocity + 32 projected appearance = 48 dimensions, while the registered constant/config still said 49. This was the first actionable failure; no data artifact was written. The minimal fix changes only the observation/model/config dimension to 48, preserving all row keys, causal fields and labels; it will be smoke-tested and targeted-regressed before formal training. Code was pushed before rerunning (next commit after `fbac396`).
- The rebuilt manifest initially contained 33,594 birth examples but zero candidates/positive reconnects because TRAIN Q0 images are sampled every 30 raw frames while the registered 16-step causal window was applied to raw frame numbers. The minimal chronology repair maps each video's sorted observed frame indices to ordinal causal steps (strictly monotonic, no future access), matching runtime observation-step semantics without changing rows or denominator. This repair is being compiled and targeted-regressed before formal training; the zero-candidate artifact remains as failure evidence.
- A tool-window return left two duplicate manifest builders alive (task-owned PIDs 15010 and 15854 with descendants 15856/15857). Both were explicitly SIGTERM-ed; no external process was touched. Their mtime checks show the prior zero-candidate manifest and fold arrays were not overwritten. The duplicate launch is retained as a process-integrity event; the next rebuild will use one blocking supervisor only.
- The repaired manifest completed with 33,594 examples and 1,127 positive reconnect labels; validation positives by fold are 228/308/340/251 and observed-history means are 1.83/2.10/1.76/2.38. Two-step smoke, fold0/500 targeted and bounded four-fold/1,000-step formal training all completed on GPUs 4--7 with valid checkpoints and atomic markers. Every formal fold predicted KEEP_Q0 for validation (false reconnect=0, repair precision/recall=0), so training loss/accuracy are not evidence of association improvement. Full native event appearance extraction/replay was not started because the 145,429-row prerequisite would exceed the remaining registered window; a 16-row native smoke after skipping no-box termination records passed.
- Phase82P residual evidence therefore remains conservative and negative under TRAIN validation: Q0 strict p16 both-reliable is 25/76, no learned O/R/C/sealed result was claimed, and the pre-authorized next route is selective overwrite/full joint assignment in a fresh window. All source changes through this point were pushed to origin/main (`40cf4cb`).

## Phase82R+ Q0-preserving performance route (2026-09-04)

- Registered independent ten-hour window `2026-09-04T02:17:53.672Z`→`2026-09-04T12:17:53.672Z` from clean HEAD `a5b7fe1a4489cd0b15245bf0035ca58181a002b0`; GPUs 4–7 are reserved, Q0 proposal/base score/denominator/evaluator and sealed boundaries remain frozen. Registration and route contract were committed/pushed as `d93260f`.
- Phase82R residual-signal diagnostic found a first actionable input bug before training: the inherited Q0 DINOv2 cache was extracted with TAO `bbox` stored as `(x,y,w,h)` but the crop helper interpreted it as `(x1,y1,x2,y2)`. The cache had exact repeated vectors (many pair cosines 1.0), while direct corrected RGB-crop inference differed (fresh/cache cosine about 0.01–0.10 for sampled rows). The failed direct audit first hit a PIL crop error; this was a diagnostic-only invocation and wrote no scientific artifact. Corrected Phase82R smoke conversion produced 8 distinct vectors (unique=8; sample cosine 0.44–0.89), proving the issue is cache extraction, not DINO model output. Phase82P artifacts are untouched; corrected extractor/audit were committed/pushed as `22cba01`, `98f4d15`, and `2d23fba`.
- Existing residual signal statistics remain diagnostic only: 1,127 positive reconnect examples; candidate rank recall top1/4/8/16 `0.6202/0.8447/0.9201/1.0000`; raw appearance pair AUC `0.4992` is invalidated by the crop-key bug and must be recomputed after corrected extraction. No TRAIN model was started before this repair. Next action is a four-GPU corrected Q0 appearance extraction, followed by the registered balanced two-stage residual; no controller/StateMemory/threshold/backbone/public or sealed evaluation is authorized before physical sanity gates.
- Corrected four-shard extraction completed at `2026-09-04T03:22Z` with 111,387/111,387 rows and no failures; merged cache SHA256 is `735bd5bf037666382f2995804825cc321c7f42a1c35389de33ca9bcec6601c0f`. Recomputed TRAIN causal diagnostic gives 1,186 positives, candidate top1/4/8/16 `0.6872/0.8929/0.9545/1.0000`, corrected DINOv2 pair AUC `0.7902` (positive mean cosine `0.7378`, hard-negative `0.5245`). This supersedes the invalid Phase82P appearance statistic without changing row keys or labels.
- Phase82R manifest rebuilt in its own `/data2/usr_for_deadline/trackocd_phase82r/data` target: 33,594 examples and folds with fit positives `935/864/835/924`; validation positives `251/322/351/262`. A first wrapper smoke caught a missing `scripts` import path; the one-line path repair was pushed as `2431d23`, smoke then passed and full build completed without overwriting Phase82P.
- Balanced two-stage residual implementation (`BalancedResidualGate`: one-layer causal GRU, class-balanced gate BCE plus positive-only candidate CE, raw KEEP fallback) was committed/pushed as `b92f24f` and gate-dimension repair `6dd9916`. Contract smoke verified finite outputs, nonzero gradients, 16-way logits and empty-support KEEP. Fold0 targeted 500 updates produced nonzero validation reconnect (`0.0790`) with recall `0.1873`; four 15-epoch formal folds completed on GPUs4–7 with checkpoints/markers and no OOM. Natural validation final reconnect rates/repair recall were f0 `0.0950/0.1673`, f1 `0.2808/0.1242`, f2 `0.0940/0.0912`, f3 `0.1909/0.1985`; these are TRAIN-disjoint diagnostics only and include substantial false reconnect, so no physical/O/R/C gate is claimed. Native event appearance/replay remains pending.
## Phase82R+ execution continuation (2026-09-04)

- Corrected native feature extraction was reused once (145,429 rows; 768-D feature SHA256 `fecdfc3bf341fc28f81fed2fa19dba57063c49793a083f3d1c18835a9d722245`; native lineage SHA256 `d33e60f4603aaa8aa744d8d73553b42153be9f9b88a3a19aa6eb26884d31a2e1`). Selective Q0 overwrite at the single TRAIN-fixed `p>=0.9` gate completed all folds and was strict-O identical to Q0 (25/76, source 49/76, target 40/76).
- Full causal replay was repaired in two minimal cycles: `c342611` enforces dormant/terminated-only candidates and excludes current-frame continuations; `8ff54c3` adds same-frame canonical-ID collision fallback while preserving every detection. The pre-repair TrackEval duplicate-ID failure is retained as evidence.
- Collision-safe `full_causal_r3` processed 91 event videos/145,429 rows in 93.7 s (RSS 1.09 GiB, no OOM): 6,385 reconnects, 19,624 keeps, 562 collision groups, and zero bbox-bearing duplicate IDs. Cheap proxy versus Q0: tracks 971 vs 1,026; GT switches 2,723 vs 2,766; fragmented GT tracks 353 vs 354; merged tracks 334 vs 383; duplicate-birth proxy 1,757 vs 1,902; reliable assignments unchanged at 7,784.
- Class-agnostic TRAIN TrackEval (diagnostic copy only; 91 videos/3,373 frames/9,830 GT boxes, original TRAIN JSON untouched) completed: Q0 macro HOTA/DetA/AssA/MOTA/IDF1/IDSW/Frag `14.619/6.815/32.246/-822.32/7.749/2647/748`; full causal `14.732/6.815/32.724/-821.94/7.960/2610/748`.
- Frozen strict-O full causal r3 remains 25/76 (source 49/76, target 40/76; folds 8/2/10/5), so no O improvement and no retrieval/controller/sealed authorization. This is physical-lineage progress plus a negative strict-O result, not a universal infeasibility claim; a future route would need genuine jointly trained association/source supervision rather than threshold or controller tuning.
### Phase82R full learned causal association (2026-09-04)

- The corrected all-row TRAIN manifest was built from the key-aligned DINOv2 cache and Q0 stream: 84,489 causal action examples across 43 non-event videos.  Candidate states are strictly prior observed steps (horizon 16, K=8, max 16); labels are used only for TRAIN supervision and never enter inference tensors.  Fold fit/validation existing-assignment counts are (4773/1788), (4760/1801), (5143/1418), (5007/1554); arrays are stored on `/data2/usr_for_deadline/trackocd_phase82r/full_assoc_data` and referenced from Phase82R manifests.
- Manifest smoke first exposed a missing `frame` field in the read-only observation helper state; the minimal fix added causal `frame`/`age` metadata and was smoke-tested on one video.  Full rebuild completed without OOM (RSS peak about 3.8 GB).  The initial shell used the system Python without torch; the supervisor was corrected to the audited `/home/lwr/anaconda3/envs/ovtr/bin/python` runtime before any formal training.
- FullAssociation is a separate one-layer-GRU model with explicit NEW+existing masked CE over all rows.  Contract smoke passed finite logits, nonzero gradients, empty-candidate NEW fallback and 48-D causal observations.  Fold0 500-update targeted run learned nonzero existing decisions (val existing precision/recall 0.278/0.277; pred_existing 0.0691), so the route proceeds to four-fold formal training.  No DEV+/Q1/public/sealed labels were accessed.
- Full learned causal replay (`full_assoc_replay_r1`, f0 selected by TRAIN-validation existing-F1) preserved all 145,429 native rows and made 509 reconnect decisions.  Physical proxy was non-inferior to Q0 (tracks 1,023 vs 1,026; GT switches 2,758 vs 2,766; fragmented tracks 353 vs 354; merged tracks 372 vs 383; duplicate-birth proxy 1,858 vs 1,902).  TRAIN class-agnostic TrackEval was HOTA/DetA/AssA/IDF1/IDSW/Frag = 14.647/6.816/32.324/7.838/2644/747 versus Q0 14.619/6.815/32.246/7.749/2647/748.  Frozen strict-O remained exactly 25/76 (source49, target40; fold [8,2,10,5]); no controller, StateMemory, threshold sweep, DEV+/Q1/public/sealed evaluation was authorized.
- A single registered raw-appearance-anchor association variant added an explicit causal cosine feature while retaining the same 48-D observations, candidate set, seed, and loss.  Four-fold validation existing recall was .404/.360/.293/.402; replay made 620 reconnect decisions and physical proxy remained safe (tracks1,024; switches2,759; fragmented353; merged366; duplicate-birth1,852), but strict-O again stayed 25/76.  TRAIN TrackEval HOTA/DetA/AssA/IDF1/IDSW/Frag = 14.707/6.818/32.598/7.859/2645/747.  The diagnostic raw-anchor validation sweep was stopped after ~10 minutes by explicit SIGTERM to task-owned PIDs 31552/31553 because it was redundant with the registered replay; no external process was touched.  This is a recorded diagnostic interruption, not a scientific result.
- A fixed parameter-free temporal appearance-mean route (running normalized sum of each causal track's observed DINOv2 vectors) was then smoke-tested and replayed on all 91 event videos.  It produced 8,035 reconnects with collision-safe row preservation.  Physical proxy improved versus Q0 (964 tracks, 2,723 GT switches, 353 fragmented, 327 merged, duplicate-birth proxy 1,731), while strict-O remained exactly 25/76.  TRAIN class-agnostic TrackEval was HOTA/DetA/AssA/IDF1/IDSW/Frag = 14.755/6.813/32.825/8.018/2606/748, the best physical diagnostic in this window but not an OCD result.
- Phase82R escalation decision: physical lineage has small non-inferiority gains, but neither learned association route improves the frozen 25/76 observability ceiling.  Under the registered rule, downstream retrieval/controller/sealed stages remain NOT_RUN; the evidence points to proposal observability and event-support coverage as the active bottleneck, not a claim that the association model has no TRAIN signal.

## Phase83 dual-path physical→R / O-support (2026-09-04)

- Registered from `7132667` with Q0/Phase75B rows, five causal prefixes, 76 positive + 76 negative denominator and sealed boundaries frozen. Outputs/checkpoints use `/data2/usr_for_deadline/trackocd_phase83` through the project symlink; no DEV+/Q1/public-new/sealed labels were accessed.
- Stage0 support-assignment callgraph is read-only: `assigned`, `row_iou` and `track_temporal_iou` originate in the upstream role-row table and are preserved by the corrected builder; Phase75B computes event reliability, while native q0 max-IoU is post-hoc pool evidence. p16 pool upper bound is source/target/both `72/76,64/76,61/76`; frozen event reliability is `49/76,40/76,25/76`. Taxonomy counts across all 76 positives: B(pool max IoU<.5)=15, D(assigned/transformed-IoU gap)=18, E(pool good but assignment selection gap)=36, G=7; no proposal-missing class.
- Branch A initial physical-R process (PID 17813; wait shell 17963) was stopped explicitly after profiling showed quadratic repeated raw-vector recomputation and no artifact. Caching per-track vectors was the smallest repair; smoke/targeted and formal reruns completed. Exact TRAIN validation p16 mixed-universe temporal mean R@1/mAP `0.882735/0.847251` vs raw `0.893219/0.848374`, hard-gap `0.198022` vs `0.189559`, unsafe flips `5`, with only 1/4 folds non-decreasing in both metrics. Native mapping covered 1,046/6,213 tracks; the mapped-only view also declined. Decision `R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT`.
- Branch B used non-event TRAIN videos/roles only, excluding all 91 event videos; input tensor contains score/geometry/causal history/density and corrected-DINO temporal scalars, never GT/IDs/category/text/future. A 100-step smoke, 500-step fold0 targeted and four 1,000-step formal CPU folds completed with finite atomic checkpoints and no OOM. At p16, learned support selection was 46/76 positives and 52/76 negatives, but both-side reliable support was only 8/76 versus frozen 25/76; negative reliable selections were 10/76. Decision `O83_FAIL_NO_SUPPORT_GAIN`; no controller/StateMemory/threshold/backbone/sealed evaluation was run.
- Reports/artifacts: `docs/iclr27_phase83/{PHYSICAL_TO_R_REPORT,O_SUPPORT_REPORT}.md`, `outputs/iclr27_phase83/audit/{support_assignment_callgraph,failure_taxonomy_76,failure_taxonomy_summary,phase83_decision}.json`, `outputs/iclr27_phase83/metrics/{physical_r_temporal,o_support_replay_formal}.json`. All code changes were compiled, committed and pushed before execution (latest report generator fix pending commit). Next action requires a new, evidence-based support/assignment contract; do not run controller or model lottery on Phase83 evidence.

## Phase83 resume window (2026-09-04)

- The first Phase83 window was finalized prematurely after the registered R83 and O83 routes failed, despite substantial time remaining. The same fixed window was resumed at `2026-09-04T09:03:35Z` (original deadline `2026-09-04T17:43:07Z`) from clean HEAD `5db09123655368e643a29664ae77feed9ebb6ce6`; `outputs/iclr27_phase83/audit/resume_status.json` and `finalization_lock.json` preserve this correction. No prior result or marker was removed.
- A2 is registered as an inference-only full public-TRAIN Q0 physical-lineage route: reuse the pinned Q0 checkpoint and native OVTR exporter over all 370 Phase30/75D public videos, with no detector retraining, no raw fallback for unmapped rows in the headline, and no sealed/DEV+/Q1 access. `scripts/iclr27_phase83/inventory_full_r.py`, `configs/iclr27_phase83/ovtr_q0_public_train.py`, and `scripts/iclr27_phase83/run_a2_full_q0.py` were added; inventory confirms 43,423 public rows, 6,213 tracks and 984 queries, while historical/native streams cover only 49/91 public videos and therefore cannot serve as complete A2 evidence.
- B2 remains the independent contract-level listwise O-support assignment route, to be run only after A2 physical coverage is audited. It must reconstruct real per-image candidate sets, use deterministic IoU-soft TRAIN labels only for loss, expose an explicit DEFER choice, and report 76-positive/76-negative replay without changing the old `assigned` fields. Neither A2 nor B2 has been used to select a sealed/public checkpoint.
- A2 smoke and 20-video targeted Q0 inference completed on GPU4 before an unrelated job occupied that card. The full 370-video lineage then completed successfully (`682,335` native rows, `13,678` image traces; checkpoint SHA256 `809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738`). A later resource snapshot showed GPU4 at ~8.9 GiB and GPUs5–8 idle; the appearance supervisor was therefore minimally remapped to GPUs5–8 before launch, with no external process termination.
- A2 full native appearance extraction and merge completed on the frozen Q0 lineage (682,335 rows, corrected DINOv2 cache SHA256 `ed4405f7946f87579c086c332db743e59c79b9d046ab55ca756a1aea46723714`). The causal temporal-R replay is explicitly incomplete rather than headline evidence: only 23,341/43,423 proposal rows mapped at IoU≥0.5 to the native stream, 74/76 positive events had both sides mapped, and mapped-subset p16 R@1/mAP `0.880983/0.850773` versus raw `0.909890/0.851003` (hard-gap `0.191834` vs `0.193455`, unsafe 8). No raw fallback was used to hide unmapped rows; A2 remains `A2_R_INCOMPLETE_MAPPING_NO_HEADLINE_GATE`.
- B2 listwise O-support is now being implemented as the second independent registered route. Its candidate-set manifest excludes all 91 event videos from fit/validation, groups non-event public TRAIN rows by `(video_id,image_id)` with deterministic candidate order, uses TRAIN `assigned/row_iou` only for an explicit candidate-or-DEFER target, and keeps all forbidden fields out of the model tensor. The binary row-quality router is not reused. Manifest build and subsequent listwise smoke/targeted/formal runs must remain causal and post-hoc event replay only.
- B2 listwise smoke tag `b2_smoke` trained 100 updates and wrote an initial checkpoint, but replay failed before metrics/.done because the loader restored scalar `bc/bd` as `(1,)` arrays, making the DEFER logit two-dimensional. The marker/checkpoint are retained as failed evidence; the minimal loader fix converts both values to scalars. A fresh tag is required for smoke/targeted, with no overwrite of the failed tag.
- B2 repair smoke (`b2_smoke_repair1`) and fold0 targeted (`b2_targeted_repair1`) completed with finite set-aware logits, explicit DEFER outputs, causal feature tensors and atomic checkpoints. The targeted fold0 validation selected candidates on 129/3803 groups and achieved candidate top-1 recall 0.4619 (top-5 0.9361, top-20 0.9988), but candidate-or-DEFER accuracy was only 0.1770 because the validation distribution is strongly shifted; p16 event replay selected support on all 12 positive and 12 negative fold0 events, with 7/12 positive and 4/12 negative both-side reliable selections. This is a targeted diagnostic, not a gate; the registered four-fold formal run proceeds without changing protocol.
- B2 formal listwise training completed all four folds at 1,000 updates with atomic checkpoints and no OOM. Validation candidate top-1 recall was `[0.4152, 0.7304, 0.5405, 0.2286]`; video/category-disjoint fold sizes were `[3803,855,85,68]`, exposing strong frozen-fold imbalance. In the 76+76 post-hoc replay, p16 learned support was selected on 48/76 positives and 48/76 negatives, but both-side reliable selections were only 6/76 positive and 12/76 negative versus frozen Q0 25/76. This closes B2 as `B2_FAIL_LISTWISE_NO_SUPPORT_GAIN`; no controller, StateMemory, threshold, backbone or sealed evaluation was run. The next registered evidence route after A2/B2 failure is the one-time A3 identity-good/semantic-bad diagnostic.
- A3 diagnostic first failed before execution because the script lacked the project-root import path when called directly (`ModuleNotFoundError: src`); no marker or scientific artifact was produced. The minimal `sys.path` repair is recorded and pushed before rerunning the same diagnostic path.
- A3 identity/semantic audit completed on causal prefix16: 1,194/1,298 frozen-R tracks mapped to native Q0, mean native appearance variance `0.1435` versus Q0 `0.2068`, mean native reconnected segments `3.018`, adjacent segment cosine `0.5584`, raw-to-native mean shift `0.1374`, and query positive-minus-hard-negative gap `0.02782` versus Q0 `0.02501`. This supports a viewpoint/reconnection and pooling-drift hypothesis but is descriptive only. The one authorized follow-up is a fixed M=3 contiguous causal multi-prototype representation with symmetric max prototype cosine; no training, controller, threshold or held selection is allowed.
- A3 fixed M=3 multi-prototype causal retrieval completed on the exact 984-query R universe. At p16, R@1/mAP changed `0.893219/0.848374`→`0.890469/0.850450`, hard-negative gap `0.189559`→`0.167946`, and unsafe flips were 6; only 1/4 folds were non-decreasing in both R@1 and mAP. The registered multi-prototype hypothesis therefore failed its safe-improvement diagnostic. B2 had high TRAIN validation top-20 candidate recall but poor event selection, so the next evidence route is the authorized B3 joint support-set matcher rather than another binary/router variant.
- B3 is registered after B2's high candidate-recall/low ranking result: retain the exact candidate-set listwise action space and DEFER, add only causal full-DINO candidate-to-history and candidate-to-set similarities, and train a compact listwise matcher on non-event public TRAIN groups. No controller, threshold, semantic ID, physical ID, future or sealed input is authorized; smoke/targeted/formal replay will be reported separately.
- Before B3 smoke, code review found two pre-execution implementation issues (global rather than per-row DINO normalization and potentially empty balanced-sampling pools). Both were corrected before any B3 marker or scientific output was created; the route contract and data remain unchanged.
- B4 native candidate-set build first failed before writing any artifact because the new builder referenced an omitted `norm` helper (`NameError`). No marker or partial data was created. The minimal helper repair is recorded; the same build path will be rerun once.
- B4 native runtime candidate manifest then completed from the frozen Q0 lineage: 13,631 image candidate sets, 464,146 bbox-bearing native rows, 6,077 TRAIN target-present groups and 7,554 explicit DEFER groups; fold fit/validation group counts are `[840/3122, 4296/597, 4767/63, 5121/43]`. Native candidate counts are now the same runtime units audited above; no event video enters fitting and all outputs are on `/data2`.
- B3 smoke (100 updates) and fold0 targeted (500 updates) completed with finite listwise outputs and explicit DEFER. Fold0 targeted validation candidate top-1 recall was `0.5381` (top-5 `0.9410`, top-20 `0.9988`) but candidate-or-DEFER accuracy was `0.2793`; p16 replay selected support on all 12 positive and 12 negative fold0 events, with 5/12 positive and 1/12 negative both-side reliable. This is not a gate and formal four-fold training remains required.
- B3 formal joint support matching completed all four folds at 1,000 updates. Validation candidate top-1 recall was `[0.5541,0.8696,0.5405,0.5714]`, but p16 event replay selected support on 36/76 positives and 33/76 negatives; both-side reliable support was only 9/76 positive and 4/76 negative. B3 therefore failed to improve the frozen 25/76 O reference and remains a diagnostic only. A short candidate-contract audit is registered to test whether public-row grouping matches the native Q0 runtime candidate set before any further route.
- The B3 candidate-contract audit found the decisive interface mismatch: the observability runtime's `q0_candidate_count` equals native Q0 candidate count on 14,691/14,691 rows, while the public-row groups used by B2/B3 match native counts on only 237/14,691 (1.61%; absolute count-difference median 19, mean −26.25). This is not a geometry or row-key error; it means B2/B3 trained/selected over the wrong candidate universe. A new, short B4 hypothesis is therefore registered: train one listwise matcher directly over native Q0 candidate sets, with TRAIN GT only for post-hoc IoU targets and no event-video fitting.
