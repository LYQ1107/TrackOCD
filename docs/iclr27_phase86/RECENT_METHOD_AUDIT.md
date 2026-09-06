# Phase86 recent-method audit (U2 decision)

This is a read-only source audit after the corrected U1 deployment and U2-v1
TRAIN gate. Repositories were checked with `git ls-remote` on 2026-09-06 and
licenses were checked through the GitHub repository metadata API. No external
code or weights were imported.

| method | official source / inspected HEAD | license | relevant component | TrackOCD compatibility decision |
|---|---|---|---|---|
| AGE (ICLR 2026) | [Ashengl/AGE](https://github.com/Ashengl/AGE), `d54c3147e61caf82a0de808333cebdc552fcf049` | MIT | open-world category expansion/rejection | useful conceptual selective recognition, but not a causal physical-track correspondence or persistent Commit-CT interface; not imported |
| TALON (CVPR 2026) | [ynanwu/TALON](https://github.com/ynanwu/TALON), `4091c2df89100974d316ce05659bc25654ff1e` | no license declared in API metadata | trajectory-aware OV-MOT / tracklet semantics | trajectory signal is relevant, but text/open-vocabulary and task-specific training do not expose the legal Phase86 score-to-controller contract; not imported |
| LTC (CVPR Findings 2026) | [brandinzhang/LTC](https://github.com/brandinzhang/LTC), `44584bfddae5e6b82bbc182f68588a05e45365bb` | no license declared | open-world/continual tracking direction | no verified causal support-set deployment path; no code reuse |
| TRACT (ICCV 2025) | [Nathan-Li123/TRACT](https://github.com/Nathan-Li123/TRACT), `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` | no license declared | trajectory consistency and trajectory-aware open-vocabulary tracking | useful prior for trajectory aggregation, but its semantic/text branch and OV-TAO objective are not the no-text, prior-only support contract; not imported |
| COVTrack (ICCV 2025) | [zekunqian/COVTrack](https://github.com/zekunqian/COVTrack), `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b` | Apache-2.0 | open-vocabulary detection/association confidence | proposal/semantic cues are category-aware and not a legal replacement for the frozen causal relation stream; not imported |
| ObjectRelator (ICCV 2025) | [insait-institute/ObjectRelator](https://github.com/insait-institute/ObjectRelator), `59f79d5d0fa5cfc7169b6737fd414c25d1ed83a6` | Apache-2.0 | object correspondence across views | correspondence is relevant, but the released task is cross-view object relation, not online MOT lifecycle + prior-video support + persistent controller; no causal adapter was verified |
| C3Po (NeurIPS 2025) | [c3po-correspondence/C3Po](https://github.com/c3po-correspondence/C3Po), `HEAD` resolved on audit; API license `NOASSERTION` | not verified | cross-view/cross-modality pointmap correspondence | static pointmap input/output and unclear license do not meet the causal row-key/support contract; not imported |

Paper links used for the two closest verified recent directions are [TRACT ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Attention_to_Trajectory_Trajectory-Aware_Open-Vocabulary_Tracking_ICCV_2025_paper.html) and [OVTR ICLR 2025](https://arxiv.org/abs/2503.10616). Existing local OVTR/Phase85 assets remain the physical-MOT anchor; this audit does not replace them.

Conclusion: no audited 2025/2026 repository supplies a drop-in, text-free,
causal support-set selector with the frozen Phase85 feature schema. The U2-v1
negative result therefore closes only that implementation. A future route must
first repair the TRAIN match/defer imbalance or materialize a genuinely legal
support relation, not silently import a static correspondence or text branch.
