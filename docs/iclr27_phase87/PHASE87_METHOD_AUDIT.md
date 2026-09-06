# Phase87 method audit (read-only reuse of verified Phase86 sources)

This phase does not import a new backbone or weights. The Phase86 official
source audit was re-used so that the 10-hour window could test the registered
controller hypothesis rather than start another method lottery. Repository
heads and licenses below were recorded by the earlier audit on 2026-09-06;
the Phase87 code uses none of their text, category, or ID branches.

| method | official repository / inspected revision | license status | useful idea | Phase87 decision |
|---|---|---|---|---|
| OVTR (ICLR 2025) | `https://github.com/jinyanglii/OVTR`, local official commit `500e72c` | MIT (local asset audit) | persistent query / online track memory | physical stream remains Phase26/OVTR read-only; no new query training in C0 |
| TRACT (ICCV 2025) | `https://github.com/Nathan-Li123/TRACT`, `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` | no license declared in API metadata | trajectory aggregation | semantic/text OV-TAO objective is not the no-text controller contract; not imported |
| COVTrack (ICCV 2025) | `https://github.com/zekunqian/COVTrack`, `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b` | Apache-2.0 | proposal/association confidence | category-aware open-vocabulary branch is outside C0; not imported |
| ObjectRelator (ICCV 2025) | `https://github.com/insait-institute/ObjectRelator`, `59f79d5d0fa5cfc7169b6737fd414c25d1ed83a6` | Apache-2.0 | cross-view object relation | static cross-view interface has no causal physical lifecycle or prior-video state transaction; not imported |
| C3Po (NeurIPS 2025) | `https://github.com/c3po-correspondence/C3Po`, audited remote HEAD | NOASSERTION / not verified | point-map correspondence | static point-map inputs and uncertain license fail the causal row contract; not imported |
| AGE (ICLR 2026) | `https://github.com/Ashengl/AGE`, `d54c3147e61caf82a0de808333cebdc552fcf049` | MIT | open-world reject/expand | useful conceptual defer action, but no drop-in support memory; not imported |
| TALON (CVPR 2026) | `https://github.com/ynanwu/TALON`, `4091c2df89100974d316ce05659bc25654ff1e` | no license declared | trajectory-aware OV-MOT | text/open-vocabulary and task-specific supervision violate Phase87 inputs; not imported |

**Selection.** No audited 2025/2026 repository exposes a verified text-free,
causal prior-video support set with the existing 768-D row contract and
transactional semantic state. Therefore the only authorized Phase87 route is
a small, interpretable controller redesign on the frozen physical feature
stream. It is a controller/interface test, not a claim that any audited method
solves TrackOCD.
