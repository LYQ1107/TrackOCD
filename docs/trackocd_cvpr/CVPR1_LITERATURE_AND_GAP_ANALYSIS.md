# CVPR-1 literature and gap analysis — partial, not stage acceptance

CVPR-0 remains ongoing. This is read-only source preparation while the registered
Train feature diagnostic runs; it does not replace the remaining root diagnostics,
the minimum twelve-work audit, two additional open-source works, independent folds,
three bounded mechanisms, or Goals A/B/C. No upstream experiment or weights downloaded.

## TALON — first pinned model/loss/training/evaluation source review

The official [CVPR 2026 paper record](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_TALON_Test-time_Adaptive_Learning_for_On-the-Fly_Category_Discovery_CVPR_2026_paper.html)
links [ynanwu/TALON](https://github.com/ynanwu/TALON). Read commit
`4091c2df89100974d316ce05659bc25654fcff1e`, obtained by actual Git HEAD/main query.
The complete recursive tree has no LICENSE file; permission to copy its code is
not assumed. No code was vendored, executed, installed, or trained. File-level Git
blob SHA1s and all requested audit fields are in
`outputs/trackocd_cvpr/cvpr1/talon_source_audit.json` (source hashes, not SHA256 claims).

The inspected [model](https://github.com/ynanwu/TALON/blob/4091c2df89100974d316ce05659bc25654fcff1e/methods/talon/model.py)
returns normalized image embeddings. Most backbone weights are frozen; the last
block is trainable, with additional CLIP final projection/norm parameters exposed.
Known offline training combines angular-margin classification and supervised
contrastive learning. The [CUB split implementation](https://github.com/ynanwu/TALON/blob/4091c2df89100974d316ce05659bc25654fcff1e/data/cub.py)
constructs labelled training from Known classes only; the remaining Known plus
Unknown images form the evaluation stream. This verifies that one implemented
dataset path, not every dataset or upstream checkpoint provenance.

In the inspected [trainer](https://github.com/ynanwu/TALON/blob/4091c2df89100974d316ce05659bc25654fcff1e/methods/talon/trainer.py),
Known NCM prototypes initialize a cosine-threshold discovery bank. New prototypes
are born below threshold. Own predicted groups drive norm-layer test adaptation
and confidence/support-weighted Known and anonymous EMA updates. Predictions use
whole-batch encoded features with sequential assignments, then batch-level updates.
Labels are collected for metrics, not passed into those updates. This is not our
dense, single-event, prefix-causal contract or evidence of reversible memory.

The Train-quantile threshold helper is computed in training, but the actual
evaluation call uses the supplied `self.tau`; do not assume automatic calibrated
deployment from the helper alone. The shuffled [test entry point](https://github.com/ynanwu/TALON/blob/4091c2df89100974d316ce05659bc25654fcff1e/test.py)
uses NCM reconstruction, non-strict checkpoint loading with warnings, and both
subset-wise/global Hungarian clustering metrics. Those mappings must not replace
TrackOCD's fixed Known predictions or persistent cross-video category accounting.

Transfer hypotheses, not innovations or reproduction claims: use Train-only Known
margin calibration and bounded prediction-owned support weighting as controls;
measure frozen versus adaptive geometry before any adaptation. Reject shuffled
image order as a causal video protocol, evaluation-GT remapping as a policy input,
and unbounded pseudo-label-driven writes as proven self-correction. The current
TrackOCD observations (mixed-state reuse and representation collapse) motivate
testing these hypotheses, but do not prove that TALON will repair them.

## Remaining source verification

LTC, the official Assignment-Driven Hash Learning repository (the supplied
Clarence-CV repository is a fork), Grounded Correspondence, VC-CGCD, SlotContrast,
TRACT, COVTrack, MASA, OVTR, discovery/memory decoupling, and additional relevant
open-source methods remain unaccepted until their actual source/venue/license
audits are recorded. README descriptions are not counted as completed code audits.

Architecture selection is pending CVPR-0 matched-reference and fixed-history
diagnostics and the full literature review; no new mechanism is selected here.
