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

## LTC — pinned active-path audit, venue independently unverified

The [author arXiv record](https://arxiv.org/abs/2603.13858) links
`brandinzhang/LTC`. Actual HEAD/main is
`44584bfddae5e6b82bbc182f68588a05e45365bb`; the complete tree has no LICENSE.
CVPR2026 Findings is a repository claim, not yet an independently read proceedings
binding. No upstream code was copied or executed. Full model, generator, main,
train/eval, evaluation utilities and CUB dataset construction were read; file
blob SHA1s and audit fields are in `outputs/trackocd_cvpr/cvpr1/ltc_source_audit.json`.

The [paper methodology](https://arxiv.org/html/2603.13858v1) motivates synthetic
off-support examples from Known images, a dual similarity-margin objective, and
Train-only threshold adaptation. This suggests an open-set geometry hypothesis,
not evidence that synthetic off-support points capture true future categories.
Method sections were inspected; a full experiment/limitation review is pending.

Actual [model source](https://github.com/brandinzhang/LTC/blob/44584bfddae5e6b82bbc182f68588a05e45365bb/LTCmodel.py)
forces visual-only CLIP with the last visual block trainable; dormant mapper/text
code is not the executed default. [Training](https://github.com/brandinzhang/LTC/blob/44584bfddae5e6b82bbc182f68588a05e45365bb/train_eval.py)
uses CE, SupCon, Known/pseudo max-similarity hinges and a margin-adjusted Train
quantile/EMA threshold. [MKEE](https://github.com/brandinzhang/LTC/blob/44584bfddae5e6b82bbc182f68588a05e45365bb/mkee.py)
uses image mixup then one normalized input-gradient step of entropy minus RBF
feature density. The random permutation does not enforce different-class pairs,
unlike the paper's described cross-class initialization. The warmup is hardcoded
to1 despite a parser option; the actual default quantiles/midpoint differ from the
paper's illustrative values. These are source observations, not measured failures.

Sequential nearest-prototype evaluation periodically merges new prototypes via
mutual-nearest-neighbor DSU, compacts their numeric indices, and does not remap
previous emitted predictions. That is not immutable/versioned persistent category
identity. The [metric source](https://github.com/brandinzhang/LTC/blob/44584bfddae5e6b82bbc182f68588a05e45365bb/utils/evaluate_utils.py)
restricts both prediction rows and GT-index columns when `topK` is used, then uses
Hungarian mapping. Do not transplant this into full-denominator fixed-Known/CT
evaluation. The [entry point](https://github.com/brandinzhang/LTC/blob/44584bfddae5e6b82bbc182f68588a05e45365bb/main.py)
does not branch out of its training loop under `--eval`; it was not executed.

Borrow only the hypothesis of lawful Known-only open-set margin training and
Train-only calibration, with frozen Raw/no-generator controls. A cached-feature
TrackOCD variant would be our independently designed mechanism, not an official
LTC reproduction or borrowed-code innovation. New mechanisms remain unselected.

## Grounded Correspondence — pinned base active path, not category memory

The [author paper record](https://arxiv.org/abs/2605.03650) links its
[project page](https://magenta-sherbet-85b101.netlify.app/), whose read-only
response explicitly links the [repository](https://github.com/LiZhYun/ICML2026-RethinkingOCL).
Pinned actual HEAD`5d345268797425558b449337519af3ab24aeb6f1` includes later
extensions, so exact paper-release binding remains unverified. MIT license
read; no code executed/copied. Conference acceptance independently unverified.

Actual MOVi-D [configuration](https://github.com/LiZhYun/ICML2026-RethinkingOCL/blob/5d345268797425558b449337519af3ab24aeb6f1/configs/grounded_correspondence/movi_d.yaml)
trains video-only chunks against detached frozen feature targets. Current-frame
greedy feature seeds feed learned SlotAttention and an MLP mask decoder;
cosine Hungarian correspondence reorders slots and resets per video. Feature
MSE and adjacent-frame same-slot contrastive CE are active; category-supervised
multi-positive compatibility and immutable cross-video category memory are not.
Full file versus selected active-path read ranges are explicitly distinguished
in `outputs/trackocd_cvpr/cvpr1/grounded_correspondence_source_audit.json`.

The [paper](https://arxiv.org/html/2605.03650v1) evaluates instance segmentation,
not persistent semantic categories. Its stated limitations include occlusion
reappearance and fixed slot count; baseline batch size differs from its original
setting. No dense-causal mutation certification or TrackOCD gain is inferred.

Transfer hypothesis: first separate physical identity alignment from semantic
category compatibility, comparing frozen correspondence against learned dynamics.
Do not use video-reset slot IDs as category tokens or instance-negative contrast
as category grouping. This source review is preparation, not architecture selection
or acceptance of the twelve-work literature stage.

## Assignment-driven hash learning — source availability is not reproducibility

Read the [official CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Assignment-Driven_Hash_Learning_in_a_Hyper-Semantic_Space_for_On-the-Fly_Category_CVPR_2026_paper.pdf)
through the existing proxy after browser403, streamed text only, no storedPDF.
The supplied Clarence-CV repository is a fork of
[KeibingYang's repository](https://github.com/KeibingYang/Assignment-Driven-Hash-Learning-in-a-Hyper-Semantic-Space-for-OCD),
actual pinned main`22c94f4d66380256022b264a81383cc7edf6a2e7`.
The author PDF Code-link destination is not yet independently bound, so
official-code status stays unverified. MIT read; backbone Apache2.0 header
retained. No code copied, installed or run.

The paper studies Known-only hash geometry with augmented prototype pairs,
interpolated regions and soft assignments. Its image clustering evaluation is
not fixed Known recognition or persistent cross-video category memory.

Full actual training, hashhead, FPA, augmentation, CUB split and metric paths
were inspected. Pinned tree lacks imported PHE/dataset modules; training has
local-class-name shadowing and unresolved loss names; evaluator collects NumPy
arrays then calls torch.cat. FPA view-major features use sample-major repeated
labels, and top-k settings do not prune its inspected update. These static source
findings are not claimed as our measured execution failures. No speculative
patch or upstream reproduction. The inspected default semantic output is12D,
not a independently bound768D paper branch. File-level evidence and read ranges
are in `outputs/trackocd_cvpr/cvpr1/assignment_hash_source_audit.json`.

Use only the hypothesis of lawful multi-positive geometry controls, with Raw
preservation and no synthetic-negative baseline. Do not label sign-hash memory
as persistent semantics or substitute optimal-GT remapping for fixed Known IDs.
This incomplete release-binding audit does not select a new TrackOCD mechanism.

## Remaining source verification

### VC-CGCD — readable release and unlabeled protocol gaps

The [author paper](https://arxiv.org/abs/2607.04984) directly links
[VC-CGCD](https://github.com/Mrxjh105/VC-CGCD); pinned commit
`6a4e73d966d9e311ec0dfab8d31bf3c232926ef7`. ECCV2026 acceptance is an author
claim, not independently checked proceedings. MIT root and Apache2.0 backbone
header read. Actual model/head, loss, offline/session training, evaluation,
prototype and dataset paths inspected with explicit full/partial ranges in
`outputs/trackocd_cvpr/cvpr1/vc_cgcd_source_audit.json`. No upstream execution.

The [paper](https://arxiv.org/html/2607.04984v1) proposes ambiguity-buffered
virtual-category supervision and expanded-neighbor learning in unlabeled
image sessions, not persistent frame-causal track memory. Unknown category
count and noisy early predictions remain limitations.

The complete pinned tree lacks `models/vc.py`, retaining only CPython3.8
bytecode; no bytecode decoded or run. More importantly, the online entry passes
retained dataset labels into VC loss, which prefers valid true labels for
nonconfusing samples despite an all-zero labeled mask. Classifier size also
counts true novel labels. Best test accuracy chooses the checkpoint loaded by
the next session. These inspected paths cannot be adopted into our label-free,
unknown-K, locked-validation protocol. This is static evidence about this
commit, not an executed failure or a rebuttal of reported paper performance.

The inspected optimizer precedes construction of the extra VC-specific head;
its parameters are absent from that student parameter group. Missing VCHAP
source limits complete gradient-flow verification. Expanded-neighbor code
hardcodes256 and counts K*K rather than constructed K*expanded_K. Default
splits/sessions/learning rate/neighbors differ from the paper, so exact release
reproduction is not established. Optimal test-GT remapping includes Known IDs.

Borrow only candidate ambiguity buffering and lawful representation hypotheses;
do not transfer true labels/counts, test-selected weights, unread bytecode or
virtual training classes as persistent category tokens. CVPR1 remains PARTIAL.

Full LTC paper/venue verification, Assignment-Driven Hash Learning's author-Code
link/release completeness, original-release/venue binding for Grounded
Correspondence, VC-CGCD readable VC module/paper-release binding, SlotContrast,
TRACT, COVTrack, MASA, OVTR, discovery/memory decoupling, and additional relevant
open-source methods remain unaccepted until their actual source/venue/license
audits are recorded. README descriptions are not counted as completed code audits.

Matched-reference and fixed-history diagnostics are now completed within
their capped Train scope; they do not certify dense causal policy/memory.
Architecture selection still requires the full literature review and lawful
new folds; no new mechanism is selected here.
