# CVPR-1 literature and gap analysis — partial, not stage acceptance

CVPR-0 remains ongoing; its capped matched-reference/fixed-history diagnostics
have finished, while dense atomic policy/category-memory integration is pending.
This is pinned source preparation, not an upstream experiment or a new training
run. Independent folds, three bounded mechanisms and Goals A/B/C remain pending.
Historical entries below preserve their original partial-review boundaries.

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

## LTC — pinned active-path audit, Findings venue subsequently verified

The [author arXiv record](https://arxiv.org/abs/2603.13858) links
`brandinzhang/LTC`. Actual HEAD/main is
`44584bfddae5e6b82bbc182f68588a05e45365bb`; the complete tree has no LICENSE.
The initial Findings claim was unverified. The later official CVF accepted PDF
now independently binds CVPR2026 Findings and the author repository (see follow-up
below). No upstream code was copied or executed. Full model, generator, main,
train/eval, evaluation utilities and CUB dataset construction were read; file
blob SHA1s and audit fields are in `outputs/trackocd_cvpr/cvpr1/ltc_source_audit.json`.

The [paper methodology](https://arxiv.org/html/2603.13858v1) motivates synthetic
off-support examples from Known images, a dual similarity-margin objective, and
Train-only threshold adaptation. This suggests an open-set geometry hypothesis,
not evidence that synthetic off-support points capture true future categories.
Initial method-only scope was later extended to experiments, conclusion and
textual appendicesA-E, including explicit limitations. Not all figures were
visually inspected or accepted/arxiv versions compared line by line.

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

### SlotContrast — temporal instance consistency is not category persistence

[Official CVF proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html)
verified through existing proxy after browser403. The
[author paper](https://arxiv.org/html/2412.14295v2) links
[project page](https://slotcontrast.github.io/), whose Code link binds
[martius-lab/slotcontrast](https://github.com/martius-lab/slotcontrast).
Pinned main`55ec66dc02eeade630805789ef4a6c5df06f21ff`, MIT read; no code/weights
run or installed. Exact released-checkpoint binding and compatibility untested.

Actual configs/model/head, recurrent scan, initialization, SlotAttention,
frozen encoder, decoder, contrast/MSE, train and mask-evaluator paths reviewed.
Complete versus selected ranges are explicit in
`outputs/trackocd_cvpr/cvpr1/slotcontrast_source_audit.json`.
Image-only four-frame Train chunks optimize learned slots against detached
frozen DINO features. Adjacent same-slot positives compete against other
batch slots, not lawful same-category multi-positive supervision. Current
slot attention and recurrent prior-state updates are structurally causal;
whole-video mapped frame encoding alone is not evidence of future leakage.
No actual future-mutation certification or dense-stream adapter was run.

Slots reset per video, and whole-video ARI/best-mask-overlap assess instances,
not fixed Known IDs or cross-video category CT. Fixed slots, occlusion and
disappeared-slot behavior remain limitations. Use temporal consistency only
as a controlled hypothesis; do not substitute instance IDs for category tokens.
CVPR1 twelve-work/two-extra requirement remains pending.

### TRACT — official repository, incomplete core release

[Official ICCV2025 CVF abstract](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Attention_to_Trajectory_Trajectory-Aware_Open-Vocabulary_Tracking_ICCV_2025_paper.html)
explicitly binds [Nathan-Li123/TRACT](https://github.com/Nathan-Li123/TRACT).
Pinned commit`19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` has no `models` tree:
the imported track dataset, TraCLIP classifier/loss and MASA association
implementation are missing. Entry, Train/Val scripts and conversion/metric
utilities read, not a complete actual-model/loss review or runnable reproduction.
Root/TraCLIP license unestablished; no upstream code copied or run.

The [paper](https://arxiv.org/html/2503.08145v1) separates previous-trajectory
association history from final full-trajectory, category-name-assisted
classification. The latter uses Novel names/attributes and is outside our
vocabulary-free causal contract. Source converter broadcasts one final class
back to every detection, not a versioned prefix category event. No referenced
Test asset or class-name file opened and no source entry executed.

Missing classifier source means entry-passed evaluation labels are neither
proven harmless nor a demonstrated runtime leak. Retain a narrowly controlled
past-visible visual aggregation hypothesis only. This availability finding is
not counted as a complete source audit toward twelve-model/loss acceptance.
Evidence: `outputs/trackocd_cvpr/cvpr1/tract_source_availability_audit.json`.

### EAGC — active training-gradient path, not online memory

[Official CVF CVPR2026 proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_The_Devil_Is_in_Gradient_Entanglement_Energy-Aware_Gradient_Coordinator_for_CVPR_2026_paper.html)
and [author paper](https://arxiv.org/html/2604.14176v1) bind
[project](https://haiyangzheng.github.io/EAGC/) to
[HaiyangZheng/EAGC](https://github.com/HaiyangZheng/EAGC), pinned
`87af6ff66be58a0fef929c343639443d596d533d`. This is additional to the supplied
ten repositories. Actual SimGCD model/loss/hook, Train, CUB split/augmentation,
shell/config, evaluation and requirements read fully; file blob hashes are in
`outputs/trackocd_cvpr/cvpr1/eagc_source_audit.json`. No upstream execution.
No whole-project license in complete pinned tree; Apache backbone header alone
does not license all sources. Exact paper-checkpoint release unestablished.

Known-only frozen reference anchors labeled feature gradients; a soft Conceptor
modulates unlabeled gradients by relative Known energy. This is offline image
GCD, not category memory. Dependence on reference quality and added cost remains,
and dynamic/domain-shift generalization is unproven. Paper gradient diagnostics
must not be conflated with our selected-checkpoint CE/risk analysis.

Actual source facts: main adds a768D ReLU MLP; the DINOHead classifies normalized
768D input, independently of its256D contrast projection. Hooks change backbone
feature gradients, not head gradients or scalar printed loss. AGA is normalized
by labeled view count and feature dimension; EEP is clamped and uses a0.95
labeled-energy EMA. Frozen reference is called in train mode, but inspected
default DINO dropout/drop-path are zero; this alone does not demonstrate drift.

Unlabeled GT targets are masked out of direct supervised losses, but their All
ACC selects best checkpoints, and GT Novel cardinality sizes the default head.
`--estk` accepts an external count, not an active-path estimator. Hungarian
evaluation can remap Known identities. Those setup/selection rules cannot be
transferred to our protocol. Parser-default `v2p` has no evaluator registration;
the inspected CUB shell uses valid `v2`, so no shell-run failure is claimed.
Machine-local requirements are not installed. Consider reference/geometry
controls only after lawful independent folds and complete literature review;
no mechanism chosen, no training, no total CVPR1 acceptance here.

### AGE — explicit Gaussian update, still no reversible category lifecycle

[ICLR2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)
and its [accepted PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/ba9e3d60610f3525717665966d86e0cd-Paper-Conference.pdf)
bind [Ashengl/AGE](https://github.com/Ashengl/AGE), letter l (earlier OCR digit1
not accepted), pinned`d54c3147e61caf82a0de808333cebdc552fcf049`. MIT fully
read. Actual model/loss/Train/calibration/stream/update/CUBsplit/evaluation
source reviewed; evidence`outputs/trackocd_cvpr/cvpr1/age_source_audit.json`.
No source installed/executed, dataset or checkpoint downloaded.

Known-support-only training is followed by held-out Known calibration and
sequential density-based category discovery. Simple thresholding, covariance
quality and image-stream benchmark limits remain; no video CT certified.

Actual source: CE plus SupCon0.5 fine-tunes block11 and a256D BN/GELU head;
calibration fits thresholds/PCA42/covariance on held-out support only. Inliers
keep Known IDs, outliers create a new row when density argmax is Known or
update an existing discovered row. Query labels/mask feed metrics, not this
update. Whole-query frozen feature collection alone is not future leakage;
separate default-on oracle-K KMeans does not feed AGE updates. No mutation test.

Density exponentiation reintroduces underflow risk after computing log-PDF;
covariance/features have unresolved float64/float32 conventions. These are
static issues, not measured failures. New-cluster prior averages all current
covariances, not exactly fixed global covariance. Percluster window32 does not
bound total cluster count. Numeric rows append within one evaluation, with no
WAIT, quarantine, rollback or versioned split/merge. Every5epoch scheduler/saves
and fixed seed0 also require faithful reporting, not silent reproduction fixes.
Global Hungarian evaluation can remap Known identities. No best-by-GT selection
is implemented in this active path, unlike EAGC; no new method chosen here.

AGE is an additional MIT repository source review. EAGC is an additional
public-code review but has no whole-project license, so it is not certified as
the second licensed open-source extra. Twelve-work/two-extra acceptance remains
pending; availability findings cannot replace missing actual core audits.

### DebGCD — second MIT extra, source/paper gates distinguished

[Accepted ICLR2025 proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/789fc893d4b6e5b384f4eb058a35e31d-Paper-Conference.pdf)
links [project](https://visual-ai.github.io/debgcd/) whose Code link binds
[Visual-AI/DebGCD](https://github.com/Visual-AI/DebGCD), pinned
`9ec30fe3d54dfccd9934a862cf5cb9ff05de6f05`. MIT2025VisualAILab read.
Actual head/loss/Train/evaluation/DINOv1/CUBsplit/config/script paths reviewed;
file hashes in`outputs/trackocd_cvpr/cvpr1/debgcd_source_audit.json`.
No upstream code or asset executed, installed or downloaded.

The paper proposes separate distribution and auxiliary classifier branches for
offline image GCD. This is a representation hypothesis, not a persistent video
category memory. Actual source uses a5layer distribution MLP,2*Known OVA logits,
two768D normalized K-way classifiers and a separate256D contrast projection.
Masked Known labels supervise CE/SupCon/OVA, while detached predictions and OVA
certainty weight unlabelled hard-target learning. No direct unlabelled GT loss.

Paper task-consistency gate F(Eq9/11) is absent from the inspected auxiliary
loss path, which uses only confidence mask and certainty. Actual head also
computes all branches during evaluation; paper-only inference pruning is not
assumed. Source defaults/CUBshell differ explicitly. No silent source repairs
or claimed upstream replication. Wholequery Hungarian can remap Known; disjoint
test Old ACC selects best weights, and default head size uses GT totalK. These
selection/cardinality rules cannot transfer into our sealed protocol.

AGE and DebGCD now provide two additional MIT actual-source reviews beyond the
supplied ten; EAGC remains public-code-only with licensing unresolved. This
substep does not complete the full twelve-work audit, fair protocol or any
mechanism. Missing release/core sources remain explicitly partial.

### COVTrack — actual native path is not an anonymous causal category frontend

[Official ICCV2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Qian_COVTrack_Continuous_Open-Vocabulary_Tracking_via_Adaptive_Multi-Cue_Fusion_ICCV_2025_paper.pdf)
explicitly binds [author repository](https://github.com/zekunqian/COVTrack),
pinned `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`.
Root Apache2 license fully read. Actual C-TAO config/runner, detector/RoI,
fusion/association losses, current/past tracker and format/evaluation path
reviewed. Full-file versus selected line-range reads and blob hashes are in
`outputs/trackocd_cvpr/cvpr1/covtrack_source_audit.json`. No weights,
annotations, vocabulary lists or bytecode opened; no upstream code executed.

The paper's cue-reliability framework motivates a visual/geometry control,
not persistent category identity. Native source uses supplied vocabulary
before output: text embeddings, validation category subset and Novel-role blend
affect classification, dynamic thresholds and NMS. Removing output names does
not remove these dependencies. Per-video physical-instance IDs reset at frame0;
default formatting performs full-track majority-vote category rewriting.
This postprocessing is retrospective, not a causal category ledger.

Actual fusion uses residual appearance plus squared low-confidence scale,
not paper Eq5's convex blend. Actual training detaches inter-frame confidence
diagonals; cycle-MSE is disabled. Raw bbox coordinates enter the active fusion
path. Source/config/run/paper schedules differ; none is silently corrected.
Tracker's optional default-on host-specific GT observer remains inside forward;
logging indices do not select fusedfeature rows in inspected code, but no actual
GT/future/ID mutation certificate or legal supervision provenance is claimed.
Valid recovered shards remain protected assets, not accepted frontend evidence.

### DYDM — discovery/memory decoupling paper read, author source not verified

[Official CVPR2026 main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yu_Decouple_Your_Discovery_and_Memory_in_Continual_Generalized_Category_Discovery_CVPR_2026_paper.pdf)
and [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yu_Decouple_Your_Discovery_CVPR_2026_supplemental.pdf)
reviewed for method, experiments, Algorithm1 and implementation. The framework
separates trainable discovery from a frozen-basis analytical memory classifier,
fitted to newly predicted classes after each dataset stage. It is a useful
decoupling hypothesis, not an implemented TrackOCD method. Whole-stage training
and final refitting are not arrival-time causal discovery; global Hungarian ACC
is not fixedKnown categoryCT. No inspected reversible memory lifecycle.

Official page/selected paper text and targeted repository searches did not yield
a verified author core source. This is scoped search evidence, not proof that
no code exists. `outputs/trackocd_cvpr/cvpr1/dydm_paper_source_availability_audit.json`
therefore records **paper-only**, not an actual-code or licensed-extra count.
No third-party paper-note claims substitute for primary paper/source evidence.

### MASA — pinned actual instance model/train/loss/evaluation audit

[Official CVPR2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.pdf)
binds [author source](https://github.com/siyuanliii/masa/tree/c5472b9c7615f35abdf1188cb1a0c5408fe50d66).
Root Apache2 read;17 complete source/config/license files and3 installed
mmdet3.3 dependency files inspected, hashed and scoped in
`outputs/trackocd_cvpr/cvpr1/masa_source_audit.json`.
Main method/experiments/ablation/conclusion text read; figures/supplement and
exact local checkpoint release binding are not certified. No upstream code
executed or copied, new assets fetched, inference or training started.

Training uses SAM proposals from raw static SA1B images as instance-pair
supervision, not TAO category training. Given-box current-image RoI inference
has per-video physical-ID/EMA/TTL memory, not persistent semantic categories.
The R50 plug-and-play branch uses mmdet QuasiDenseTrackHead while released
GDINO training uses custom MasaTrackHead. Instance contrastive/distillation
losses may help physical continuity but same-category instances are distinct;
this does not by itself repair semantic representation collapse.

Actual tracker is score-ordered greedy bisoftmax/cosine association; stored
with_cats flag is not used in the inspected matching loop. Empty-frame early
return bypasses TTL update. Default evaluator's completed-track majority vote
rewrites historical categories and cannot stand in for an online ledger.
Given four-column boxes receive confidence1; anonymous-score legality requires
the separately frozen foreground contract, not this adapter alone.
Paper/config learning-rate and sampling differences are recorded, not repaired.
All are static observations, not local performance or mutation certificates.

### OVTR — native query-state/source audit, not anonymous discovery

[Official ICLR2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7cdbcf7f2037db2ccbfb9636db50399d-Paper-Conference.pdf)
binds [author source](https://github.com/jinyanglii/OVTR/tree/500e72c19bf5f7f8717546911a5639fdc26bfee5).
RootMIT and16 actual model/train/loss/data/eval file scopes reviewed, with
wholefile/selectedrange hashes in `outputs/trackocd_cvpr/cvpr1/ovtr_source_audit.json`.
Paper methods/experiments/conclusion and textual appendicesA-D reviewed;
accepted-versus-arxiv exact equivalence/local checkpoints not certified.

OFA/CTI dual branches and CIP carry physical query state across frames.
Supplied text already affects proposal ranking and previous decoder content;
OFA is not automatically vocabulary-free. Novel text/category-KL isolation,
source-domain filename-selected thresholds and completed-track majority votes
cannot be imported as anonymous causal category discovery. Native validation
loader includes GT labels, although inspected image-only inference does not
reference them; no actual mutation certificate is claimed.

Selected config disables encoder fusion; firstlayer content mask is commented
out in actual transformer. Main's truthy default eval list prevents its resume
optimizer/epoch guard from running under the inspected train script. Actual
alignment loss sums dimensions rather than paper n*d averaging. These are
static source/config distinctions, not executed failures or silently repaired
paper replications. Known-only supervision still needs processed-annotation
lineage: parser does not apply the stored rare-class ignore_ids.

Assignment-Driven Hash Learning's author-Code
link/release completeness, original-release/venue binding for Grounded
Correspondence, VC-CGCD readable VC module/paper-release binding,
SlotContrast exact checkpoint release/stream qualification,
TRACT missing core model/loss release, COVTrack lawful frontend/release binding,
MASA/OVTR exact local frontend/release qualification and DYDM author core source
open-source methods remain unaccepted until their actual source/venue/license
audits are recorded. README descriptions are not counted as completed code audits.

Matched-reference and fixed-history diagnostics are now completed within
their capped Train scope; they do not certify dense causal policy/memory.
Architecture selection still requires the full literature review and lawful
new folds; no new mechanism is selected here.

### VB-CGCD — covariance hypothesis, released incremental GT path rejected

[Accepted ICML2025 paper](https://proceedings.mlr.press/v267/dai25a.html)
explicitly binds [author source](https://github.com/daihao42/VB-CGCD/tree/7fe9501b425b5775588d0a06d60c817ee1f04663).
Nine whole active model/loss/train/data/eval files read and hashed in
`outputs/trackocd_cvpr/cvpr1/vb_cgcd_source_audit.json`.
Main method/experiments/ablation/discussion and selected appendixC.3/D.1-D.2
read; not all proofs, supplemental tables or figures. Complete tree has no
LICENSE/COPYING; source inspection does not grant copying/import permission.

Actual incremental entry uses true training labels to Hungarian-align GMM
clusters, then fits the classifier from those aligned labels. The second pass
filters predicted-new examples but retains incoming targets, rather than using
classifier reassignment as targets. Whole-session fitting and preallocated
oracle total class count cannot become causal unlabeled discovery by renaming.

The released SVI wrapper has an empty guide, deterministic class mean/covariance
parameters and only an observed Gaussian likelihood; no explicit latent prior
or KL was found on the active path. Neither matrix-normal/inverse-Wishart priors
nor covariance EMA should be attributed to this inspected implementation.
Paper contrastive fine-tuning/kmeans/fixed learning rate/stop criterion differ
from actual CE-only BOFT/GMM/scheduled optimization/determinant stopping.
Optimizer-state rollback is not reversible semantic membership.

Borrow only as a separately implemented, lawful small-sample dispersion control
for the observed representation/mixed-memory failures. No upstream execution,
dataset/weight access, Bayesian replication or scientific improvement claimed.

### HypCD — licensed geometry control, no online category lifecycle

[Accepted CVPR2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.pdf)
links the [author project](https://visual-ai.github.io/hypcd/) and
[pinned source](https://github.com/Visual-AI/HypCD/tree/ffc895b7ff38be8502bb7e3bd851f1d2a55dff73).
MIT license, Apache backbone header and14 whole relevant active-path files
reviewed in `outputs/trackocd_cvpr/cvpr1/hypcd_source_audit.json`.
Main method/protocol/experiments/hyperparameters/conclusion read; supplementary,
other algorithm paths and exact weight release not certified.

Selected HypSimGCD maps DINO CLS768 directly to a clipped Poincare ball and
Mobius classifier. The declared256 dimension does not build a projection MLP
in this entry. Known CE/SupCon, self-distillation/mean entropy and angle/distance
contrastive objectives train a fixed-K offline model; no WAIT, dynamic births,
versioned events or cross-video semantic membership. Image-only eval is a
structural observation, not a mutation certificate.

Best checkpoint selection uses disjoint Test old-ACC and mixed-global Hungarian
mapping, which cannot transfer to our lawful independentfold/fixedKnown protocol.
Paper versus script curvature/schedule/memax settings remain qualified, not
silently repaired. No datasets, weights, upstream imports or new training.
HypCD is another additional MIT source review; AGE and DebGCD already satisfy
the two licensed-extra source-inspection substep. That does not certify CVPR1
as complete: remaining exact release, causal/lineage qualification, cross-work
gap synthesis and actual downstream scientific acceptance are separate.

### Review coverage checkpoint — twelve scoped cores, not twelve replications

`outputs/trackocd_cvpr/cvpr1/review_coverage_index.json` distinguishes the
inspection substep from scientific acceptance. Twelve unique actual active
model/loss/train/eval path reviews now have relevant main-method/experiment
paper text; no README, missing core, unread bytecode or paper-only entry counted.
Selected ranges remain explicit, not whole-repository claims.

| Work/group | Scope counted | Transfer boundary retained |
| --- | --- | --- |
| TALON, LTC | 2 current-image OCD cores | Batch/order, mutable rows, metric remapping and no rollback |
| Grounded Correspondence, SlotContrast | 2 temporal object-centric cores | Physical slots and adjacent instance consistency != semantic category persistence |
| COVTrack, MASA, OVTR | 3 physical frontend cores | Vocabulary/supervision/release lineage and retrospective voting |
| EAGC, DebGCD, HypCD | 3 representation/gradient/distribution cores | Offline fixed-K, test-selected weights, no causal lifecycle |
| AGE, VB-CGCD | 2 Gaussian-state cores | Unbounded birth/no rollback or session-GT-aligned fitting |

AGE, DebGCD and HypCD are three additional MIT source inspections, surpassing
the minimum two licensed-extra inspection requirement. This permits neither
unlicensed-source copying nor automatic import or scientific acceptance.
AssignmentHash/PHE, VC-CGCD, TRACT and DYDM are excluded from actual-core counts.
BIA pinned tree contains only README, also availability-only.

TALON follow-up reads main experiments/ablation/hyperparameters/conclusion and
textual appendicesA-C, including author limits on backbone/GPU dependence and
large/nonstationary-stream instability. Its actual TTA SGD0.1(CLIP)/0.01(DINO)
differs from paper1e-4. Literal paper positive separation-distance term differs
from actual negative mean-distance term; no silent correction or runtime claim.
LTC follow-up reads experiments and textual appendicesA-E. Accepted
[Findings paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_Learning_through_Creation_A_Hash-Free_Framework_for_On-the-Fly_Category_Discovery_CVPRF_2026_paper.pdf)
firstpage independently verifies venue/repo. Author old/new tradeoff and possible
future text do not justify Novel-name input or image-generator equivalence on
cached track features. The source warmup1 agrees with paper appendixB.2; the
unused parser3 is a CLI/source distinction, not a paper warmup mismatch.

Cross-work root-supported mechanism synthesis, new lawful folds and actual
dense policy/memory mutation tests remain the next scientific gates.

Cross-work planning synthesis is now recorded in
[CVPR2_ROOT_SUPPORTED_MECHANISMS.md](CVPR2_ROOT_SUPPORTED_MECHANISMS.md):
three distinct geometry/candidate-decision/versioned-memory hypotheses with
finite fair controls, not three explored methods or scientific acceptance.
New folds, actual learned mechanisms, dense causality and primary frontend
binding remain open; twelve inspected cores are not twelve replications.
