# Train-first TrackOCD architecture

This is the independent user-authorized main protocol, not R2. Training uses
TAO Train inherited Known only. Historical M1 remains BLOCKED_FRONTEND_QUALITY;
Native MASA is limited-coverage/provenance-incomplete annotated-cadence replay.
Architecture implementation and budgets below are fixed before final heldout.
Engineering completion does not imply any scientific contribution is supported.

## Shared visual and category evidence

Frozen pinned DINOv2 ViT-B/14 produces768D normalized crop descriptors,stored
FP16. Crop/context/518bilinear normalization is identical across methods.
Only chronological first min(cap,available) observations are exposed; no
future-score selection,full-track mean,total duration,category/GT or text in
forward. Keys/video IDs route samples and constrain the external sampler only.

Existing SemanticAdapter is Linear768→512,GELU,Linear512→256,L2. Category-level
contrastive positives are same category across distinct videos/individuals;
negatives are other categories. Every anchor has both. Teacher Gram weights
0/1/5 are the finite development-only geometry control,not repeated R1 tuning.

A1 uses quality-weighted current-prefix mean. Existing A2 CategoryEvidence
computes per-observation261D cues:adapted256D vector,quality,causal consistency,
dispersion,log count and elapsed time.261→64→1 GELU yields sigmoid reliability;
positive quality-weighted evidence is normalized,with uncertainty and effective
maturity1/sum(normalized_weights²). Its extra16833parameters are matched by a
static261→32→256 residual plus scalar control,with no dynamic temporal weights.
Both have541889parameters,shared525056parameter adapter initialization,batches,
pairs,optimizer and1000-step budget. Matched evaluation uses selected A1 step.
No extra auxiliary reliability supervision in the main comparison.

## Prediction-owned category memory and decisions

Known prototypes are aggregates of the separate legal Train prototype videos.
Episode prototypes include only15fitting Known classes; pseudo categories have
no oracle prototypes. Final inherited78Known IDs have48Train prototypes and30
explicit gaps. No Val-GT supplementation. Anonymous state contains only visual
vector sums,member count,mean quality and dispersion/reliability. Tokens S:id
are monotonic and never recycled,including reset-per-video ablations.

Both D1/D2 use exactly17causal scalar cues and a17→32→4 GELU MLP708parameters.
These cues include top2 Known/anonymous similarities,margins,relative score,
track uncertainty/maturity/quality/time and predicted-memory reliability/
dispersion/count,presence masks. Only the best actual Known ID and best actual
existing anonymous token are candidates. A missed correct second-ranked
candidate is a real limitation,not replaced with a GT-selected candidate.

KNOWN assigns a legal Known ID and does not write anonymous memory. EXISTING
adds current evidence to the selected existing token. NEW creates a unique
token. WAIT writes nothing. All training/inference updates are actual predicted
actions; memory remains wrong/contaminated after errors. No GT repairs,
reclustering,evaluator Hungarian mapping or future evidence reaches live state.

D1 learns nominal action cross entropy. D2 adds expected asymmetric action
costs and delayed credit to a real previous predicted write when a later top
candidate encounters absorbing mixed-category history. GT category histories
are separate trainer-only target/risk bookkeeping,not model inputs or memory.
This is lightweight cost-sensitive learning,not a claim of reinforcement learning
or complete long-horizon optimal credit assignment. Risk constants and finite
development checkpoint/WAIT-bias choices are in persistent_policy_training.json.
Category-balanced loss weights limit dominance of the two largest Known classes.

## Causal evaluation and claim boundaries

Four frozen video orders and five independent observation-cap replays include
all short tracks. WAIT is censored/unresolved at that operating prefix,has a
positive observation-dependent training cost,and is not assigned zero latency.
This is not a verified dense frame-online retry controller. False Merge is
compared only when BOTH fixed Novel reuse-opportunity coverage and total
commitment coverage are within the registered0.05tolerance; incompatible points
are explicitly not matched,no favorable interpolation.

Audited Standard OCD uses exact Known IDs and one evaluator-only global Novel
Hungarian mapping. Persistent CT does not remap tokens:past mixed/unknown
membership cannot become certified pure through later updates. GT controlled
identity joins are separate from predicted geometry joins. Every actual
predicted MASA track,including unmatched/unknown,is required in the latter;
missing targets retain fixed4413Known/819Novel and527/529/547/531reuse denominators.
The physical HOTA/AssA/DetA/DetRe stay unchanged for all semantic postprocessors.

PHE is INCOMPARABLE under the current common-input/checkpoint/supervision
contract; no invented score. A2-vs-A1 and D2-vs-D1 require independent evidence;
if not supported,the final conclusion must retain scientific FAIL. Conditional
video bootstrap holds global mapping and stream state fixed and cannot establish
independent-video/category confidence with sparse pseudo category support.
