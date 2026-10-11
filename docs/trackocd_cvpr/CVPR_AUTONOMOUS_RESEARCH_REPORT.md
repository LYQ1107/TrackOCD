# TrackOCD → CVPR autonomous research

## Scope and state

Client date: 2026-10-12, Asia/Shanghai. New research branch:
`research/cvpr-trackocd-autonomous-v2`, immutable parent
`7546527e5f9540d351785ade34c8e3292a74441d`.
The 1,715-line taskbook is bound by SHA256
`e38de91f68c9ce2aa0ada6b5ff0a6992d44ba789918728b7c298b1724903e0d9`.

Overall state: **IN_PROGRESS**, Goals A/B/C not achieved. A single negative
diagnostic does not end this research goal. Historical results are preserved,
not renamed as new CVPR experiments. No new training or inference yet.

## Registered execution sequence

CVPR-0 first: retrospective sealed FULL decision distributions for all three
seeds, four orders and five prefixes (60 executions at the previously frozen
coverage target 1.0). This is a finite substep, not the whole root-cause phase.
Val has already been repeatedly observed: these results cannot select new
architectures, checkpoints or operating thresholds. New mechanism selection
must use registered Train-development only.

Next CVPR-0 work: unchanged frozen-checkpoint Train-development replay with
action logits, Top-K candidate availability, action-only/candidate-only/first
bad-write upper bounds, explicit checkpoint gradient replay (not historical
training gradients), geometry and input-shift diagnostics. A lawful matched
Train predicted-track counterfactual is still to be established.

Then verify at least 12 related works and actual official model/loss/train/eval
code before bounded geometry, candidate and reversible-memory mechanisms.
New independent Train folds and locked evaluation must precede new training.
Dense causal-online evidence, independent novelty and multi-seed coverage-
matched gains remain pending; no CVPR readiness claim is made.

## Safety and budgets

Initial training ceiling: 600 cumulative GPU-hours, at most four GPUs.
CVPR-0 retrospective substep: one CPU worker, no GPU, no optimization,
1,800-second wall ceiling, 4 GiB RSS ceiling. Reserve at least 25% system RAM.
Preserve existing assets; retain incremental storage soft 15 GiB / hard 30 GiB
and at least 10 GiB free disk until a separately justified budget amendment.
Initial free disk approximately 62 GiB; this is not authorization to fill it.
No TAO Test, unauthorized holdout, Novel vocabulary, evaluator feedback into
live policy, future evidence, external process interference or destructive Git.

## Historical reference, not a new finding

Old FULL mean H and CT were zero, but the mean Old and New components include
different successful seeds. This does not establish a unique causal failure.
Old models used only top-1 candidates and irreversible prediction-owned memory;
their capped-prefix replay is not dense frame-online evidence. Historical
ledgers retain decisions but not logits, candidate ranks or gradient logs.
Unmatched predicted tracks are unknown, not proven background/pollution.

## Delivery

Each substage: tests, source/config and aggregate evidence, commit/push, fetch
and exact remote SHA verification. Raw trajectories, evaluator label joins,
checkpoints, data and weights remain local.

First new diagnostic completed:60historical FULL executions,18,273,660events,
75.66s CPU-only and847216640B peak RSS, with exact evaluator-count checks.
The p16seed-dependent action collapse is documented in
[CVPR0_ROOT_CAUSE_REPORT.md](CVPR0_ROOT_CAUSE_REPORT.md). This is not a new
method result or complete root-cause attribution. Train-only checkpoint replay
and geometry diagnostics now also completed:930replays/100geometry/18gradient
cases,243.03s CPU-only. Zero-bias WAIT collapse, selected-checkpoint gradient
antagonism and sparse-fold geometry loss are documented, with explicit oracle
and causality limits. Memory/known-count and real tied-ID controls are next;
matched lawful predicted Train input and CVPR-1 source review remain pending.
No training budget spent; no new successful method or CVPR readiness claimed.
