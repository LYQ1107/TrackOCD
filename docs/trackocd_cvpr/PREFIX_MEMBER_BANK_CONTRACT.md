# Committed prefix member support bank

`PrefixMemberBank` replaces a trajectory's earlier contribution when its
committed prefix grows, so cumulative prefixes do not become extra member
votes. Exact input aliases can later split without changing earlier commits.
It is a Raw memory primitive verified with synthetic inputs, not a learned
CVPR-4 method or a repair of historical results. Real GT read authority remains
pending; this implementation requires no dataset, cache, weight or training.

## Why replacement is necessary

The unchanged `AtomicLegacyControl` calls old `CandidateMemory.apply()` once
for each observed group. For an existing state this appends the normalized
*whole cumulative prefix* and increments its count. A single track first
represented by `e0`, then `(e0+e1)/sqrt(2)`, produces two member votes: older
evidence is counted again and the prototype tilts toward the first prefix.
This is observable code behavior, not an attribution of all old failures.

New `PrefixMemberBank` instead keeps the latest committed Raw prefix for each
opaque route handle within its current video generation. Recommitting that
member replaces its contribution. Features are summed in canonical visible
content order, never physical-ID order. Exact same-frame full-prefix aliases
share one support group; subsequent divergence or partial advancement leaves
distinct current supports. Identical input observed at different times/videos
is not silently treated as the same witness.

| Event | Active support | Immutable commit history |
| --- | --- | --- |
| Same member, longer committed prefix | Replace old contribution | Append new commit; keep old |
| Two exact aliases in one current group | One group, two routed members | Both output bindings retained |
| Aliases later diverge | Split current supports | Initial shared decision unchanged |
| Policy changes a member's state token | Move current support, old token remains even if empty | Record previous/new tokens; never erase earlier mistake |
| WAIT / empty frame | Keep prior committed support | No fabricated commit |
| New video reuses a physical handle | New route scope, category support persists | Past video commits retained |

## Bounds and causal ambiguity

Only canonical observed-prefix arrays and externally predicted anonymous
bindings enter the bank; there is no category/role/purity argument. Raw evidence
is quality-weighted inside each prefix and normalized, then support groups
are equally weighted. This is a fixed engineering control, **not learned
reliability or a posterior**. Groups are not certified independent samples;
alias branches can share early observations. Do not call their count effective
sample size or infer cross-video semantic correctness from handle reuse.

All groups are validated/staged before one frame commit. Invalid bindings,
fingerprints, mutated/shrunk committed prefixes, noninteger/rebased future
elapsed values, duplicate/noncanonical groups and bounds refuse the frame.
No silent filtering, eviction or partial history. Resource limits cover
members, retained token bindings, commits, retained prefix payload and video
generations. Actual dataset sizing/stream integration must be registered
separately; prototype defaults do not establish full-benchmark scalability.

Opposing *distinct-member* evidence remains recorded even if its resultant
is zero. Such a state has no usable normalized prototype, rather than dropping
conflicting members or claiming purity. Empty/zero/defined are numerical
statuses, **not** Tentative/Confirmed/Quarantined semantic lifecycle labels.
Token allocation, candidate tie-equivariance, learned quarantine/revision,
latency/coverage-matched evaluation and dense online adaptation remain open.

Reassignment can improve an active prototype by removing old support, but
cannot undo sealed outputs or old contamination. Strict lineage-history
purity and current-version purity must both be evaluated externally after
decision sealing, under a pre-frozen CT/version/split/merge contract. This
component contains no GT and cannot certify either purity metric.

## Evidence and unresolved integration

Synthetic tests execute actual prefix replacement, old repeated-write bias
comparison, alias splitting/partial advancement, route renaming/reordering,
future mutation with sealed past preservation, reassignment/tombstone history,
WAIT, cross-video scope, antipodal contradiction, whole-frame rollback and
bounds. They do not run a learned scorer or a real-data benchmark.

The existing legacy controller, old checkpoints/ledgers/metrics and actual
cached diagnostic receipts are unchanged. Integration into a policy-owned
atomic scheduler remains required before any learned rollout or causal-bank
claim. Fresh lawful protocol, GT scope authorization, empirical R/C/M trials,
matched strong baselines and Goals A/B/C remain unmet.
