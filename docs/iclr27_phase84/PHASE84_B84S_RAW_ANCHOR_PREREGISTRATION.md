# Phase84 B84S-RA raw-anchor bounded-residual diagnostic

## Trigger

The repaired query-conditioned B84S-Q matcher used the correct source/query
contract and the exact native candidate sets, but its frozen event replay did
not preserve the strong same-space raw source-mean signal.  At prefix 16 it
selected reliable candidates on 7/76 positive events versus 20/76 for the
post-hoc raw source-mean diagnostic.  All native candidate sets were nonempty.

## Single registered test

Use the already frozen `b84sq_b84sq_formal_v3` checkpoints.  For each native
candidate set, let `r_i` be the source-mean cosine and `m_i` the frozen model's
candidate logit.  Standardize `m_i` within that candidate set and score

`s_i = r_i + 0.05 * tanh(z(m_i))`.

The bound `0.05` is fixed before event replay and is not selected from event
labels.  If the frozen matcher emits DEFER, the raw source-mean candidate is
used; an empty native set remains DEFER.  This is an inference-only diagnostic,
not a new model or threshold sweep.

## Boundaries and decision

The native Q0 stream, candidate order, prefixes, 76 positive/76 negative
events, causal timing, source cache, feature space and fold mapping are
unchanged.  Event GT is used only after choices are frozen to measure selected
IoU and negative activation.  No controller, StateMemory, semantic/physical
ID, category text, future row, DEV+, Q1, public-new or sealed label is read.

The route is retained only as evidence about raw-anchor preservation.  It
cannot by itself authorize alignment, controller or sealed evaluation.
