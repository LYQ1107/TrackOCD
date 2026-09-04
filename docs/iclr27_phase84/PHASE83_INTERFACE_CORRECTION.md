# Phase83 interface correction for Phase84

This note is a read-only correction record.  No Phase83 file or metric was
modified.

## A2 report source

The p16 table in `PHASE83_RESUMED_FINAL_REPORT.md` uses the partial
`metrics/physical_r_temporal.json` values (`temporal R@1=0.882735`) rather than
the actual A2 artifact `metrics/a2_temporal_r.json` (`R@1=0.880983`).  The A2
artifact also records only 23,341/43,423 public rows mapped to the native
stream and 74/76 event pairs.  Consequently that table is not evidence of a
full-coverage physical improvement.  Phase84 records both hashes and treats
the A2 result as incomplete.

## Physical-to-R semantics

The A2 replay changed appearance vectors for rows that could be joined to a
native stream, but it did not perform a causal reassociation of the complete
native Q0 candidate stream.  Physical membership therefore remained the
public-row membership for most of the R universe.  A true Phase84 physical
route must create canonical roots by causal unions and then map those roots
back to the frozen R rows without changing the 984-query denominator,
candidate order, or same-video exclusion.

## Support-route corrections

The Phase83 B2/B3 rankers were trained on public grouped candidate sets rather
than the native Q0 runtime candidate universe (237/14,691 group counts matched,
1.61%).  B4 used native sets but remained query-agnostic.  B5 mixed native and
public feature spaces.  Phase84 therefore treats those results as interface
diagnostics and registers a source-conditioned matcher with one shared,
corrected DINOv2 space and the full native candidate set.

