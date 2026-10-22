# Failure analysis

## Observed failure

DDFS underperformed constant fixed share, adaptive share, directed fixed share,
Hedge, and AdaHedge on the equal-series development endpoint. Every key factorial
contrast was unfavorable on all ten series.

## Mechanistic attribution

The $2\times2$ control design separates two additions to fixed share:

1. Adaptive share rate with a uniform target was worse than constant fixed share by
   +0.0021715770 MASE.
2. Directed targeting at a constant rate was worse by +0.0004485800 MASE.
3. Combining the two produced +0.0031951934 MASE degradation.

The expert bank itself is not an adequate explanation: Hedge and AdaHedge on the
identical experts were the two best aggregators. A plausible interpretation is that
the detector increases sharing when preserving concentration is valuable, while the
short-memory target further dilutes the strongest expert. The protocol did not
identify a general response curve for the detector, so this is a post-result
mechanistic hypothesis, not a proved causal law.

## Ruled-out implementation explanations

- All modes use identical expert predictions, expert losses, and detector state.
- Prediction is recorded before the current outcome enters any update.
- Exact start and end days and all 366 evaluation dates were enforced.
- The uploaded source matched the local 23-file manifest.
- Twenty-five hosted tests passed before evaluation.
- Independent metric recomputation found no discrepancy.

## Rejected repair strategies

Inverting the target, reducing drift gain, selecting a different detector, or tuning
against these ten series would define a data-informed successor. None can be used to
rewrite revision 2 or justify opening protected data. The negative result is retained.
