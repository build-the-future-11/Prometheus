# Generated experiment report

Study: `DLML-S2-P0`. Manifest status: `COMPLETE`. All numbers below are generated from `summary.json`; no table value is hand-entered.

## Final

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 5/0 | 0.96482 | 0.96482 | 0 | not reached | 150912 |
| Analytic residual targets (primary) | 5/0 | 0.958369 | 0.958392 | 0 | not reached | 158400 |
| Simultaneous residual-target ablation | 5/0 | 1.01227e+06 | 217977 | 0 | not reached | 158400 |
| No-derivation block-gradient ablation | 5/0 | 1.0001 | 1.0055 | 0 | not reached | 153960 |
| Fixed normalized-LMS rule | 5/0 | 0.959222 | 0.959309 | 0 | not reached | 154560 |

Paired tasks: 5; primary equal-resource wins: 3/5. Relative equal-resource improvement versus Adam: 0.0066619. 

Predeclared quality criterion: `FAIL`. Efficiency criterion: `FAIL`. Split interpretation: `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.

## Long Horizon

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 5/0 | 1.26797 | 1.26804 | 0 | not reached | 452736 |
| Analytic residual targets (primary) | 5/0 | 1.2637 | 1.2637 | 0 | not reached | 460800 |
| Simultaneous residual-target ablation | 5/0 | 2.64782e+23 | 4.52053e+22 | 0 | not reached | 460800 |
| No-derivation block-gradient ablation | 5/0 | 1.25344 | 1.25344 | 0 | not reached | 451616 |
| Fixed normalized-LMS rule | 5/0 | 1.26752 | 1.26749 | 0 | not reached | 453376 |

Paired tasks: 5; primary equal-resource wins: 4/5. Relative equal-resource improvement versus Adam: 0.00342453. 

Predeclared quality criterion: `FAIL`. Efficiency criterion: `FAIL`. Split interpretation: `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.

## Locked selection

Final and long-horizon task answers were not used for selection. The locked parameters were:

- `adam`: `{"learning_rate": 0.08}`
- `analytic_residual`: `{"regularization": 1.0}`
- `block_gradient_no_derivation`: `{"learning_rate": 0.08}`
- `block_jacobi_residual`: `{"regularization": 1.0}`

## Claim boundary

This is an internally executed synthetic study, not independent reproduction. Passing would support only the frozen grouped-additive matrix under the recorded CPU accounting model; it would not establish a general learning mechanism. A failed criterion is retained as negative evidence and is not repaired by changing the endpoint.
