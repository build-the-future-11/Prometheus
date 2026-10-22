# Generated experiment report

Study: `DLML-S1-P0`. Manifest status: `COMPLETE`. All numbers below are generated from `summary.json`; no table value is hand-entered.

## Final

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 5/0 | 0.0226095 | 0.0241395 | 5 | 41184 | 101376 |
| Analytic residual targets (primary) | 5/0 | 0.00331227 | 0.00347502 | 5 | 10816 | 108160 |
| No-derivation block-gradient ablation | 5/0 | 0.167804 | 0.167804 | 1 | 92448 | 100152 |
| Fixed normalized-LMS rule | 5/0 | 0.0125522 | 0.0141041 | 5 | 54208 | 100672 |

Paired tasks: 5; primary equal-resource wins: 5/5. Relative equal-resource improvement versus Adam: 0.856044. 

Predeclared quality criterion: `PASS`. Efficiency criterion: `PASS`. Split interpretation: `SUPPORTED_ON_ADMITTED_MATRIX`.

## Long Horizon

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 3/0 | 0.00471137 | 0.00470095 | 3 | 41184 | 300960 |
| Analytic residual targets (primary) | 3/0 | 0.00475254 | 0.0047539 | 3 | 10816 | 302848 |
| No-derivation block-gradient ablation | 3/0 | 0.0444868 | 0.0444868 | 3 | 215712 | 300456 |
| Fixed normalized-LMS rule | 3/0 | 0.00489357 | 0.00491076 | 3 | 65824 | 302016 |

Paired tasks: 3; primary equal-resource wins: 1/3. Relative equal-resource improvement versus Adam: -0.0112646. 

Predeclared quality criterion: `FAIL`. Efficiency criterion: `PASS`. Split interpretation: `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.

## Locked selection

Final and long-horizon task answers were not used for selection. The locked parameters were:

- `adam`: `{"learning_rate": 0.08}`
- `analytic_residual`: `{"regularization": 0.1}`
- `block_gradient_no_derivation`: `{"learning_rate": 0.08}`

## Claim boundary

This is an internally executed synthetic study, not independent reproduction. Passing would support only the frozen grouped-additive matrix under the recorded CPU accounting model; it would not establish a general learning mechanism. A failed criterion is retained as negative evidence and is not repaired by changing the endpoint.
