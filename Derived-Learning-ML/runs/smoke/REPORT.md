# Generated experiment report

Study: `DLML-S1-SMOKE`. Manifest status: `COMPLETE`. All numbers below are generated from `summary.json`; no table value is hand-entered.

## Final

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 1/0 | 0.248185 | 0.248185 | 0 | not reached | 21216 |
| Analytic residual targets (primary) | 1/0 | 0.00513325 | 0.00743456 | 1 | 5440 | 21760 |
| No-derivation block-gradient ablation | 1/0 | 0.585052 | 0.634203 | 0 | not reached | 23184 |
| Fixed normalized-LMS rule | 1/0 | 0.14319 | 0.16258 | 0 | not reached | 21472 |

Paired tasks: 1; primary equal-resource wins: 1/1. Relative equal-resource improvement versus Adam: 0.970044. 

Predeclared quality criterion: `PASS`. Efficiency criterion: `FAIL`. Split interpretation: `NOT_SUPPORTED_ON_ADMITTED_MATRIX`.

## Long Horizon

| Condition | Tasks ok/failed | Terminal mean NMSE | Equal-resource mean NMSE | Threshold reached | Median work to threshold | Mean terminal work |
|---|---:|---:|---:|---:|---:|---:|
| Tuned Adam baseline | 1/0 | 0.0202316 | 0.0202316 | 1 | 26112 | 40800 |
| Analytic residual targets (primary) | 1/0 | 0.00812866 | 0.00911363 | 1 | 10880 | 43520 |
| No-derivation block-gradient ablation | 1/0 | 0.591941 | 0.625936 | 0 | not reached | 42504 |
| Fixed normalized-LMS rule | 1/0 | 0.0577265 | 0.0644392 | 1 | 37088 | 40992 |

Paired tasks: 1; primary equal-resource wins: 1/1. Relative equal-resource improvement versus Adam: 0.549535. 

Predeclared quality criterion: `PASS`. Efficiency criterion: `PASS`. Split interpretation: `SUPPORTED_ON_ADMITTED_MATRIX`.

## Locked selection

Final and long-horizon task answers were not used for selection. The locked parameters were:

- `adam`: `{"learning_rate": 0.05}`
- `analytic_residual`: `{"regularization": 0.1}`
- `block_gradient_no_derivation`: `{"learning_rate": 0.05}`

## Claim boundary

This is an internally executed synthetic study, not independent reproduction. Passing would support only the frozen grouped-additive matrix under the recorded CPU accounting model; it would not establish a general learning mechanism. A failed criterion is retained as negative evidence and is not repaired by changing the endpoint.
