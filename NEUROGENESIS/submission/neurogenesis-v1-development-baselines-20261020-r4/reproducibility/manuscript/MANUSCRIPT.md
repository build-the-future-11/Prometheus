# A fail-closed benchmark foundation for an unresolved adaptive-learning mechanism

## Abstract

NEUROGENESIS was opened with conflicting historical descriptions: learned objectives or update rules, and bounded memory that substitutes for some parameter updates. No qualifying local owner decision, charter, paper draft, codebase, or experiment manifest established which description was authoritative. We therefore implemented the owner-directed fallback: a mechanism-neutral sequential benchmark, conventional controls, isolated candidate interfaces, strict information timing, total resource accounting, and protected evaluation gates. A five-seed development study compared no adaptation, two delayed-label SGD configurations, and delayed-label Adam. SGD at learning rate 0.01 reduced overall prequential mean squared error from 1.0110 to 0.9484 but worsened post-switch and returning-regime error. SGD at 0.05 degraded mean squared error to 3.3480. These findings establish baseline behavior and harness correctness; they do not test either candidate mechanism.

## 1. Motivation and scope

Adaptive learners can improve performance by changing parameters, consulting bounded state, or learning the adaptation rule itself. Those mechanisms have different information sets, costs, and falsifiers. Treating them as interchangeable would make negative and positive outcomes uninterpretable. The local identity audit found no authoritative mechanism decision, so protocol v1 forbids candidate execution. It preserves both candidates as separate interfaces and evaluates only shared conventional controls.

## 2. Benchmark

Each registered task is a 480-step prequential regression stream. Four informative features determine the target and four distractor features undergo variance shift. The stream moves through an initial linear regime, a sign-conflicting regime, a return to the first regime, and a novel regime. Labels arrive four steps after prediction. A method can update only when the registered availability time is reached.

Task identities are separated into meta-training, development, and protected evaluation registries. Protocol v1 uses five fixed development seeds. Meta-training is unused because no learned mechanism is admitted. Protected evaluation has committed identities and zero runs.

## 3. Methods and accounting

The primary endpoint is mean squared error over all prequential predictions. It is computed from external task targets and cannot be changed by a future learned objective. Secondary endpoints measure early post-switch error, returning-regime error, non-finite behavior, maximum prediction and update magnitude, wall time, state bytes, gradient evaluations, and parameter updates.

The controls are a frozen zero-initialized linear predictor, SGD at learning rates 0.01 and 0.05, and Adam at learning rate 0.01. Gradients are norm-clipped at 10. Delayed updates use the prediction made for the labelled example when it first appeared. The final four labels arrive beyond the horizon, so each adaptive run performs 476 updates for 480 predictions.

Every attempt is retained with a start record, result, 480-row trace, and checkpoint. Runs execute serially on CPU. The independent verifier binds protocol, config, and producer source hashes and checks every result and trace hash.

## 4. Development results

All 20 attempts completed across five seeds. SGD 0.01 achieved the lowest mean overall MSE, 0.9484 with sample standard deviation 0.0627, compared with 1.0110 ± 0.0671 for no adaptation. The paired mean difference was −0.0626. However, its post-switch MSE was 1.9652 versus 0.9611 for the frozen control, and its return MSE was 2.2782 versus 1.2241.

Adam 0.01 reached 0.9978 ± 0.0943 overall MSE, a paired mean difference of −0.0132 from no adaptation, while also worsening switch and return error. SGD 0.05 reached 3.3480 ± 0.2161 and had return MSE 12.2006. No policy produced a non-finite prediction, but SGD 0.05 reached a maximum absolute prediction of 11.2766 and is a retained negative development configuration.

## 5. Interpretation

Overall prequential error and rapid adaptation do not rank these controls the same way. SGD 0.01 learns within stable stretches but enters new and returning regimes with stale parameters, producing a lower average error alongside larger transients and poorer retention. The high-rate SGD result shows that successful execution and finite arithmetic are insufficient evidence of a useful adaptation policy.

No result favors learned rules over memory-weight adaptation. Neither was implemented. The evidence supports only the benchmark, admission boundaries, and conventional baseline characterization.

## 6. Limitations and next decision

The task family is synthetic, linear, small, and development-only. Five seeds are enough to expose consistent baseline tradeoffs but not to establish broad generalization. The evaluation identities remain protected, and no independent reproduction has occurred.

A dated local owner decision must select exactly one candidate mechanism. A learned-rule successor must freeze meta-training tasks, unrolling and auxiliary costs, tuned optimizer controls, transfer tests, and ablations. A memory-weight successor must freeze capacity, retrieval, write, eviction and consolidation rules, cache/replay controls, and delayed-label semantics. That successor must retain protocol v1 and its negative baseline as historical evidence.

## Evidence availability

The full development bundle, including all configurations, seeds, traces, checkpoints, aggregate tables, and verification receipt, is stored under `results/development/shared-baselines-v1-20261019`. Reproduction commands are in `REPRODUCE.md`.

## 7. Derived analysis and paper artifacts

The retained record has been reanalyzed without modifying the source bundle. `scripts/analyze_bundle.py` produces hash-bound JSON, CSV, LaTeX, an offline report, policy-mean and individual-attempt rolling-MSE SVGs, and a stability--plasticity SVG. The accepted derived-analysis receipt is `analysis_receipts/development-baselines-v1-20261020-r2-verification.json`; it regenerated all seven outputs from the raw bundle with status `VALID` and input-artifact-map SHA-256 `b39fa49ab15dea6072ecd8fe0de36bca2e4ac71b8b6a598db8075ccc277b4267`.

`scripts/assemble_submission.py` packages the verified v1 baseline report as LaTeX source, generated tables/figures, raw artifacts, configs, source, and reproduction scripts. This package is a reviewable benchmark/baseline report, not a submission-ready mechanism-efficacy manuscript. Its explicit readiness decision remains `NOT SUBMISSION READY` until a separately approved candidate protocol is executed through the required development and protected-evaluation gates.

