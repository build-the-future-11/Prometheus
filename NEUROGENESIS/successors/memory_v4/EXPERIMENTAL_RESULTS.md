# Experimental results ledger

Updated 2026-10-21. This ledger separates engineering verification, development selection, protected synthetic confirmation, and real-transfer evidence. It is not a substitute for the raw immutable attempt artifacts.

## Engineering and mathematical verification

- 26 unit tests pass.
- The fixed-ridge Cholesky implementation agrees with independent batch ridge solutions across 1,200 randomized cases and remains finite for 5,000 steps with duplicate/unexcited features.
- Delayed-feedback legality, periodic sufficient-statistic accounting, checkpoint restoration, data hashes, and chronological split contracts are tested.
- These checks establish implementation behavior, not predictive efficacy.

## Development and selection

- `runs/development-01`: resource admission refused before a model attempt; preserved.
- `runs/development-02`: 220/220 registered attempts completed, zero failures.
- Exact replay: VALID over 211,200 trace rows.
- Separate arithmetic/timing audit: VALID.
- Selected candidate: `bank_c0.1_a0.5_k8`.
- Selected conventional baseline: `rls_0.95`.
- Frozen lock: `runs/freeze.json`.
- Local pre-confirmation anchor: commit `dd71324e6e262ebd60bad5b57a7b3071802a7af5`, tree `aa31778fd1371a5d77c87916b5f01d243f44e114`, parent `b333c415ca9a50aa468b48ef7675b6984228e709`.

Development numbers are selection evidence and are not merged with confirmation estimates.

## Protected synthetic confirmation

- `runs/confirmation-01`: resource admission refused before a model attempt; preserved.
- `runs/confirmation-02`: the single registered campaign completed 1,400/1,400 attempts across 20 fresh seeds, seven conditions, and ten methods; zero failed cells.
- Exact replay: VALID over 1,536,000 trace rows with artifact-map SHA-256 `3a2380f272463d3c8ac26caeb1c4ebc960beff55f4b254936c5c66dcb1a39dea`.
- Separate arithmetic/timing audit: VALID; maximum streamed-versus-reference MSE summation difference `2.6645352591003757e-15`.
- Registered disposition: **SUPPORTED** within the synthetic scope.

Primary paired result over base, delay, noise, distractor, and dimension conditions:

- Mean candidate relative MSE change versus selected baseline: **-30.6247%**.
- Fixed 10,000-draw seed-bootstrap 95% interval: **[-33.9694%, -27.4379%]**.
- Mean candidate weight-solve reduction versus baseline: **58.8905%**.

Mechanism and simpler-explanation checks:

- Versus the always-update bank, mean relative MSE change is **-16.1625%**, interval **[-19.0633%, -13.2438%]**.
- Versus the every-third-label periodic solver, mean relative MSE change is **-32.8484%**, interval **[-36.1964%, -29.6279%]**. The candidate uses 23.5876% more solves than this periodic control, so update savings alone do not explain the accuracy result.
- Baseline and no-bank rows coincide, as specified, because both are the selected persistent-ridge RLS method.
- The novel-regime stress condition is unfavorable: pooled candidate MSE `0.3243036909` versus baseline `0.3233800979` (about **0.2856% worse**).
- The uniform-bank ablation is much worse than the selected gate on every reported synthetic condition, consistent with the gate being necessary for this implementation; this does not prove a universal causal mechanism.

## Real streams

- `runs/air-regression-01`: resource admission refused before a model attempt; preserved.
- Exposed Air Quality regression outcome: pending a stable host and fresh run ID.
- Protected Beijing PM2.5 transfer outcome: not run.

Until all registered real methods complete and pass exact replay/audit, no numerical robustness or real-transfer efficacy claim is admitted. Air Quality is already exposed regression evidence; only Beijing is the preselected protected chronological stream.

## Evidence boundary

Supported now: the frozen persistent-ridge memory candidate improves the registered synthetic error/weight-solve tradeoff under the declared generator family and fresh seeds.

Unsupported now: cross-dataset generalization, state of the art, independent external reproduction, hosted execution, hardware speed/energy improvement, production readiness, and real-stream efficacy.
