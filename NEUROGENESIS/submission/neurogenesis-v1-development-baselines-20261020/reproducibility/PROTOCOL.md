# NEUROGENESIS protocol v1

## Scientific status

Protocol v1 is a shared-infrastructure protocol under `SCOPE_UNRESOLVED`. It admits conventional development baselines and rejects both candidate mechanisms. It also rejects the protected evaluation split.

## Adaptation problem

Each task is a 480-step sequential regression stream with eight features. Four features determine the target and four are distractors. The stream contains four equal regimes:

1. `A_initial`: initial linear mapping;
2. `B_conflict`: the informative mapping changes sign and distractor variance increases;
3. `A_return`: the first mapping returns while distractor variance remains shifted;
4. `C_novel`: a distinct informative mapping appears.

Targets include fixed Gaussian noise with standard deviation 0.05. Each target becomes available four steps after its prediction. The task generator is deterministic for a registered task identity and seed.

## Information available at time t

Before predicting at time `t`, a method receives the current features and state derived only from labels whose registered `label_available_at` is at most `t`. The current target and all future targets are unavailable. The runner scores the prediction before it enqueues the current label. Labels that fall beyond the horizon are retained as undelivered and do not update the model.

The trace records `updates_before_prediction` and `label_available_at` at every step. Tests lock the delay behavior: steps 0–3 have zero updates and step 4 may incorporate only step 0.

## Fixed identities and split boundary

- Meta-training registry: three `piecewise_linear_smooth` identities. No meta-training is run under unresolved scope.
- Development: five `delay_distractor_conflict_return` identities with seeds 101, 211, 307, 401, and 503.
- Evaluation: three committed identities in `configs/evaluation.json`. Their seeds are represented only by commitments. The split is protected, unadmitted, and has run count zero.

Identity validation rejects duplicate task IDs and repeated family/seed pairs across registered splits.

## Endpoints

The primary endpoint is **prequential mean squared error over all predictions**, computed from task targets outside the method. A future learned objective cannot redefine it.

Secondary endpoints are:

- mean MSE over the first 24 predictions after each regime switch;
- first-24-step MSE when regime A returns;
- return MSE divided by late-initial-A MSE;
- per-regime MSE;
- non-finite prediction count, maximum absolute prediction, and maximum update norm;
- wall time, algorithmic state bytes, gradient evaluations, and parameter updates.

Lower is better for all error endpoints. The retention ratio is descriptive and can be volatile when the late-initial denominator is small; raw return MSE remains the interpretable retention measure.

## Controls

- `no_adaptation`: frozen zero-initialized linear predictor.
- `sgd_lr_0p01`: ordinary delayed-label SGD, gradient norm clipped at 10.
- `sgd_lr_0p05`: a second fixed learning rate, otherwise identical.
- `mechanism_off_adam_lr_0p01`: ordinary delayed-label Adam and the reserved mechanism-off control.

All adaptive controls receive the same features, delayed labels, horizon, and prediction count. SGD and Adam use the prediction made when the labelled example first arrived; the delayed update does not recompute an error with later parameters. No baseline uses memory retrieval, writes, replay, consolidation, or future labels.

## Resource contract

Each run is capped at 480 predictions, 480 gradient evaluations, 480 parameter updates, zero retrievals, zero writes, zero consolidations, 131,072 algorithmic state bytes, and 60 seconds. The admitted matrix contains exactly 20 serial CPU runs and at most 9,600 predictions and 7,200 gradient/update operations. No paid compute or workers are admitted.

Actual adaptive runs perform 476 updates because the final four labels become available after the prediction horizon. Every attempt receives an immutable attempt directory containing its start record, result, per-step trace, and final checkpoint. Failures are retained and remain in aggregation denominators.

## Candidate-specific falsifiers

These falsifiers are dormant until exactly one line is owner-approved in protocol v2.

**Learned rule candidate.** Reject it if held-out external prequential MSE fails to improve over the development-selected tuned conventional optimizer under total accounting for meta-training, unrolling, auxiliary networks, inner adaptation, and inference. Gradient correctness, numerical stability, horizon transfer, parameter-scale transfer, checkpoint restoration, and distinct learned-loss/update-rule ablations are mandatory where applicable.

**Memory-weight candidate.** Reject it if held-out external prequential MSE fails to improve over information- and resource-matched frozen, bounded-cache, replay, and ordinary-update controls after charging storage bytes, indexing, retrievals, writes, eviction, consolidation, and parameter updates. Delayed-label legality, distractors, conflicting and returning regimes, capacity bounds, and reset correctness are mandatory.

Neither falsifier authorizes tuning against protected evaluation. Reduced gradient counts, favorable task ordering, or performance measured with a learned training objective cannot support a success claim.

