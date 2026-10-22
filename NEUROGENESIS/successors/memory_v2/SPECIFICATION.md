# Bounded delayed-credit predictor memory (v2)

## Scientific contract

Existing online learners overwrite useful weights under recurring drift. We test a bounded bank of frozen predictors because accurate past predictors may substitute for some future weight updates once delayed feedback identifies their relevance. This is an empirical mechanism study, not a claim that predictor reuse or online gating is new.

The input is a finite feature vector x in R^d, augmented by a constant 1. The output is a scalar regression prediction. The model sees neither regime identity nor future/current labels. Before prediction t it processes only labels from steps s<t with availability s+delay<=t. Zero-delay feedback is processed after prediction, so it cannot affect itself. All methods share this ordering.

## Candidate

State consists of an active normalized least-mean-squares (NLMS) predictor w, up to K frozen weight vectors v_j, exponentially weighted observed losses L_j, feedback counts n_j, monotone expert identifiers, and a bounded pending prediction cache. Start with w=0 and an empty bank. Every S actual active updates, insert a copy of w and its unique identifier; evict the oldest stored predictor if capacity is exceeded. Frozen vectors never change after insertion.

At prediction time compute the active and every archived predictor's outputs. Choose the eligible archived expert with smallest loss (oldest ID breaks ties), provided n_j>=m and L_j < g L_active; otherwise choose active. Store all prediction-time outputs and selected ID with the pending feature vector. A new expert never receives credit for an example predicted before it existed.

On feedback y, update each surviving expert's loss with its cached prediction: L <- (1-alpha)L + alpha*(prediction-y)^2; the first eligible observation initializes L to its actual squared error. Update active loss with its cached output too. The gate is re-evaluated using only now-available feedback. If an archived expert is eligible and beats active, the candidate skips the active gradient; otherwise perform w <- w + eta*(y-w^T x)*x/(epsilon+||x||^2). The residual uses current weights on the now-available example; epsilon=1e-8. No stale gradient is silently substituted. All-update ablation always performs the gradient but has the identical bank, scoring, and prediction route. No-bank ablation is ordinary NLMS.

Each dot product, gradient, coordinate multiply, cache write, archive write, eviction, label, selected-memory prediction, and skipped update is counted. Algorithmic bytes count numerical state and pending records; serialized checkpoint bytes and process RSS are separate measurements. Equal storage caps are ceilings, not evidence of equal actual compute. Total-operation efficiency is a separate claim from fewer gradient updates.

## Controls

- Frozen zero predictor; delayed last-observation predictor.
- NLMS with eta in {0.1,0.5,1.0}; same legal features and current-residual semantics.
- Adam with eta in {0.001,0.01,0.05}, current-residual semantics, norm-clipped gradient at 10.
- Diagonal-free recursive least squares (RLS), forgetting factors {0.95,0.99,1.0}, ridge initial covariance 10 I.
- FIFO nearest-neighbor regressor with 32 examples and k=3, Euclidean distance on all features.
- FIFO replay NLMS, 32 examples, one extra oldest replay update per available label.
- All-update predictor-bank ablation; K=1 capacity ablation; uniform-bank prediction ablation. All receive identical pending-label handling.

## Mathematical limits

For one available example with squared error e^2 and augmented norm q, an unclipped NLMS update changes the same-example residual to e*(1-eta*q/(epsilon+q)). For eta in (0,2), its magnitude cannot increase. This is a one-example algebraic property, not a convergence or changing-distribution guarantee. Unit tests check the identity.

With K frozen experts, d coordinates and delay D, prediction needs O(Kd) arithmetic and state O(Kd+D(d+K)); loss scoring O(K) per delivered label. The gate can save gradients while increasing total work. Delayed labels impose an identification interval: before feedback distinguishes two regimes with identical feature distributions, no legal policy can reliably infer which target mapping applies. Retention does not remove that limitation.

## Experiment design before outcomes

Eight candidate configurations: eta in {0.5,1.0}, alpha in {0.05,0.2}, K in {4,8}; S=32, m=8, g=0.9. Development: five new seeds, base recurring and no-recurrence conditions; 960 predictions, eight features, four informative, delay 4, noise 0.05. One configuration is selected by mean MSE across both conditions, lexicographic name for ties. Conventional baseline is selected independently from the complete registered grid using the same endpoint and data. No second-version development unless an explicit falsifiable revision is recorded; two-version budget is an upper bound, not a requirement.

Confirmation: 20 fresh independent seeds for each of base recurring, no recurrence, delay 16, noise 0.2, distractor scale 6, 16 dimensions, and 1920-step horizon. All configurations frozen before generating any confirmation outcomes. Each seed is one independent experimental unit; time steps are not independent units. Primary: per-seed average of five recurrent conditions (base, delay, noise, distractor, dimensions), then paired candidate/baseline relative MSE. Horizon scaling and no-recurrence are secondary. Paired 95% bootstrap CI over seed-level relative differences (10,000 resamples, fixed analysis seed). Support requires the upper CI <=0.02 (2% noninferiority margin), mean update reduction >=20%, and no missing/failed primary cells. Accuracy superiority additionally requires upper CI < -0.05. These are scoped criteria, not universal guarantees.

Mechanism attribution additionally requires comparison against the all-update bank: same prediction/state family with suppression disabled. Report all-update comparison separately; do not infer causal update-saving benefits from a weak no-memory comparator alone. All other endpoint/condition analyses descriptive; no uncorrected multiple-test significance claims.

Real transfer: UCI Air Quality CO(GT) from sensor channels S1-S5, temperature, relative and absolute humidity. Exclude other ground-truth gases. Fixed physical scales (sensor/1000; T/50; RH/100; AH/2; target CO/10), not full-data normalization. Missing feature -200 -> zero with an indicator; missing targets remain in timeline but unscored/unupdated. First 40% of chronological rows development; final 60% protected transfer. Candidate and baseline chosen on synthetic development only; no real-data tuning. Artificial four-hour label delay; this is imposed feedback latency, not claimed measured label latency. Single real stream, descriptive temporal error and seven contiguous block summaries; blocks are not independent datasets.

## Failure and reproducibility

Every attempt retains config, source/protocol/data hashes, raw per-step predictions/errors/actions, final exact checkpoint, timing, counters, and a completed or failed receipt. No overwrite. Checkpoint includes pending prediction-time outputs and absolute clock; interruption/resumption must be identical. All comparisons require exact task/config coverage. Every paper number is generated from saved results. Source changes after confirmation freeze invalidate admission; preserve affected attempts and create a corrected lock only for documented validity defects.
