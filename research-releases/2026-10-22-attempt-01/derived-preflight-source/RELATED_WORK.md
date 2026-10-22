# Related work and novelty boundary

This study sits at the intersection of target-based credit assignment and classical block optimization, but its mechanism should not be described as a new general target-propagation algorithm.

## Target propagation

[Difference Target Propagation](https://arxiv.org/abs/1412.7525) computes layer targets for deep networks using learned autoencoders and a correction for imperfect inverses. DLML-S1 also uses the language of a local target, but there is no backward network, learned inverse, stochastic unit, or deep credit-assignment claim. Its target is the exact partial residual implied by an additive model.

## Learned synthetic gradients

[Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343) trains models to predict future error gradients so modules can update without waiting for ordinary backpropagation. DLML-S1 has no learned controller or gradient predictor. Its target-generator training count is exactly zero, so a learned-controller ablation is not applicable to this protocol.

## Teacher distillation

[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) transfers information from teacher or ensemble outputs to a student. DLML-S1 makes no teacher calls and exposes no teacher logits. Calling the partial residual a distilled target would conflate distinct mechanisms.

## Block coordinate minimization and backfitting

Coordinate methods successively minimize along coordinate or block directions; useful overviews include Wright's [Coordinate Descent Algorithms](https://arxiv.org/abs/1502.04759) and Nesterov's [Efficiency of Coordinate Descent Methods on Huge-Scale Optimization Problems](https://doi.org/10.1137/100802001). Classical additive-model backfitting likewise updates one component against the residual after subtracting the others. DLML-S1's sequential ridge solution is best understood as a small, fully instrumented partial-residual/block-minimization instance. Its research value here is the controlled target-versus-gradient comparison and evidence package, not a claim that partial residuals themselves are novel.

## Conventional optimizer baseline

The baseline implements the moment estimates and bias correction of [Adam](https://arxiv.org/abs/1412.6980) in a dependency-free full-batch setting. Its learning rate is selected without final-task access. This is a credible conventional baseline for the bounded test, but the study does not compare every optimizer or closed-form global ridge regression.

## What remains unanswered

The experiment does not show that target derivation helps when the additive partition is wrong, when modules are nonlinear, when targets require a learned inverse, or when wall-clock/memory rather than deterministic arithmetic proxies dominate. The longer-horizon result already shows the final-family quality advantage does not persist at the larger budget. Those questions require versioned successor protocols; they cannot be inferred from S1.

