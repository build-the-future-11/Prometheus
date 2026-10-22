# Project charter

Status: **provisional, study S1 only**. No pre-existing charter, mechanism definition, code, queue, results, or prior Git history was present in this directory on 2026-10-19. This charter therefore records the fallback hypothesis required by the commissioning instruction; it does not claim prior approval.

## Provisional starting hypothesis

Learning targets or update rules derived from known task structure can improve held-out learning performance or training efficiency over conventional optimization.

S1 narrows that statement to a falsifiable mechanism: on additive grouped linear-regression tasks, analytically derived block-residual targets will improve held-out normalized mean-squared error (NMSE) at equal accounted arithmetic work and require less work to reach NMSE 0.08 than a development-selected Adam baseline.

## Primary mechanism

The model is additive: `f(x) = sum_g f_g(x_g)`. For block `g`, the method derives a local target using only support labels and current predictions:

`t_g = y_support - sum_(h != g) f_h(x_h)`.

It then performs a regularized least-squares solve for `f_g`, updating blocks sequentially. Inputs are support features, support labels, the declared feature partition, current block predictions, and a development-selected ridge constant. Outputs are new block weights and accounting counters. Assumptions are additivity, a correct feature partition, finite numeric inputs, and an invertible ridge-regularized block system. Per sweep cost is explicitly recorded in arithmetic work units and support-example reads.

Admissible information is limited to the public family configuration, meta-training/development support and evaluation data during selection, and final support data during locked fitting. Final/long-horizon evaluation targets are used only to score predeclared checkpoints. Hidden coefficient/component labels are never exposed.

## Mechanism taxonomy

- **Primary:** analytically derived targets. No target model is trained.
- **Learned target generator:** absent from S1; training steps are recorded as zero.
- **Teacher distillation:** absent from S1; teacher calls are recorded as zero.
- **Auxiliary component supervision:** absent from S1; latent coefficients are generator internals and never learner inputs.

These categories are not combined. Learned-controller and auxiliary-supervision ablations are therefore not applicable; the manifest records their zero use. The direct mechanism ablation removes the residual target and uses ordinary blockwise MSE gradients on the same block schedule.

## Claim boundary

S1 is a synthetic, internally executed mechanism test. It cannot establish broad superiority, real-world utility, publication readiness, or independent reproduction. A passed study admits only the frozen matrix in `PROTOCOL.md`; a failed study remains negative evidence.
