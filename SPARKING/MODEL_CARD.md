# Model card: SPARKING v3 development candidate

## Intended use

SPARKING v3 is a controlled research study of composition-only experimental band-gap prediction under declared leave-one-element-out shifts. It tests whether target-independent element-frequency weighting improves a randomized tree ensemble. It is not a materials discovery, stability, synthesis, safety, or deployment system.

## Inputs and outputs

The input is a chemical formula. Parsing rejects invalid formulas rather than silently repairing them. The fixed representation concatenates the original SPARKING elemental-fraction and periodic-statistic features with Matminer Magpie `ElementProperty` and `Stoichiometry` descriptors. Missing feature values are imputed from the outer training partition only. The output is a nonnegative point estimate of experimental band gap in eV.

## V3 mechanism

For each outer training partition, v3 counts how many training formulas contain each element. A composition receives the stoichiometric-fraction-weighted inverse occurrence frequency of its elements, raised to exponent alpha. The weights are normalized, clipped to [0.25, 4.0], and renormalized to mean one. They never use target values.

The production candidate matrix uses alpha 0.5 and 1.0. Alpha 0.0 is an exact unweighted control. Each setting trains three Extra Trees forests with 900 trees per seed and evaluates both mean and median aggregation across seeds. Every candidate uses the same rows, features, seeds, tree count, and hyperparameters as the matched alpha 0.0 control.

## Controls

- Identical unweighted Extra Trees with mean and median aggregation.
- Three-seed GPU CatBoost ensemble and its single-seed member.
- Squared-error histogram gradient boosting.
- Standardized ridge regression.

The development panel is O, S, In, Ge, Sb, Ba, Co, and F. It was chosen from target-independent occurrence counts before v3 outcomes. Cu, Ag, and B are prior observed development scenarios and are not reused as fresh evidence. Li, Zn, Ga, Se, and Te remain protected.

## Selection and admission

The nonzero-exponent candidate with lowest element-macro MAE is selected. Candidates within 0.003 eV choose the smaller exponent and then mean aggregation. Protected confirmation is permitted only if the selected candidate:

- improves macro MAE by more than 0.01 eV over its aggregation-matched alpha 0.0 control;
- improves at least five of eight development elements over that control;
- improves macro MAE by more than 0.02 eV over the CatBoost ensemble; and
- yields complete, finite, manifest-valid evidence with nondegenerate weights.

## Limitations

The panel is a declared set of chemical shift scenarios, not a random sample of the periodic table. A formula can appear in more than one element holdout, so row-level bootstrap analyses must cluster by source row. Composition descriptors omit crystal structure, processing, measurement context, phase, and thermodynamic information. Frequency weighting may reduce effective sample size and can only be interpreted within the clipped weighting rule and frozen model budget.

## Current status

V3 is `FROZEN_RUN_READY_NOT_EVALUATED`. The local suite passes 21 tests and a bounded CPU integration run through the production evaluator passes. No v3 performance claim exists until the private Kaggle GPU run completes and its raw arrays, manifest, environment, and independent verification are recovered.

V2 remains a preserved valid negative development study: its selected risk gate worsened macro MAE by 0.000130 eV versus CatBoost and did not qualify for protected confirmation. The protected Li/Zn/Ga/Se/Te outcomes and official Matbench folds remain unopened by both v2 and v3.
