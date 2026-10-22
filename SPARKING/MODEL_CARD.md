# Model card: SPARKING v3 development candidate

## Intended use

SPARKING v3 is a controlled research study of composition-only experimental band-gap prediction under declared leave-one-element-out shifts. It tests whether target-independent element-frequency weighting improves a randomized tree ensemble. It is not a materials discovery, stability, synthesis, safety, or deployment system.

## Inputs and outputs

The input is a chemical formula. Parsing rejects invalid formulas rather than silently repairing them. The fixed representation concatenates the original SPARKING elemental-fraction and periodic-statistic features with Matminer Magpie `ElementProperty` and `Stoichiometry` descriptors. Missing feature values are imputed from the outer training partition only. The output is a nonnegative point estimate of experimental band gap in eV.

## V3 mechanism and controls

For each outer training partition, v3 counts how many training formulas contain each element. A composition receives the stoichiometric-fraction-weighted inverse occurrence frequency of its elements, raised to exponent alpha. The weights are normalized, clipped to [0.25, 4.0], and renormalized to mean one. They never use target values.

The production matrix used alpha 0.5 and 1.0, with alpha 0.0 as the exact unweighted control. Each setting trained three Extra Trees forests with 900 trees per seed and evaluated mean and median aggregation. Controls also included a three-seed GPU CatBoost ensemble and its single-seed member, squared-error histogram gradient boosting, and standardized ridge regression.

The development panel was O, S, In, Ge, Sb, Ba, Co, and F. It was chosen from target-independent occurrence counts before v3 outcomes. Cu, Ag, and B are prior observed development scenarios. Li, Zn, Ga, Se, and Te remain protected.

## Frozen selection and admission

The nonzero-exponent candidate with lowest element-macro MAE was selected. Candidates within 0.003 eV chose the smaller exponent and then mean aggregation. Protected confirmation required the selected candidate to improve macro MAE by more than 0.01 eV over its aggregation-matched alpha 0.0 control, improve at least five of eight elements, improve by more than 0.02 eV over CatBoost, and yield complete finite manifest-valid evidence with nondegenerate weights.

## Verified development result

The selected `extra_trees_alpha0.50_mean` model had macro MAE 0.709135 eV. Its matched unweighted control had 0.718258 eV, an improvement of 0.009123 eV. Five of eight elements improved. CatBoost had 0.750949 eV, an improvement of 0.041814 eV for the selected model. Evidence, completeness, finite-value, manifest, and weight checks passed.

The strict greater-than-0.010 eV matched-control requirement failed by 0.000877 eV. The scientific disposition is therefore `DEVELOPMENT_NOT_SUPPORTED`. No protected confirmatory or official-fold outcomes were observed, and no confirmatory run was launched.

The hosted test suite passed 21/21. The independent verifier reproduced 320 metrics from raw arrays. The post-hoc cluster bootstrap 95% interval for the matched-control improvement was [0.006650, 0.011586] eV, with bootstrap probability 0.24445 of exceeding 0.010 eV. This analysis was not preregistered and does not change the primary disposition.

## Limitations

The panel is a declared set of chemical shift scenarios, not a random sample of the periodic table. A formula can appear in more than one element holdout, so post-hoc row uncertainty was clustered by source row. Composition descriptors omit crystal structure, processing, measurement context, phase, and thermodynamic information. The ridge control failed catastrophically on Co with MAE 22.631720 eV; this retained failure demonstrates model instability under that holdout and is included in the macro result. Frequency weighting can only be interpreted within the clipped weighting rule and frozen compute budget.
