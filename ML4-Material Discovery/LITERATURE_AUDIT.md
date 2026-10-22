# Prior art and contribution boundary

Reviewed 2026-10-21 from primary publications and official documentation.

| Source | Established result or method | Consequence for this project |
|---|---|---|
| [Dunn et al., Matbench, 2020](https://doi.org/10.1038/s41524-020-00406-3) | Standardized materials-property benchmarks and reference pipeline | Official rows/folds are immutable; a group-mean successor must be reported separately. |
| [Zhuo et al., 2018](https://doi.org/10.1021/acs.jpclett.8b00124) | Experimental band-gap prediction from composition | Composition-only gap prediction itself is not novel. |
| [Goodall and Lee, Roost, 2020](https://www.nature.com/articles/s41467-020-19964-7) | Learned stoichiometric graph representations without crystal structures | No claim that a hand-built composition vector is a new representation or that structures are necessary for strong predictors. |
| [Wang et al., CrabNet, 2021](https://www.nature.com/articles/s41524-021-00545-1) | Attention-based composition property prediction | Neural composition baselines exist; absent matched implementations, no state-of-the-art claim. |
| [De Breuck et al., MODNet, 2021](https://doi.org/10.1088/1361-648X/ac1280) | Matbench evaluation of a descriptor-selection neural model, with explicit bias/imbalance and uncertainty analysis | MODNet is a strong small-data comparator; ridge and classical forests alone do not establish competitive completeness. |
| [Baird et al., SAASBO-CrabNet, 2022](https://doi.org/10.1016/j.commatsci.2022.111505) | Nested hyperparameter optimization of CrabNet on `matbench_expt_gap`; reported official-fold MAE 0.3310 eV | This is the most direct task-specific comparator located. The grouped v3 task is different, and the official neural result was not independently reproduced here. |
| [Li et al., MD-HIT, 2024](https://doi.org/10.1038/s41524-024-01426-z) | Redundancy control for composition and structure datasets; shows performance degradation as redundant neighbors are removed | Exact canonical grouping repairs duplicate leakage only. It does not establish extrapolation under broader chemical-distance control. |
| [Geurts et al., 2006](https://doi.org/10.1007/s10994-006-6226-1) | Extremely randomized trees | Core tree learner is established prior art. |
| [Meinshausen, 2006](https://www.jmlr.org/papers/v7/meinshausen06a.html) | Quantile regression forests estimate response distributions using within-leaf responses | A median of tree means is different and must not be called quantile regression forests or a consistent conditional-quantile estimator. |
| [Lei et al., 2018](https://doi.org/10.1080/01621459.2017.1307116) | Split conformal regression with exchangeability assumptions | Marginal coverage does not establish OOD or subgroup coverage. Score fitting and evaluation require separation. |
| [Romano et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | Conformalized quantile regression combines quantile estimation and calibration | Direct quantiles alone need empirical calibration; CQR is not a new contribution here. |
| [scikit-learn GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | Loss-dependent constant initialization, quantile and squared-error losses | Diagnose the zero-heavy quantile failure at initialization; do not infer lack of signal from a constant model. |

The five most important competitive or validity references for this study are
Zhuo's composition-only gap model, Roost, CrabNet (including its SAASBO-tuned
task result), MODNet, and MD-HIT's redundancy-controlled evaluation. They were
not all executed under a matched protocol; therefore this project cannot claim
to outperform them.

The defensible contribution is an auditable, resource-bounded empirical study
of robust tree aggregation, calibration, and retrospective ranking on a
zero-heavy experimental target. It includes exact matched-tree controls,
group-isolated repetitions, retained negative experiments, and explicit shift
failures. Neither median aggregation nor split conformal prediction is claimed
as a novel algorithm. A broad priority or exhaustive-literature claim is not
supported. Further architecture research requires its own mechanism and protocol.
