# DLML-S3-R0 frozen real-data GPU protocol

Status: **FROZEN BEFORE PROTECTED TEST EXECUTION**. This is a successor study;
it does not modify or reinterpret S1, S2, or S2.1.

## Question and falsifier

On the public Kaggle California Housing Prices dataset, does sequential
support-residual training of parameter-matched additive block MLPs improve
held-out RMSE by at least 2% over joint end-to-end Adam training at the same
number of example passes, with wins on at least four of five independently
seeded protected splits?

The hypothesis is rejected if either clause fails. Results are additionally
compared with full-feature ridge and histogram gradient boosting. Beating the
matched neural control does not imply state-of-the-art performance, novelty of
backfitting, or superiority over the strongest baseline.

## Data and splits

- Source: `camnugent/california-housing-prices` on Kaggle, `housing.csv`.
- License recorded by Kaggle: CC0 Public Domain.
- Target: `median_house_value`.
- Five seeds: 3101--3105.
- Per seed: deterministic 70% train, 15% development, 15% protected test.
- Hyperparameters are selected using train/development only. Test targets are
  read only after a winning candidate is locked for every method and seed.
- Numeric medians, means, and scales are fitted on the training partition.
  The declared categorical vocabulary is fixed in the protocol.

## Models and accounting

The derived condition is a sum of five small block MLPs over geography,
structure, demography, income, and ocean-proximity inputs. Each block is
updated against the exact residual after subtracting all other block outputs.
The matched control has the same modules and parameter count but trains all
parameters jointly with Adam. Both receive the same total number of training
example passes. Ridge and histogram gradient boosting are strong non-neural
baselines; ridge also exposes the classical linear ceiling.

For every attempt retain seed, split hashes, preprocessing state hash,
hyperparameters, parameter count, example passes, wall time, peak allocated GPU
memory, development score, terminal state hash, exception, and traceback.

## Resource and stopping rules

Execution requires a real CUDA device, at least 512 MiB available memory, and
no rising Linux `pswpout` during admission. A rising pageout count during the
matrix stops the run and preserves partial evidence. There are no adaptive
extra seeds, endpoint changes, or post-test retuning. Failed cells remain
failed. The run is terminal as SUPPORTED, MIXED, NEGATIVE, INCONCLUSIVE, or
INVALID; a positive outcome is not required for validity.

## Claim boundary

This is one real tabular dataset on Kaggle, not independent reproduction.
Publication readiness additionally requires artifact verification, paper
regeneration, citation review, and an independent clean-environment rerun.
