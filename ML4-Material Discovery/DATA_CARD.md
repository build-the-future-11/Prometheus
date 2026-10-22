# Data card — Matbench experimental band gap

## Identity

- Dataset: `matbench_expt_gap`
- Version: Matbench v0.1
- Size: 4,604 rows; 4,601 canonical compositions after local equivalence parsing
- Input: composition string
- Target: experimentally measured band gap (`gap expt`), eV
- Local ID: `mb-expt-gap-{one-based row:04d}`
- Download: `https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz`
- Pinned artifact SHA-256: `783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`

The upstream metadata traces the measurements to Zhuo, Mansouri Tehrani, and Brgoch (2018), with Matbench curation described by Dunn et al. (2020). Exact citations are embedded in `data/raw/matminer_dataset_metadata.json`.

## License

The upstream Figshare source-data landing page declares MIT. The downloaded Materials Project mirror does not embed a machine-readable license. This repository records that limitation instead of inferring broader rights. Code in this subproject is separately MIT-licensed.

## Curation and exclusions

Upstream metadata says records were deduplicated by composition; compositions with reported gaps spanning more than 0.1 eV were removed, and remaining repeats used the observed value closest to their mean. Local canonicalization nevertheless found three equivalent-composition pairs with spans from 0.58 to 2.30 eV. They are preserved to avoid rewriting the benchmark and are explicitly audited.

Missing formulas, missing/non-finite targets, negative targets, malformed formulas, hash drift, or split drift fail closed. No local outlier exclusion is performed after inspecting labels.

## Leakage and split use

The exact official five-fold IDs are used only for the benchmark-comparable study. Two canonical groups cross official train/test boundaries in four folds. Custom calibration and element-holdout splits isolate canonical groups. No custom result is labeled an official Matbench result.

## Representation limits

Composition does not identify crystal structure, phase, polymorph, defects, temperature, measurement method, uncertainty, sample quality, or provenance beyond the curated source. Near-identical structures and polymorph leakage cannot be tested because structures are not present. The same formula may correspond to materially different specimens or phases.

## Intended and prohibited uses

Intended: software testing, retrospective composition-property benchmarking, uncertainty-method development, and fixed-pool policy evaluation.

Prohibited claim: treating a prediction or historical label as proof of stability, synthesizability, safety, device suitability, novelty, or independent replication.

