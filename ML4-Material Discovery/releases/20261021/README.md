# Verified development research release, 21 October 2026

- `paper.pdf`: complete 11-page manuscript, visually inspected.
- `arxiv-source-20261021.tar.gz`: main.tex, original arxiv.sty, bibliography,
  generated tables, PDF figures, and evidence map. Extract and compile main.tex.
- Wheel and source distribution: installable release software; dependencies are
  pinned for hosted reproduction in the repository requirements-hosted.txt.
- `robust-source.bundle`: the exact pre-execution Kaggle source transport, a tar
  archive with an opaque extension. Check its companion SHA-256 before extraction.
- `manifest.json`: file sizes and SHA-256 identities of the packages.

The repository includes raw predictions, splits, checkpoints, run manifests,
failed runs, tests, protocols, and numerical verifiers. The software distribution
alone deliberately omits the large raw evidence and checkpoints; reproduce from
the complete repository. The release code includes a post-execution lazy-import
optimization verified by exact installed-wheel inference checks; consult the
frozen bundle for the code actually executed on Kaggle.

The primary median-versus-mean aggregation endpoint is SUPPORTED; the broader
findings are MIXED and remain development evidence. Independent retraining and
external/prospective validation are absent. The paper author is Ryan Gomez,
with no affiliation listed, as specified by the user. This is a prepared source
package, not an arXiv submission or venue acceptance.
