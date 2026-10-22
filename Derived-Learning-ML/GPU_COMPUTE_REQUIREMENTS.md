# GPU compute status

S1, S2, and S2.1 are dependency-free CPU studies and have been executed
locally. The separately frozen successor `DLML-S3-R0` is implemented and does
require a real CUDA device. Its private Kaggle job is the only admitted GPU
execution in this repository.

S3 requires at least 512 MiB available memory and no rising Linux `pswpout`
during admission. It uses five fixed seeds, a frozen candidate grid, equal
example passes for the two neural conditions, and retains every candidate,
failed test cell, exception, traceback, runtime, and peak CUDA allocation. It
must run with `GPU T4 x2`, internet disabled, and exactly the frozen private
source plus `camnugent/california-housing-prices` attached.

The local source archive and post-run verifier are clean-room tested. The
corrected source artifact must replace the older Kaggle dataset version before
submission; `HOSTED_EXECUTION_STATUS.md` records that remote handoff gate. A
queued notebook, passing unit suite, or complete archive is not a scientific
result until the downloaded archive, sidecar, manifest, raw records, failure
denominator, and classification pass independent verification.

The learned-controller checklist item remains unimplemented and is not part of
S3. No GPU result may be used to imply that a learned target controller exists.
