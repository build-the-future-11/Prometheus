"""Import compatibility for the hash-bound S3 test suite.

The archived test loads run_s3_gpu_study with exec_module without registering
its module. Dataclass annotation resolution needs that module namespace.
Register the unchanged script normally before pytest collects the frozen test;
do not edit or rebuild the archived test, runner, configuration, or release.
Importing this script does not invoke its guarded experiment entry point.
"""

import importlib.util
import sys
from pathlib import Path


_NAME = "run_s3_gpu_study"
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / f"{_NAME}.py"
_existing = sys.modules.get(_NAME)

if _existing is not None:
    if Path(_existing.__file__).resolve() != _SCRIPT:
        raise RuntimeError(f"{_NAME} is already loaded from a different source")
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _SCRIPT)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load the frozen S3 helper: {_SCRIPT}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    try:
        _spec.loader.exec_module(_module)
    except BaseException:
        if sys.modules.get(_NAME) is _module:
            del sys.modules[_NAME]
        raise
