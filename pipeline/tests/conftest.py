"""
Shared test fixtures/helpers.

Most pipeline scripts have filenames starting with a digit and/or containing a
hyphen (e.g. "02_peer-groups.py", "fd02_feature-prototypes.py"), which are not
valid Python identifiers and therefore can't be `import`ed with a normal
statement. `load_module()` loads them directly from their file path instead.

Only import modules here whose top-level code is guarded by
`if __name__ == "__main__":` — anything else runs its script body (data
loading, training, ...) as a side effect of the import.
"""

import importlib.util
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent


def load_module(relative_path: str):
    """Import a pipeline script by path, e.g. load_module("prd_net/02_peer-groups.py")."""
    file_path = PIPELINE_DIR / relative_path
    module_name = "pipeline_under_test_" + file_path.stem.replace("-", "_")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
