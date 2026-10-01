import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

# pycolmap and open3d are only needed to run the full pipeline. The functions
# under test are pure NumPy, so stub the heavy libraries when they're absent
# (CI installs just numpy, opencv, pillow and matplotlib).
for name in ("pycolmap", "open3d"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)
