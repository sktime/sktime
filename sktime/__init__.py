"""sktime - the package for AI with time series.

See https://www.sktime.net/ for documentation and tutorials.
"""

__version__ = "1.0.1"

__all__ = ["show_versions"]

# Initialize and apply dependency monkeypatches (e.g. for requires_cython)
import sktime.utils.dependencies._dependencies
from sktime.utils._maint._show_versions import show_versions
