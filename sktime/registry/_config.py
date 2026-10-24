"""Common configuration for the sktime registry."""

# modules to ignore in sktime for lookup
MODULES_TO_IGNORE = (
    "conftest",
    "tests",
    "setup",
    "contrib",
    "benchmarking",
    "utils",
    "all",
    "plotting",
    "_split",
    "test_split",
    "registry",
    "normal",
    "_normal",
    "libs",
)

# modules to ignore in scikit-learn for lookup
MODULES_TO_IGNORE_SKLEARN = [
    "array_api_compat",
    "conftest",
    "tests",
    "experimental",
]
