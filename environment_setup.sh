#!/bin/bash
# sktime Local CI Environment Setup Script
# This script sets up the complete development environment to mirror the CI pipeline
# and enables comprehensive testing of all detection algorithms

set -e  # Exit on first error

echo "=========================================="
echo "sktime Local CI Environment Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"
echo "  (Required: >=3.10, <3.15)"
echo ""

# Step 1: Update pip, setuptools, wheel
echo "Step 1: Updating pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Dependencies updated"
echo ""

# Step 2: Install sktime in editable mode with all dependencies
echo "Step 2: Installing sktime with all dependencies..."
echo "  Installing: dev, all_extras, detection modules"
pip install -e ".[dev,all_extras,detection]"
echo "✓ sktime installed"
echo ""

# Step 3: Verify key detection dependencies
echo "Step 3: Verifying detection soft dependencies..."
DEPS_TO_CHECK=(
    "hmmlearn"
    "numba"
    "pyod"
    "pytest"
    "numpy"
    "pandas"
    "scikit-learn"
)

for dep in "${DEPS_TO_CHECK[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        VERSION=$(python3 -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null)
        echo "  ✓ $dep ($VERSION)"
    else
        echo "  ✗ $dep (NOT INSTALLED - optional, may not be required)"
    fi
done
echo ""

# Step 4: Verify sktime imports
echo "Step 4: Verifying sktime core modules..."
python3 -c "
import sktime
from sktime.registry import all_estimators
from sktime.utils.estimator_checks import check_estimator
from sktime.detection import *
print('  ✓ sktime.registry')
print('  ✓ sktime.utils.estimator_checks')
print('  ✓ sktime.detection')
" && echo "✓ All core modules imported successfully" || { echo "✗ Import failed"; exit 1; }
echo ""

# Step 5: List available detection estimators
echo "Step 5: Detecting available anomaly detectors..."
python3 << 'EOF'
from sktime.registry import all_estimators

detectors = all_estimators(
    estimator_types="detector",
    filter_tags={"task": "anomaly_detection"},
    return_names=True
)

if detectors:
    print(f"  Found {len(detectors)} anomaly detectors:")
    for name, cls in detectors[:10]:  # Show first 10
        print(f"    - {name}")
    if len(detectors) > 10:
        print(f"    ... and {len(detectors) - 10} more")
else:
    print("  No anomaly detectors found - may need to install additional dependencies")
EOF
echo "✓ Detector inventory complete"
echo ""

echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python hunt.py"
echo "     (Automated API compliance testing)"
echo "  2. Run: python stress_test.py"
echo "     (Edge-case stress testing)"
echo "  3. Review output and identify bugs"
echo ""
