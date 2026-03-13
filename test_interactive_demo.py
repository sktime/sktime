#!/usr/bin/env python3
"""
Simple test script to verify interactive plotting functionality.
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        print("✓ numpy and pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy/pandas: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✓ plotly imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import plotly: {e}")
        return False
    
    try:
        import ipywidgets as widgets
        print("✓ ipywidgets imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ipywidgets: {e}")
        return False
    
    try:
        from sktime.datasets import load_airline
        from sktime.forecasting.theta import ThetaForecaster
        from sktime.split import ExpandingWindowSplitter
        print("✓ sktime modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sktime modules: {e}")
        return False
    
    try:
        from sktime.utils.plotting_interactive import (
            plot_interactive_cv,
            plot_interactive_series,
            InteractiveForecaster
        )
        print("✓ Interactive plotting modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import interactive plotting modules: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without creating plots."""
    print("\nTesting basic functionality...")
    
    try:
        # Load data
        from sktime.datasets import load_airline
        y = load_airline().iloc[:50]
        print("✓ Data loaded successfully")
        
        # Create CV splitter
        from sktime.split import ExpandingWindowSplitter
        import numpy as np
        cv = ExpandingWindowSplitter(
            fh=np.arange(1, 13),
            initial_window=24,
            step_length=12
        )
        print("✓ CV splitter created successfully")
        
        # Test CV splitting
        splits = list(cv.split(y))
        print(f"✓ CV splitting works: {len(splits)} splits created")
        
        # Create forecaster
        from sktime.forecasting.theta import ThetaForecaster
        forecaster = ThetaForecaster(sp=12)
        forecaster.fit(y, fh=np.arange(1, 13))
        print("✓ Forecaster fitted successfully")
        
        # Test prediction
        pred = forecaster.predict()
        print(f"✓ Prediction works: {len(pred)} predictions generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_interactive_functions():
    """Test interactive functions (without displaying plots)."""
    print("\nTesting interactive functions...")
    
    try:
        from sktime.datasets import load_airline
        from sktime.split import ExpandingWindowSplitter
        from sktime.utils.plotting_interactive import (
            plot_interactive_cv,
            plot_interactive_series,
            InteractiveForecaster
        )
        import numpy as np
        
        # Load data
        y = load_airline().iloc[:50]
        
        # Test interactive CV
        cv = ExpandingWindowSplitter(
            fh=np.arange(1, 13),
            initial_window=24,
            step_length=12
        )
        
        fig, controls = plot_interactive_cv(cv, y, title="Test CV")
        print("✓ plot_interactive_cv works")
        print(f"  - Figure created: {fig is not None}")
        print(f"  - Controls created: {controls is not None}")
        
        # Test interactive series
        outliers = [10, 20, 30]
        corrections = {10: 400, 20: 450, 30: 500}
        
        fig, controls = plot_interactive_series(y, outliers, corrections)
        print("✓ plot_interactive_series works")
        print(f"  - Figure created: {fig is not None}")
        print(f"  - Controls created: {controls is not None}")
        
        # Test interactive forecaster
        from sktime.forecasting.theta import ThetaForecaster
        base_forecaster = ThetaForecaster(sp=12)
        interactive_fc = InteractiveForecaster(base_forecaster)
        
        interactive_fc.fit(y, fh=np.arange(1, 13))
        print("✓ InteractiveForecaster works")
        print(f"  - Is fitted: {interactive_fc.is_fitted}")
        
        # Test forecaster methods
        fig, controls = interactive_fc.plot_interactive_cv(cv, title="Test")
        print("✓ InteractiveForecaster.plot_interactive_cv works")
        
        fig, controls = interactive_fc.plot_interactive_series()
        print("✓ InteractiveForecaster.plot_interactive_series works")
        
        return True
        
    except Exception as e:
        print(f"✗ Interactive functions test failed: {e}")
        traceback.print_exc()
        return False

def test_plot_creation():
    """Test that plots can be created and saved."""
    print("\nTesting plot creation...")
    
    try:
        from sktime.datasets import load_airline
        from sktime.split import ExpandingWindowSplitter
        from sktime.utils.plotting_interactive import plot_interactive_cv
        import numpy as np
        
        # Load data
        y = load_airline().iloc[:50]
        
        # Create CV splitter
        cv = ExpandingWindowSplitter(
            fh=np.arange(1, 13),
            initial_window=24,
            step_length=12
        )
        
        # Create plot
        fig, controls = plot_interactive_cv(cv, y, title="Test Plot")
        
        # Test plot properties
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'show'), "Figure should have show method"
        assert hasattr(fig, 'update_layout'), "Figure should have update_layout method"
        
        print("✓ Plot creation works")
        print(f"  - Figure type: {type(fig)}")
        print(f"  - Figure methods available: {[m for m in dir(fig) if not m.startswith('_')][:10]}")
        
        # Test saving plot (optional)
        try:
            fig.write_html("test_interactive_cv.html")
            print("✓ Plot can be saved as HTML")
        except Exception as e:
            print(f"⚠ Plot saving failed (non-critical): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Plot creation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Interactive Plotting Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Interactive Functions Test", test_interactive_functions),
        ("Plot Creation Test", test_plot_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Interactive plotting functionality is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 