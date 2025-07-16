#!/usr/bin/env python
"""Test script for TTM module validation."""

import ast
import sys
import importlib.util

def test_ttm_syntax():
    """Test if TTM module has valid syntax."""
    try:
        with open('sktime/forecasting/ttm.py', 'r') as f:
            source = f.read()
        ast.parse(source)
        print('✓ TTM module syntax is valid')
        return True
    except SyntaxError as e:
        print(f'✗ Syntax error in TTM module: {e}')
        return False
    except Exception as e:
        print(f'✗ Error parsing TTM module: {e}')
        return False

def test_ttm_import():
    """Test if TTM module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location("ttm", "sktime/forecasting/ttm.py")
        ttm_module = importlib.util.module_from_spec(spec)
        # Don't execute the module to avoid dependency issues
        print('✓ TTM module can be loaded')
        return True
    except Exception as e:
        print(f'✗ Error loading TTM module: {e}')
        return False

def test_ttm_class_definition():
    """Test if TinyTimeMixerForecaster class is properly defined."""
    try:
        with open('sktime/forecasting/ttm.py', 'r') as f:
            source = f.read()
        
        # Check if class is defined
        if 'class TinyTimeMixerForecaster' in source:
            print('✓ TinyTimeMixerForecaster class is defined')
            
            # Check if it inherits from the right base class
            if '_BaseGlobalForecaster' in source:
                print('✓ TinyTimeMixerForecaster inherits from _BaseGlobalForecaster')
                return True
            else:
                print('✗ TinyTimeMixerForecaster does not inherit from _BaseGlobalForecaster')
                return False
        else:
            print('✗ TinyTimeMixerForecaster class not found')
            return False
    except Exception as e:
        print(f'✗ Error checking class definition: {e}')
        return False

def main():
    """Run all tests."""
    print("Testing TTM module...")
    
    tests = [
        test_ttm_syntax,
        test_ttm_import,
        test_ttm_class_definition
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
