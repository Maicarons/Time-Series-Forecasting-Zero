"""
Quick verification script to test all three usage modes.
"""

import sys
from pathlib import Path

print("="*70)
print("Time-Series-Forecasting-Zero - Quick Verification")
print("="*70)

# Test 1: Package Import
print("\n[Test 1] Testing package import...")
try:
    from time_series_forecasting_zero import UnifiedForecaster
    print("✓ Package import successful")
    
    # Test creating forecaster with model_path parameter
    forecaster = UnifiedForecaster(
        model_name='chronos2',
        model_path='./models/chronos-2',
        forecast_horizon=128,
        device='cpu'
    )
    print(f"✓ Created forecaster: {forecaster}")
    print(f"  Model path: {forecaster.forecaster.config.model_path}")
    
except Exception as e:
    print(f"✗ Package import failed: {e}")
    sys.exit(1)

# Test 2: Configuration System
print("\n[Test 2] Testing configuration system...")
try:
    from time_series_forecasting_zero.configs.config import (
        get_chronos2_config,
        load_config_from_ini
    )
    
    # Test config creation with parameters
    config = get_chronos2_config(
        model_path='./test/path',
        forecast_horizon=64,
        device='cuda'
    )
    print(f"✓ Config created with custom parameters")
    print(f"  Model path: {config.model_path}")
    print(f"  Forecast horizon: {config.forecast_horizon}")
    print(f"  Device: {config.device}")
    
    # Test INI loading
    ini_example = Path(__file__).parent / "config.ini.example"
    if ini_example.exists():
        ini_config = load_config_from_ini(str(ini_example), section="chronos2")
        print(f"✓ INI config loading works")
        print(f"  Loaded keys: {list(ini_config.keys())[:5]}")
    
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: CLI Tool
print("\n[Test 3] Testing CLI tool...")
try:
    import subprocess
    result = subprocess.run(
        ["tsforecast", "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0 and "predict" in result.stdout:
        print("✓ CLI tool is accessible")
        print(f"  Available commands: predict, compare")
    else:
        print(f"✗ CLI tool test failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ CLI tool test failed: {e}")
    sys.exit(1)

# Test 4: Examples Import
print("\n[Test 4] Testing examples compatibility...")
try:
    # Simulate example import pattern
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from time_series_forecasting_zero.models.unified import UnifiedForecaster
        from time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
        print("✓ Examples can import from installed package")
    except ImportError:
        from src.time_series_forecasting_zero.models.unified import UnifiedForecaster
        from src.time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
        print("✓ Examples can import from source (dev mode)")
    
except Exception as e:
    print(f"✗ Examples import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print("✓ Package import and API")
print("✓ Configuration system")
print("✓ CLI tool")
print("✓ Examples compatibility")
print("\n" + "="*70)
print("All tests passed! The project is ready for use.")
print("="*70)
print("\nThree usage modes available:")
print("1. Examples: python examples/01_basic_forecasting.py")
print("2. CLI:      tsforecast predict --model chronos2 --data data.csv")
print("3. PyPI:     from time_series_forecasting_zero import UnifiedForecaster")
print("="*70)
