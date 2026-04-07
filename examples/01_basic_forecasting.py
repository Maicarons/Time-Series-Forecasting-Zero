"""
Example 1: Basic Forecasting with All Models

This example demonstrates how to use the UnifiedForecaster to make predictions
with all supported models: Chronos-2, TimesFM-2.5, and TiRex.

Usage:
    python examples/01_basic_forecasting.py
    python examples/01_basic_forecasting.py --model-path ./models/chronos-2
    python examples/01_basic_forecasting.py --chronos2-path ./models/chronos-2 --timesfm-path ./models/timesfm-2.5
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import from installed package first, then fallback to local src
try:
    from time_series_forecasting_zero.models.unified import UnifiedForecaster
    from time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    from time_series_forecasting_zero.utils.forecast_utils import compute_all_metrics, print_metrics
except ImportError:
    from src.time_series_forecasting_zero.models.unified import UnifiedForecaster
    from src.time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    from src.time_series_forecasting_zero.utils.forecast_utils import compute_all_metrics, print_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic Forecasting Example")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Default model path for all models")
    parser.add_argument("--chronos2-path", type=str, default=None,
                       help="Path to Chronos-2 model")
    parser.add_argument("--timesfm-path", type=str, default=None,
                       help="Path to TimesFM model")
    parser.add_argument("--tirex-path", type=str, default=None,
                       help="Path to TiRex model (not used, loads from HF)")
    return parser.parse_args()


def main():
    """Run basic forecasting example."""
    args = parse_args()
    
    print("="*70)
    print("Basic Forecasting Example")
    print("="*70)
    
    # Auto-detect device (prefer CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load real data from CSV
    print("\n[Step 1] Loading real data from CSV...")
    loader = TimeSeriesDataLoader(data_dir=str(project_root / "data"))
    df = loader.load_csv("test_data.csv")
    
    print(f"  Loaded {len(df)} data points")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    
    # Prepare data
    data = df['value'].values
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    forecast_horizon = len(test_data)
    
    print(f"\n  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Forecast horizon: {forecast_horizon}")
    
    # Define models to test
    models = ["chronos2", "timesfm", "tirex"]
    
    print("\n[Step 2] Testing all models...")
    print("-" * 70)
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Prepare model config
            model_kwargs = {
                "forecast_horizon": forecast_horizon,
                "device": device
            }
            
            # Add model path if specified
            if model_name == "chronos2" and args.chronos2_path:
                model_kwargs["model_path"] = args.chronos2_path
            elif model_name == "timesfm" and args.timesfm_path:
                model_kwargs["model_path"] = args.timesfm_path
            elif args.model_path:
                model_kwargs["model_path"] = args.model_path
            
            # Initialize forecaster
            print(f"Initializing {model_name}...")
            forecaster = UnifiedForecaster(
                model_name=model_name,
                **model_kwargs
            )
            
            # Load model
            print(f"Loading model from local storage...")
            forecaster.load_model()
            
            # Make prediction
            print(f"Making predictions...")
            predictions = forecaster.predict(
                context=train_data,
                forecast_horizon=forecast_horizon,
                quantiles=[0.1, 0.5, 0.9]  # 10th, 50th (median), 90th percentiles
            )
            
            # Calculate metrics using utility function
            metrics = compute_all_metrics(
                y_true=test_data,
                y_pred=predictions['mean'],
                lower_bound=predictions.get('lower_bound'),
                upper_bound=predictions.get('upper_bound'),
                y_train=train_data
            )
            
            print(f"\n[PASS] Prediction successful!")
            print_metrics(metrics, title=f"{model_name.upper()} Results")
            print(f"  Mean prediction (first 5): {predictions['mean'][:5]}")
            print(f"  Actual values (first 5):   {test_data[:5]}")
            
            results[model_name] = {
                'predictions': predictions,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<15} {'RMSE':<12} {'MAE':<12} {'Status'}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<15} {'N/A':<12} {'N/A':<12} [FAIL]")
        else:
            metrics = result['metrics']
            print(f"{model_name:<15} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} [PASS]")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
