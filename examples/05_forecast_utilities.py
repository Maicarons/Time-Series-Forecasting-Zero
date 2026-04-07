"""
Example 5: Using Forecast Utilities

This example demonstrates how to use the convenient utility functions for:
- Computing metrics (RMSE, MAE, MAPE, Coverage)
- Plotting forecasts and residuals
- Quick evaluation with one function call

Usage:
    python examples/05_forecast_utilities.py
    python examples/05_forecast_utilities.py --model chronos2
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
    from time_series_forecasting_zero import UnifiedForecaster
    from time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    # Import utility functions directly
    from time_series_forecasting_zero import (
        compute_all_metrics,
        print_metrics,
        plot_forecast,
        plot_residuals,
        quick_evaluate,
    )
except ImportError:
    from src.time_series_forecasting_zero import UnifiedForecaster
    from src.time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    from src.time_series_forecasting_zero import (
        compute_all_metrics,
        print_metrics,
        plot_forecast,
        plot_residuals,
        quick_evaluate,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast Utilities Example")
    parser.add_argument("--model", type=str, default="chronos2",
                       choices=["chronos2", "timesfm", "tirex"],
                       help="Model to use (default: chronos2)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model directory")
    return parser.parse_args()


def main():
    """Demonstrate forecast utility functions."""
    args = parse_args()
    
    print("="*70)
    print("Example 5: Using Forecast Utilities")
    print("="*70)
    
    # Load data
    print("\n[Step 1] Loading data...")
    loader = TimeSeriesDataLoader(data_dir=str(project_root / "data"))
    df = loader.load_csv("test_data.csv")
    
    print(f"✓ Loaded {len(df)} data points")
    
    # Prepare train/test split
    print("\n[Step 2] Preparing data...")
    train_size = int(len(df) * 0.8)
    train_data = df['value'].values[:train_size]
    test_data = df['value'].values[train_size:]
    timestamps_test = df.index[train_size:]
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Initialize forecaster
    print(f"\n[Step 3] Initializing {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    model_kwargs = {
        "forecast_horizon": len(test_data),
        "device": device
    }
    
    if args.model_path:
        model_kwargs["model_path"] = args.model_path
    
    forecaster = UnifiedForecaster(
        model_name=args.model,
        **model_kwargs
    )
    
    # Load model and predict
    print("\n[Step 4] Making predictions...")
    forecaster.load_model()
    
    predictions = forecaster.predict(
        context=train_data,
        forecast_horizon=len(test_data),
        quantiles=[0.1, 0.5, 0.9]
    )
    
    print("✓ Predictions generated")
    
    # ===== Method 1: Manual metric computation =====
    print("\n" + "="*70)
    print("Method 1: Manual Metric Computation")
    print("="*70)
    
    rmse = np.sqrt(np.mean((test_data - predictions['mean'])**2))
    mae = np.mean(np.abs(test_data - predictions['mean']))
    mape = np.mean(np.abs((test_data - predictions['mean']) / test_data)) * 100
    
    print(f"\n  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # ===== Method 2: Using compute_all_metrics() =====
    print("\n" + "="*70)
    print("Method 2: Using compute_all_metrics()")
    print("="*70)
    
    metrics = compute_all_metrics(
        y_true=test_data,
        y_pred=predictions['mean'],
        lower_bound=predictions.get('lower_bound'),
        upper_bound=predictions.get('upper_bound'),
        y_train=train_data
    )
    
    print_metrics(metrics, title=f"{args.model.upper()} Performance")
    
    # ===== Method 3: Individual metric functions =====
    print("\n" + "="*70)
    print("Method 3: Individual Metric Functions")
    print("="*70)
    
    from time_series_forecasting_zero import compute_rmse, compute_mae, compute_mape, compute_coverage
    
    rmse = compute_rmse(test_data, predictions['mean'])
    mae = compute_mae(test_data, predictions['mean'])
    mape = compute_mape(test_data, predictions['mean'])
    coverage = compute_coverage(
        test_data,
        predictions['lower_bound'],
        predictions['upper_bound']
    )
    
    print(f"\n  RMSE:              {rmse:.4f}")
    print(f"  MAE:               {mae:.4f}")
    print(f"  MAPE:              {mape:.2f}%")
    print(f"  80% PI Coverage:   {coverage*100:.1f}%")
    
    # ===== Visualization =====
    print("\n" + "="*70)
    print("Visualization")
    print("="*70)
    
    output_dir = project_root / "outputs" / "utilities_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot forecast
    print("\n  Creating forecast plot...")
    fig1 = plot_forecast(
        train_data=train_data,
        predictions=predictions,
        test_data=test_data,
        timestamps_test=timestamps_test,
        title=f"{args.model.upper()} Forecast with Prediction Intervals",
        save_path=str(output_dir / f"{args.model}_forecast.png")
    )
    
    # Plot residuals
    print("  Creating residual plot...")
    fig2 = plot_residuals(
        y_true=test_data,
        y_pred=predictions['mean'],
        title=f"{args.model.upper()} Residual Analysis",
        save_path=str(output_dir / f"{args.model}_residuals.png")
    )
    
    # ===== Method 4: Quick evaluate (all-in-one) =====
    print("\n" + "="*70)
    print("Method 4: Quick Evaluate (All-in-One)")
    print("="*70)
    
    print(f"\n  Running quick_evaluate for {args.model}...")
    metrics_quick = quick_evaluate(
        train_data=train_data,
        test_data=test_data,
        predictions=predictions,
        model_name=args.model.upper(),
        save_plots=True,
        output_dir=str(output_dir)
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ All utility functions demonstrated successfully!")
    print(f"\nOutput files saved to:")
    print(f"  - {output_dir / f'{args.model}_forecast.png'}")
    print(f"  - {output_dir / f'{args.model}_residuals.png'}")
    print(f"  - {output_dir / f'{args.model.lower()}_forecast.png'}")
    print(f"  - {output_dir / f'{args.model.lower()}_residuals.png'}")
    
    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("""
1. compute_all_metrics() - Compute all common metrics at once
2. print_metrics() - Pretty print metrics in a table
3. plot_forecast() - Visualize forecast with prediction intervals
4. plot_residuals() - Analyze residuals (3 subplots)
5. quick_evaluate() - One-call evaluation with metrics + plots

Import options:
    from time_series_forecasting_zero import compute_all_metrics
    from time_series_forecasting_zero.utils import forecast_utils
    """)
    
    print("="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
