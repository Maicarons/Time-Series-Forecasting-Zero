"""
Example 3: Batch Forecasting for Multiple Time Series

This example demonstrates how to make predictions for multiple time series
simultaneously using batch prediction.

Usage:
    python examples/03_batch_forecasting.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import from installed package first, then fallback to local src
try:
    from time_series_forecasting_zero.models.unified import UnifiedForecaster
    from time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
except ImportError:
    from src.time_series_forecasting_zero.models.unified import UnifiedForecaster
    from src.time_series_forecasting_zero.data.loader import TimeSeriesDataLoader


def load_and_create_multiple_series(n_series=5):
    """Load real data and create multiple series with variations."""
    
    print(f"Loading real data and creating {n_series} series...")
    
    # Load real data
    loader = TimeSeriesDataLoader(data_dir=str(project_root / "data"))
    df = loader.load_csv("test_data.csv")
    base_data = df['value'].values
    
    series_list = []
    metadata = []
    
    for i in range(n_series):
        # Create variations of the base data
        noise_level = 0.5 + i * 0.3
        noise = np.random.normal(0, noise_level, len(base_data))
        scale = 1.0 + i * 0.1
        
        values = base_data * scale + noise
        
        series_list.append(values)
        metadata.append({
            'series_id': f'series_{i+1}',
            'scale_factor': scale,
            'noise_level': noise_level
        })
    
    print(f"[OK] Created {len(series_list)} time series from real data")
    return series_list, metadata


def main():
    """Demonstrate batch forecasting."""
    
    print("="*70)
    print("Example 3: Batch Forecasting for Multiple Time Series")
    print("="*70)
    
    # Step 1: Load and create multiple time series
    print("\n[Step 1] Loading and creating multiple time series...")
    series_list, metadata = load_and_create_multiple_series(n_series=5)
    
    # Step 2: Split into train and test
    print("\n[Step 2] Preparing train/test splits...")
    train_size = 450
    contexts = [s[:train_size] for s in series_list]
    test_data = [s[train_size:] for s in series_list]
    forecast_horizon = len(test_data[0])
    
    print(f"  Training samples per series: {train_size}")
    print(f"  Test samples per series: {forecast_horizon}")
    print(f"  Number of series: {len(contexts)}")
    
    # Step 3: Initialize forecaster
    print("\n[Step 3] Initializing forecaster...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    forecaster = UnifiedForecaster(
        model_name="tirex",
        forecast_horizon=forecast_horizon,
        device=device
    )
    
    # Step 4: Load model
    print("\n[Step 4] Loading model...")
    forecaster.load_model()
    
    # Step 5: Make batch predictions
    print("\n[Step 5] Making batch predictions...")
    predictions_list = forecaster.batch_predict(
        contexts=contexts,
        forecast_horizon=forecast_horizon,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    print(f"[OK] Generated predictions for {len(predictions_list)} series")
    
    # Step 6: Evaluate each series
    print("\n[Step 6] Evaluating predictions for each series...")
    print("-" * 70)
    print(f"{'Series ID':<15} {'RMSE':<12} {'MAE':<12} {'Mean Pred':<12} {'Mean Actual'}")
    print("-" * 70)
    
    results = []
    for i, (pred, actual, meta) in enumerate(zip(predictions_list, test_data, metadata)):
        rmse = np.sqrt(np.mean((pred['mean'] - actual)**2))
        mae = np.mean(np.abs(pred['mean'] - actual))
        mean_pred = np.mean(pred['mean'])
        mean_actual = np.mean(actual)
        
        results.append({
            'series_id': meta['series_id'],
            'rmse': rmse,
            'mae': mae,
            'mean_prediction': mean_pred,
            'mean_actual': mean_actual
        })
        
        print(f"{meta['series_id']:<15} {rmse:<12.4f} {mae:<12.4f} {mean_pred:<12.2f} {mean_actual:.2f}")
    
    # Step 7: Summary statistics
    print("\n" + "-" * 70)
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    print(f"{'Average':<15} {avg_rmse:<12.4f} {avg_mae:<12.4f}")
    print("-" * 70)
    
    # Step 8: Save results
    print("\n[Step 7] Saving batch results to CSV...")
    results_df = pd.DataFrame(results)
    output_path = project_root / "outputs" / "batch_predictions_summary.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"  [OK] Results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Batch forecasting completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
