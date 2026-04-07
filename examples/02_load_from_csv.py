"""
Example 2: Loading Data from CSV Files

This example demonstrates how to load time series data from CSV files
and use it for forecasting.

Usage:
    python examples/02_load_from_csv.py
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


def main():
    """Demonstrate loading data from CSV and forecasting."""
    
    print("="*70)
    print("Example 2: Loading Data from CSV Files")
    print("="*70)
    
    # Step 1: Load real data
    print("\n[Step 1] Loading data from CSV...")
    loader = TimeSeriesDataLoader(data_dir=str(project_root / "data"))
    df = loader.load_csv("test_data.csv")
    
    print(f"[OK] Loaded {len(df)} data points")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    print(f"\n  First 5 rows:\n{df.head()}")
    
    # Step 2: Prepare training and test data
    print("\n[Step 2] Preparing data for forecasting...")
    train_size = int(len(df) * 0.8)
    train_data = df['value'].values[:train_size]
    test_data = df['value'].values[train_size:]
    forecast_horizon = len(test_data)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Forecast horizon: {forecast_horizon}")
    
    # Step 3: Initialize forecaster
    print("\n[Step 3] Initializing forecaster...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    forecaster = UnifiedForecaster(
        model_name="chronos2",
        forecast_horizon=forecast_horizon,
        device=device
    )
    
    # Step 4: Load model and make predictions
    print("\n[Step 4] Loading model and making predictions...")
    forecaster.load_model()
    
    predictions = forecaster.predict(
        context=train_data,
        forecast_horizon=forecast_horizon,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    # Step 5: Evaluate results
    print("\n[Step 5] Evaluating predictions...")
    rmse = np.sqrt(np.mean((predictions['mean'] - test_data)**2))
    mae = np.mean(np.abs(predictions['mean'] - test_data))
    
    print(f"  [OK] RMSE: {rmse:.4f}")
    print(f"  [OK] MAE:  {mae:.4f}")
    print(f"\n  Predictions (first 5): {predictions['mean'][:5]}")
    print(f"  Actual values (first 5): {test_data[:5]}")
    
    # Step 6: Save predictions to CSV
    print("\n[Step 6] Saving predictions to CSV...")
    pred_df = pd.DataFrame({
        'timestamp': df.index[train_size:],
        'actual': test_data,
        'predicted': predictions['mean'],
        'lower_bound': predictions['lower_bound'],
        'upper_bound': predictions['upper_bound']
    })
    
    output_path = project_root / "outputs" / "csv_predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    print(f"  [OK] Predictions saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
