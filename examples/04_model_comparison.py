"""
Example 4: Model Comparison

This example demonstrates how to compare predictions from different models
to choose the best one for your data.

Usage:
    python examples/04_model_comparison.py
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
    from time_series_forecasting_zero.utils.forecast_utils import (
        compute_all_metrics,
        print_metrics,
        compare_models_plot
    )
except ImportError:
    from src.time_series_forecasting_zero.models.unified import UnifiedForecaster
    from src.time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    from src.time_series_forecasting_zero.utils.forecast_utils import (
        compute_all_metrics,
        print_metrics,
        compare_models_plot
    )


def load_real_data():
    """Load real time series data from CSV."""
    
    print("Loading real data from CSV...")
    
    loader = TimeSeriesDataLoader(data_dir=str(project_root / "data"))
    df = loader.load_csv("test_data.csv")
    
    print(f"[OK] Loaded {len(df)} data points")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    
    return df['value'].values


def main():
    """Compare all available models."""
    
    print("="*70)
    print("Example 4: Model Comparison")
    print("="*70)
    
    # Step 1: Load real data
    print("\n[Step 1] Loading real data...")
    data = load_real_data()
    
    # Split into train and test
    train_size = 400
    train_data = data[:train_size]
    test_data = data[train_size:]
    forecast_horizon = len(test_data)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Step 2: Define models to compare
    print("\n[Step 2] Setting up model comparison...")
    models = ["chronos2", "timesfm", "tirex"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Models to compare: {', '.join(models)}")
    print(f"  Using device: {device}")
    
    # Step 3: Run comparison
    print("\n[Step 3] Running model comparison...")
    print("-" * 70)
    
    results = {}
    
    for model_name in models:
        print(f"\nTesting {model_name.upper()}...")
        
        try:
            # Initialize forecaster
            forecaster = UnifiedForecaster(
                model_name=model_name,
                forecast_horizon=forecast_horizon,
                device=device
            )
            
            # Load model
            print(f"  Loading model...")
            forecaster.load_model()
            
            # Make prediction
            print(f"  Making predictions...")
            predictions = forecaster.predict(
                context=train_data,
                forecast_horizon=forecast_horizon,
                quantiles=[0.1, 0.5, 0.9]
            )
            
            # Calculate metrics using utility function
            metrics = compute_all_metrics(
                y_true=test_data,
                y_pred=predictions['mean'],
                lower_bound=predictions.get('lower_bound'),
                upper_bound=predictions.get('upper_bound'),
                y_train=train_data
            )
            
            results[model_name] = {
                'metrics': metrics,
                'predictions': predictions
            }
            
            print(f"  [OK] RMSE: {metrics['rmse']:.4f}")
            print(f"  [OK] MAE:  {metrics['mae']:.4f}")
            print(f"  [OK] MAPE: {metrics['mape']:.2f}%")
            if 'coverage_80pct' in metrics:
                print(f"  [OK] 80% PI Coverage: {metrics['coverage_80pct']*100:.1f}%")
            
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results[model_name] = {'error': str(e)}
    
    # Step 4: Display comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'PI Coverage'}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} N/A")
        else:
            metrics = result['metrics']
            coverage_str = f"{metrics.get('coverage_80pct', 0)*100:.1f}%"
            print(f"{model_name:<15} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
                  f"{metrics['mape']:<12.2f} {coverage_str}")
    
    # Step 5: Identify best model
    print("\n" + "-" * 70)
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_rmse = min(valid_results.items(), key=lambda x: x[1]['metrics']['rmse'])
        best_mae = min(valid_results.items(), key=lambda x: x[1]['metrics']['mae'])
        best_mape = min(valid_results.items(), key=lambda x: x[1]['metrics']['mape'])
        
        print(f"\nBest by RMSE:  {best_rmse[0]} ({best_rmse[1]['metrics']['rmse']:.4f})")
        print(f"Best by MAE:   {best_mae[0]} ({best_mae[1]['metrics']['mae']:.4f})")
        print(f"Best by MAPE:  {best_mape[0]} ({best_mape[1]['metrics']['mape']:.2f}%)")
    
    print("-" * 70)
    
    # Step 6: Save comparison results
    print("\n[Step 4] Saving comparison results...")
    summary_data = []
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            summary_data.append({
                'model': model_name,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'pi_coverage': metrics.get('coverage_80pct', 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    output_path = project_root / "outputs" / "model_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"  [OK] Results saved to: {output_path}")
    
    # Optional: Create comparison plot
    try:
        print("\n  Creating comparison plot...")
        predictions_for_plot = {k: v['predictions'] for k, v in results.items() if 'error' not in v}
        fig = compare_models_plot(
            results=predictions_for_plot,
            test_data=test_data,
            title="Model Comparison",
            save_path=str(project_root / "outputs" / "model_comparison_plot.png")
        )
        print("  [OK] Comparison plot saved to: outputs/model_comparison_plot.png")
    except Exception as e:
        print(f"  [Warning] Could not create comparison plot: {e}")
    
    print("\n" + "="*70)
    print("Model comparison completed!")
    print("="*70)


if __name__ == "__main__":
    main()
