"""
Main entry point for time series forecasting.

Example usage:
    python main.py --model chronos2 --data data/sample.csv --horizon 128
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.unified import UnifiedForecaster
from src.data.loader import TimeSeriesDataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.utils.evaluator import MetricsEvaluator
from src.utils.visualizer import ForecastVisualizer
from src.configs.config import get_data_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting with Multiple Models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="chronos2",
        choices=["chronos2", "timesfm", "patchtst", "tirex"],
        help="Model to use for forecasting"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=128,
        help="Forecast horizon (number of steps ahead)"
    )
    
    parser.add_argument(
        "--time-col",
        type=str,
        default="timestamp",
        help="Name of timestamp column"
    )
    
    parser.add_argument(
        "--value-col",
        type=str,
        default="value",
        help="Name of value column"
    )
    
    parser.add_argument(
        "--freq",
        type=str,
        default="H",
        help="Time series frequency (H, D, W, M)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all models"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*70)
    print("Time-Series-Forecasting-Zero")
    print("="*70)
    
    # Load data
    print(f"\n[1/5] Loading data from {args.data}...")
    try:
        loader = TimeSeriesDataLoader(
            data_dir=str(Path(args.data).parent),
            time_column=args.time_col,
            value_column=args.value_col,
            freq=args.freq
        )
        
        data = loader.load_csv(Path(args.data).name)
        print(f"✓ Loaded {len(data)} records")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Preprocess data
    print(f"\n[2/5] Preprocessing data...")
    try:
        preprocessor = TimeSeriesPreprocessor(normalize=False)
        values = loader.get_values()
        timestamps = loader.get_timestamps()
        
        print(f"✓ Data range: {timestamps.min()} to {timestamps.max()}")
        print(f"✓ Value range: {values.min():.2f} to {values.max():.2f}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)
    
    # Split data
    print(f"\n[3/5] Splitting data into train/test sets...")
    try:
        train_df, val_df, test_df = loader.train_test_split(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        train_values = train_df[args.value_col].values
        test_values = test_df[args.value_col].values
        test_timestamps = test_df.index
        
        print(f"✓ Train: {len(train_values)}, Test: {len(test_values)}")
        
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        sys.exit(1)
    
    # Make predictions
    print(f"\n[4/5] Making predictions with {args.model}...")
    try:
        if args.compare:
            # Compare all models
            forecaster = UnifiedForecaster(
                model_name=args.model,
                device=args.device,
                forecast_horizon=args.horizon
            )
            
            print("Loading model...")
            forecaster.load_model()
            
            print("Comparing models...")
            results = forecaster.compare_models(
                context=train_values,
                forecast_horizon=min(args.horizon, len(test_values)),
                quantiles=[0.1, 0.5, 0.9]
            )
            
            # Use first model's result for evaluation
            model_name = list(results.keys())[0]
            predictions = results[model_name]
            
        else:
            # Single model
            forecaster = UnifiedForecaster(
                model_name=args.model,
                device=args.device,
                forecast_horizon=args.horizon
            )
            
            print("Loading model...")
            forecaster.load_model()
            
            print("Making predictions...")
            predictions = forecaster.predict(
                context=train_values,
                forecast_horizon=min(args.horizon, len(test_values)),
                quantiles=[0.1, 0.5, 0.9]
            )
        
        print(f"✓ Predictions generated")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate and visualize
    print(f"\n[5/5] Evaluating results...")
    try:
        # Ensure prediction length matches test length
        pred_len = min(len(predictions['mean']), len(test_values))
        y_true = test_values[:pred_len]
        y_pred = predictions['mean'][:pred_len]
        
        # Calculate metrics
        evaluator = MetricsEvaluator()
        metrics = evaluator.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_train=train_values,
            quantiles_pred=predictions.get('quantiles'),
            lower_bound=predictions.get('lower_bound'),
            upper_bound=predictions.get('upper_bound')
        )
        
        evaluator.print_metrics(metrics)
        
        # Create visualizations
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = ForecastVisualizer()
        
        # Plot forecast
        fig1 = visualizer.plot_forecast(
            historical=train_values,
            predictions=predictions,
            timestamps_hist=loader.get_timestamps()[:len(train_values)],
            timestamps_forecast=test_timestamps[:pred_len],
            title=f"Forecast using {args.model.upper()}",
            save_path=str(output_dir / "forecast.png")
        )
        
        # Plot residuals
        fig2 = visualizer.plot_residuals(
            y_true=y_true,
            y_pred=y_pred,
            title="Residual Analysis",
            save_path=str(output_dir / "residuals.png")
        )
        
        # Save predictions
        forecaster.save_predictions(
            predictions=predictions,
            output_path=str(output_dir / "predictions.csv"),
            timestamps=test_timestamps[:pred_len]
        )
        
        print(f"✓ Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Forecasting completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
