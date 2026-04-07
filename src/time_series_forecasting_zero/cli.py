"""
Command Line Interface for Time Series Forecasting.

Usage:
    tsforecast predict --model chronos2 --data data/test.csv --horizon 128
    tsforecast predict --config config.ini
    tsforecast compare --data data/test.csv --horizon 128
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Try to import from installed package first, then fallback to local src
try:
    from time_series_forecasting_zero.models.unified import UnifiedForecaster
    from time_series_forecasting_zero.data.loader import TimeSeriesDataLoader
    from time_series_forecasting_zero.data.preprocessor import TimeSeriesPreprocessor
    from time_series_forecasting_zero.utils.evaluator import MetricsEvaluator
    from time_series_forecasting_zero.utils.visualizer import ForecastVisualizer
    from time_series_forecasting_zero.configs.config import (
        get_chronos2_config,
        get_timesfm_config,
        get_tirex_config,
        get_data_config,
        load_config_from_ini
    )
except ImportError:
    # Fallback for development mode
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.unified import UnifiedForecaster
    from src.data.loader import TimeSeriesDataLoader
    from src.data.preprocessor import TimeSeriesPreprocessor
    from src.utils.evaluator import MetricsEvaluator
    from src.utils.visualizer import ForecastVisualizer
    from src.configs.config import (
        get_chronos2_config,
        get_timesfm_config,
        get_tirex_config,
        get_data_config,
        load_config_from_ini
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="tsforecast",
        description="Time Series Forecasting with Multiple Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model prediction
  tsforecast predict --model chronos2 --data data/test.csv --horizon 128
  
  # Using configuration file
  tsforecast predict --config config.ini
  
  # Compare all models
  tsforecast compare --data data/test.csv --horizon 128
  
  # Specify model path
  tsforecast predict --model timesfm --model-path ./models/timesfm-2.5 --data data/test.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Make predictions using a specific model"
    )
    _add_predict_args(predict_parser)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare predictions from multiple models"
    )
    _add_compare_args(compare_parser)
    
    return parser


def _add_predict_args(parser: argparse.ArgumentParser):
    """Add arguments for predict command."""
    parser.add_argument(
        "--model",
        type=str,
        default="chronos2",
        choices=["chronos2", "timesfm", "tirex"],
        help="Model to use for forecasting (default: chronos2)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to INI configuration file (overrides other arguments)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (optional, uses default if not specified)"
    )
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=128,
        help="Forecast horizon (default: 128)"
    )
    
    parser.add_argument(
        "--time-col",
        type=str,
        default="timestamp",
        help="Name of timestamp column (default: timestamp)"
    )
    
    parser.add_argument(
        "--value-col",
        type=str,
        default="value",
        help="Name of value column (default: value)"
    )
    
    parser.add_argument(
        "--freq",
        type=str,
        default="H",
        help="Time series frequency: H=hourly, D=daily, W=weekly, M=monthly (default: H)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)"
    )
    
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Quantiles to compute (default: 0.1 0.5 0.9)"
    )


def _add_compare_args(parser: argparse.ArgumentParser):
    """Add arguments for compare command."""
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
        help="Forecast horizon (default: 128)"
    )
    
    parser.add_argument(
        "--time-col",
        type=str,
        default="timestamp",
        help="Name of timestamp column (default: timestamp)"
    )
    
    parser.add_argument(
        "--value-col",
        type=str,
        default="value",
        help="Name of value column (default: value)"
    )
    
    parser.add_argument(
        "--freq",
        type=str,
        default="H",
        help="Time series frequency (default: H)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["chronos2", "timesfm", "tirex"],
        help="Models to compare (default: all)"
    )


def run_predict(args: argparse.Namespace):
    """Run prediction command."""
    print("="*70)
    print("Time-Series-Forecasting-Zero - Prediction")
    print("="*70)
    
    # Load configuration from INI file if provided
    if args.config:
        print(f"\nLoading configuration from {args.config}...")
        ini_config = load_config_from_ini(args.config, section=args.model)
        # Merge with args, CLI args take precedence
        for key, value in ini_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    
    # Validate required arguments
    if not args.data:
        logger.error("Data file is required. Use --data or --config.")
        sys.exit(1)
    
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
        # Prepare model config
        model_kwargs = {
            "device": args.device,
            "forecast_horizon": args.horizon
        }
        
        # Add model_path if specified
        if args.model_path:
            model_kwargs["model_path"] = args.model_path
        
        forecaster = UnifiedForecaster(
            model_name=args.model,
            **model_kwargs
        )
        
        print("Loading model...")
        forecaster.load_model()
        
        print("Making predictions...")
        predictions = forecaster.predict(
            context=train_values,
            forecast_horizon=min(args.horizon, len(test_values)),
            quantiles=args.quantiles
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
            save_path=str(output_dir / f"{args.model}_forecast.png")
        )
        
        # Plot residuals
        fig2 = visualizer.plot_residuals(
            y_true=y_true,
            y_pred=y_pred,
            title="Residual Analysis",
            save_path=str(output_dir / f"{args.model}_residuals.png")
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


def run_compare(args: argparse.Namespace):
    """Run comparison command."""
    print("="*70)
    print("Time-Series-Forecasting-Zero - Model Comparison")
    print("="*70)
    
    # Load data
    print(f"\n[1/4] Loading data from {args.data}...")
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
    
    # Split data
    print(f"\n[2/4] Splitting data...")
    try:
        train_df, val_df, test_df = loader.train_test_split(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        train_values = train_df[args.value_col].values
        test_values = test_df[args.value_col].values
        
        print(f"✓ Train: {len(train_values)}, Test: {len(test_values)}")
        
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        sys.exit(1)
    
    # Compare models
    print(f"\n[3/4] Comparing models: {', '.join(args.models)}...")
    try:
        # Use first model for initialization
        forecaster = UnifiedForecaster(
            model_name=args.models[0],
            device=args.device,
            forecast_horizon=args.horizon
        )
        
        print("Comparing models...")
        results = forecaster.compare_models(
            context=train_values,
            forecast_horizon=min(args.horizon, len(test_values)),
            quantiles=[0.1, 0.5, 0.9],
            models=args.models
        )
        
        print(f"✓ Comparison completed")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate and save results
    print(f"\n[4/4] Evaluating results...")
    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = MetricsEvaluator()
        
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")
        print(f"\n{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
        print("-" * 70)
        
        comparison_data = []
        
        for model_name, predictions in results.items():
            if 'error' in predictions:
                print(f"{model_name:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
                continue
            
            pred_len = min(len(predictions['mean']), len(test_values))
            y_true = test_values[:pred_len]
            y_pred = predictions['mean'][:pred_len]
            
            metrics = evaluator.calculate_all_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_train=train_values
            )
            
            print(f"{model_name:<15} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} {metrics['mape']:<12.2f}")
            
            comparison_data.append({
                'model': model_name,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"\n✓ Comparison results saved to {output_dir / 'model_comparison.csv'}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Comparison completed successfully!")
    print("="*70)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "predict":
        run_predict(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
