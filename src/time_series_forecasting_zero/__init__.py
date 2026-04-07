"""
Time Series Forecasting Zero - A unified framework for time series forecasting.

Supports multiple state-of-the-art models:
- Chronos-2 (Amazon)
- TimesFM-2.5 (Google)
- TiRex (NX-AI)

Usage modes:
1. Direct execution: Import and use in your Python code
2. CLI: Use the 'tsforecast' command
3. PyPI package: Install via pip and import

Examples:
    >>> from time_series_forecasting_zero import UnifiedForecaster
    >>> forecaster = UnifiedForecaster('chronos2', model_path='./models/chronos-2')
    >>> forecaster.load_model()
    >>> predictions = forecaster.predict(context=data, forecast_horizon=128)
    
Utility functions:
    >>> from time_series_forecasting_zero.utils import forecast_utils
    >>> metrics = forecast_utils.compute_all_metrics(y_true, y_pred)
    >>> forecast_utils.plot_forecast(train_data, predictions, save_path='forecast.png')
"""

__version__ = "0.1.0"
__author__ = "Time-Series-Forecasting-Zero Team"

from .models.unified import UnifiedForecaster
from .data.loader import TimeSeriesDataLoader
from .data.preprocessor import TimeSeriesPreprocessor
from .utils.evaluator import MetricsEvaluator
from .utils.visualizer import ForecastVisualizer

# Export utility functions for easy access
from .utils.forecast_utils import (
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_coverage,
    compute_all_metrics,
    print_metrics,
    plot_forecast,
    plot_residuals,
    compare_models_plot,
    save_predictions_to_csv,
    quick_evaluate,
)

__all__ = [
    # Core classes
    "UnifiedForecaster",
    "TimeSeriesDataLoader",
    "TimeSeriesPreprocessor",
    "MetricsEvaluator",
    "ForecastVisualizer",
    # Utility functions
    "compute_rmse",
    "compute_mae",
    "compute_mape",
    "compute_coverage",
    "compute_all_metrics",
    "print_metrics",
    "plot_forecast",
    "plot_residuals",
    "compare_models_plot",
    "save_predictions_to_csv",
    "quick_evaluate",
]
