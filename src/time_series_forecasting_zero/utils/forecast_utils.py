"""
Forecasting utility functions for evaluation and visualization.

Provides convenient functions for:
- Computing metrics (RMSE, MAE, MAPE, Coverage)
- Plotting forecasts and residuals
- Saving results

Example:
    >>> from time_series_forecasting_zero.utils import forecast_utils
    >>> metrics = forecast_utils.compute_metrics(y_true, y_pred)
    >>> forecast_utils.plot_forecast(train_data, predictions, save_path='forecast.png')
"""

from typing import Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAPE value (in percentage)
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Compute prediction interval coverage.
    
    Args:
        y_true: Ground truth values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        
    Returns:
        Coverage rate (0 to 1)
    """
    covered = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(covered))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all common forecasting metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        lower_bound: Lower bound of prediction interval (optional)
        upper_bound: Upper bound of prediction interval (optional)
        y_train: Training data for MASE calculation (optional)
        
    Returns:
        Dictionary containing:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error (%)
            - coverage_80pct: 80% prediction interval coverage (if bounds provided)
            - mase: Mean Absolute Scaled Error (if y_train provided)
            
    Example:
        >>> metrics = compute_all_metrics(test_data, predictions['mean'],
        ...                               predictions['lower_bound'],
        ...                               predictions['upper_bound'])
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
    }
    
    # Compute coverage if bounds provided
    if lower_bound is not None and upper_bound is not None:
        metrics['coverage_80pct'] = compute_coverage(y_true, lower_bound, upper_bound)
    
    # Compute MASE if training data provided
    if y_train is not None and len(y_train) > 1:
        # Naive forecast error (one-step ahead)
        naive_errors = np.abs(np.diff(y_train))
        mae_naive = np.mean(naive_errors)
        
        if mae_naive > 0:
            metrics['mase'] = float(np.mean(np.abs(y_true - y_pred)) / mae_naive)
        else:
            metrics['mase'] = float('inf')
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Forecasting Metrics") -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics from compute_all_metrics()
        title: Title for the metrics table
        
    Example:
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> print_metrics(metrics, "Model Performance")
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Value':>15}")
    print(f"{'-'*60}")
    
    metric_names = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mape': 'MAPE (%)',
        'mase': 'MASE',
        'coverage_80pct': '80% PI Coverage (%)'
    }
    
    for key, name in metric_names.items():
        if key in metrics:
            if key == 'mape' or key == 'coverage_80pct':
                print(f"{name:<25} {metrics[key]:>14.2f}%")
            else:
                print(f"{name:<25} {metrics[key]:>15.4f}")
    
    print(f"{'='*60}\n")


def plot_forecast(
    train_data: np.ndarray,
    predictions: Dict[str, np.ndarray],
    test_data: Optional[np.ndarray] = None,
    timestamps_train: Optional[pd.DatetimeIndex] = None,
    timestamps_test: Optional[pd.DatetimeIndex] = None,
    title: str = "Forecast Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot forecast with prediction intervals.
    
    Args:
        train_data: Historical training data
        predictions: Dictionary with 'mean', 'lower_bound', 'upper_bound'
        test_data: Actual test data for comparison (optional)
        timestamps_train: Timestamps for training data (optional)
        timestamps_test: Timestamps for test/forecast period (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_forecast(train_data, predictions, test_data,
        ...                    save_path='forecast.png')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis indices or timestamps
    if timestamps_train is not None:
        x_train = timestamps_train
        n_train = len(timestamps_train)
    else:
        x_train = np.arange(len(train_data))
        n_train = len(train_data)
    
    n_forecast = len(predictions['mean'])
    if timestamps_test is not None:
        x_forecast = timestamps_test
    else:
        x_forecast = np.arange(n_train, n_train + n_forecast)
    
    # Plot training data
    ax.plot(x_train, train_data, 'b-', linewidth=2, label='Historical Data')
    
    # Plot prediction intervals
    ax.fill_between(
        x_forecast,
        predictions['lower_bound'],
        predictions['upper_bound'],
        alpha=0.3,
        color='orange',
        label='80% Prediction Interval'
    )
    
    # Plot mean forecast
    ax.plot(x_forecast, predictions['mean'], 'r-', linewidth=2, label='Forecast (Mean)')
    
    # Plot actual test data if provided
    if test_data is not None:
        ax.plot(x_forecast[:len(test_data)], test_data, 'g--', linewidth=2, label='Actual')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis if using dates
    if timestamps_train is not None or timestamps_test is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Forecast plot saved to: {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot residual analysis (residuals over time and distribution).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_residuals(test_data, predictions['mean'],
        ...                     save_path='residuals.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Residuals over time
    axes[0].plot(residuals, 'b-', linewidth=1.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Residual', fontsize=11)
    axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Predicted vs Actual
    axes[2].scatter(y_true, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='navy')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[2].set_xlabel('Actual Values', fontsize=11)
    axes[2].set_ylabel('Predicted Values', fontsize=11)
    axes[2].set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[2].legend(loc='best', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Residual plot saved to: {save_path}")
    
    return fig


def compare_models_plot(
    results: Dict[str, Dict[str, np.ndarray]],
    test_data: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7)
):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary mapping model names to predictions
        test_data: Actual test data
        timestamps: Timestamps for test period (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> results = forecaster.compare_models(context=train_data, ...)
        >>> compare_models_plot(results, test_data, save_path='comparison.png')
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis
    if timestamps is not None:
        x = timestamps
    else:
        x = np.arange(len(test_data))
    
    # Plot actual data
    ax.plot(x, test_data, 'k-', linewidth=2.5, label='Actual', zorder=10)
    
    # Plot each model's forecast
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, (model_name, predictions) in enumerate(results.items()):
        if 'error' in predictions:
            continue
        
        color = colors[idx % len(colors)]
        ax.plot(
            x[:len(predictions['mean'])],
            predictions['mean'],
            '--',
            linewidth=2,
            color=color,
            label=f'{model_name}',
            alpha=0.8
        )
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
    
    return fig


def save_predictions_to_csv(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Dictionary with 'mean', 'lower_bound', 'upper_bound', etc.
        output_path: Path to save CSV file
        timestamps: Optional timestamps for the forecast horizon
        
    Example:
        >>> save_predictions_to_csv(predictions, 'output/predictions.csv', timestamps)
    """
    df_data = {'prediction_mean': predictions['mean']}
    
    if 'lower_bound' in predictions:
        df_data['lower_bound'] = predictions['lower_bound']
    if 'upper_bound' in predictions:
        df_data['upper_bound'] = predictions['upper_bound']
    if 'quantiles' in predictions:
        for q, values in predictions['quantiles'].items():
            df_data[f'quantile_{q}'] = values
    
    if timestamps is not None:
        df_data['timestamp'] = timestamps[:len(predictions['mean'])]
    
    df = pd.DataFrame(df_data)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")


def quick_evaluate(
    train_data: np.ndarray,
    test_data: np.ndarray,
    predictions: Dict[str, np.ndarray],
    model_name: str = "Model",
    save_plots: bool = False,
    output_dir: str = "./outputs"
) -> Dict[str, float]:
    """
    Quick evaluation: compute metrics and optionally save plots.
    
    This is a convenience function that combines metrics computation
    and visualization in one call.
    
    Args:
        train_data: Training data
        test_data: Test data (ground truth)
        predictions: Dictionary with prediction results
        model_name: Name of the model for titles
        save_plots: Whether to save plots
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of computed metrics
        
    Example:
        >>> metrics = quick_evaluate(train_data, test_data, predictions,
        ...                         model_name='Chronos-2', save_plots=True)
    """
    # Compute metrics
    metrics = compute_all_metrics(
        y_true=test_data,
        y_pred=predictions['mean'],
        lower_bound=predictions.get('lower_bound'),
        upper_bound=predictions.get('upper_bound'),
        y_train=train_data
    )
    
    # Print metrics
    print_metrics(metrics, title=f"{model_name} Performance")
    
    # Save plots if requested
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Forecast plot
        plot_forecast(
            train_data=train_data,
            predictions=predictions,
            test_data=test_data,
            title=f"{model_name} Forecast",
            save_path=str(output_path / f"{model_name.lower().replace(' ', '_')}_forecast.png")
        )
        
        # Residual plot
        plot_residuals(
            y_true=test_data,
            y_pred=predictions['mean'],
            title=f"{model_name} Residuals",
            save_path=str(output_path / f"{model_name.lower().replace(' ', '_')}_residuals.png")
        )
    
    return metrics
