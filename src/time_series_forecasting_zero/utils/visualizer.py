"""Visualization utilities for time series forecasting."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ForecastVisualizer:
    """Create visualizations for time series forecasts."""
    
    @staticmethod
    def plot_forecast(
        historical: Union[np.ndarray, pd.Series],
        predictions: Dict[str, np.ndarray],
        timestamps_hist: Optional[pd.DatetimeIndex] = None,
        timestamps_forecast: Optional[pd.DatetimeIndex] = None,
        title: str = "Time Series Forecast",
        xlabel: str = "Time",
        ylabel: str = "Value",
        show_quantiles: bool = True,
        quantile_levels: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 7)
    ) -> plt.Figure:
        """
        Plot historical data and forecast with prediction intervals.
        
        Args:
            historical: Historical time series values
            predictions: Dictionary with 'mean', 'quantiles', etc.
            timestamps_hist: Timestamps for historical data
            timestamps_forecast: Timestamps for forecast period
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_quantiles: Whether to show quantile bands
            quantile_levels: Which quantiles to display
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare historical data
        if isinstance(historical, pd.Series):
            hist_values = historical.values
            if timestamps_hist is None:
                timestamps_hist = historical.index
        else:
            hist_values = historical
        
        # Create timestamps if not provided
        if timestamps_hist is None:
            timestamps_hist = pd.RangeIndex(len(hist_values))
        
        if timestamps_forecast is None:
            forecast_len = len(predictions['mean'])
            timestamps_forecast = pd.RangeIndex(
                len(hist_values),
                len(hist_values) + forecast_len
            )
        
        # Plot historical data
        ax.plot(timestamps_hist, hist_values, 'b-', linewidth=2, label='Historical')
        
        # Plot forecast mean
        ax.plot(timestamps_forecast, predictions['mean'], 'r-', linewidth=2, label='Forecast')
        
        # Plot prediction intervals if available
        if show_quantiles and 'quantiles' in predictions:
            if quantile_levels is None:
                quantile_levels = [0.1, 0.9]
            
            if len(quantile_levels) == 2:
                lower_q, upper_q = quantile_levels
                if lower_q in predictions['quantiles'] and upper_q in predictions['quantiles']:
                    lower = predictions['quantiles'][lower_q]
                    upper = predictions['quantiles'][upper_q]
                    
                    ax.fill_between(
                        timestamps_forecast,
                        lower,
                        upper,
                        alpha=0.2,
                        color='red',
                        label=f'{int((upper_q-lower_q)*100)}% Prediction Interval'
                    )
        
        # Plot bounds if available
        if 'lower_bound' in predictions and predictions['lower_bound'] is not None:
            ax.plot(
                timestamps_forecast,
                predictions['lower_bound'],
                'r--',
                linewidth=1,
                alpha=0.5,
                label='Lower Bound'
            )
        
        if 'upper_bound' in predictions and predictions['upper_bound'] is not None:
            ax.plot(
                timestamps_forecast,
                predictions['upper_bound'],
                'r--',
                linewidth=1,
                alpha=0.5,
                label='Upper Bound'
            )
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        
        # Format x-axis for dates
        if isinstance(timestamps_hist, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_multiple_forecasts(
        historical: Union[np.ndarray, pd.Series],
        predictions_dict: Dict[str, Dict[str, np.ndarray]],
        timestamps_hist: Optional[pd.DatetimeIndex] = None,
        timestamps_forecast: Optional[pd.DatetimeIndex] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 7)
    ) -> plt.Figure:
        """
        Plot forecasts from multiple models for comparison.
        
        Args:
            historical: Historical time series values
            predictions_dict: Dictionary mapping model names to predictions
            timestamps_hist: Timestamps for historical data
            timestamps_forecast: Timestamps for forecast period
            title: Plot title
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare historical data
        if isinstance(historical, pd.Series):
            hist_values = historical.values
            if timestamps_hist is None:
                timestamps_hist = historical.index
        else:
            hist_values = historical
        
        if timestamps_hist is None:
            timestamps_hist = pd.RangeIndex(len(hist_values))
        
        # Plot historical data
        ax.plot(timestamps_hist, hist_values, 'k-', linewidth=2, label='Historical')
        
        # Plot each model's forecast
        colors = ['r', 'b', 'g', 'orange', 'purple', 'brown']
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = colors[idx % len(colors)]
            
            if timestamps_forecast is None:
                forecast_len = len(predictions['mean'])
                timestamps_forecast = pd.RangeIndex(
                    len(hist_values),
                    len(hist_values) + forecast_len
                )
            
            ax.plot(
                timestamps_forecast,
                predictions['mean'],
                color=color,
                linewidth=2,
                label=model_name
            )
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc='best')
        
        if isinstance(timestamps_hist, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Forecast Residuals Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot residual analysis (histogram and Q-Q like plot).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram of residuals
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Residual Distribution')
        ax1.set_xlabel('Residual')
        ax1.set_ylabel('Frequency')
        
        # Add mean and std annotations
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax1.text(
            0.05, 0.95,
            f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Residuals over time
        ax2.plot(range(len(residuals)), residuals, 'o-', markersize=4, linewidth=1, color='steelblue')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Residuals Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Residual')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_show: Optional[List[str]] = None,
        title: str = "Model Metrics Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot comparison of metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metrics_to_show: List of metrics to display
            title: Plot title
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        if metrics_to_show is None:
            # Get all unique metrics
            metrics_to_show = list(set().union(*[m.keys() for m in metrics_dict.values()]))
        
        models = list(metrics_dict.keys())
        x = np.arange(len(models))
        width = 0.8 / len(metrics_to_show)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for idx, metric in enumerate(metrics_to_show):
            values = [metrics_dict[model].get(metric, np.nan) for model in models]
            ax.bar(x + idx * width, values, width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics_to_show) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


# Import Union at module level - REMOVED (already imported at top)
