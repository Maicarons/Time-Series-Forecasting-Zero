"""Evaluation metrics for time series forecasting."""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger


class MetricsEvaluator:
    """Calculate evaluation metrics for forecasting results."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(MetricsEvaluator.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            logger.warning("All true values are zero, MAPE is undefined")
            return np.nan
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            SMAPE value (as percentage)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            logger.warning("All denominators are zero, SMAPE is undefined")
            return np.nan
        
        diff = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
        
        return np.mean(diff) * 100
    
    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonality: int = 1
    ) -> float:
        """
        Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_train: Training data (for scaling)
            seasonality: Seasonal period
            
        Returns:
            MASE value
        """
        # Calculate naive forecast error on training data
        if len(y_train) <= seasonality:
            logger.warning("Training data too short for MASE calculation")
            return np.nan
        
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        scale = np.mean(naive_errors)
        
        if scale == 0:
            logger.warning("Scale factor is zero, MASE is undefined")
            return np.nan
        
        return np.mean(np.abs(y_true - y_pred)) / scale
    
    @staticmethod
    def quantile_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantile: float
    ) -> float:
        """
        Calculate quantile loss (pinball loss).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted quantile values
            quantile: Quantile level (0 to 1)
            
        Returns:
            Quantile loss value
        """
        errors = y_true - y_pred
        return np.mean(
            np.maximum(quantile * errors, (quantile - 1) * errors)
        )
    
    @staticmethod
    def coverage(
        y_true: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> float:
        """
        Calculate coverage rate of prediction intervals.
        
        Args:
            y_true: Ground truth values
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            
        Returns:
            Coverage rate (0 to 1)
        """
        covered = (y_true >= lower_bound) & (y_true <= upper_bound)
        return np.mean(covered)
    
    @staticmethod
    def interval_width(
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> float:
        """
        Calculate average width of prediction intervals.
        
        Args:
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            
        Returns:
            Average interval width
        """
        return np.mean(upper_bound - lower_bound)
    
    @classmethod
    def calculate_all_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        quantiles_pred: Optional[Dict[float, np.ndarray]] = None,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        seasonality: int = 1
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values (mean/median)
            y_train: Training data (for MASE)
            quantiles_pred: Dictionary of quantile predictions
            lower_bound: Lower prediction interval
            upper_bound: Upper prediction interval
            seasonality: Seasonal period for MASE
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'MSE': cls.mse(y_true, y_pred),
            'RMSE': cls.rmse(y_true, y_pred),
            'MAE': cls.mae(y_true, y_pred),
            'MAPE': cls.mape(y_true, y_pred),
            'SMAPE': cls.smape(y_true, y_pred),
        }
        
        # MASE requires training data
        if y_train is not None:
            metrics['MASE'] = cls.mase(y_true, y_pred, y_train, seasonality)
        
        # Quantile metrics
        if quantiles_pred is not None:
            for q_level, q_values in quantiles_pred.items():
                metrics[f'Quantile_Loss_{q_level}'] = cls.quantile_loss(
                    y_true, q_values, q_level
                )
        
        # Prediction interval metrics
        if lower_bound is not None and upper_bound is not None:
            metrics['Coverage'] = cls.coverage(y_true, lower_bound, upper_bound)
            metrics['Interval_Width'] = cls.interval_width(lower_bound, upper_bound)
        
        logger.info(f"Metrics calculated: {metrics}")
        
        return metrics
    
    @classmethod
    def print_metrics(cls, metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        print("\n" + "="*50)
        print("Forecasting Evaluation Metrics")
        print("="*50)
        
        for name, value in metrics.items():
            if np.isnan(value):
                print(f"{name:25s}: N/A")
            elif value > 1000 or value < 0.001:
                print(f"{name:25s}: {value:.4e}")
            else:
                print(f"{name:25s}: {value:.4f}")
        
        print("="*50 + "\n")
