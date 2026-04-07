"""Base forecaster class for all time series forecasting models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, model_name: str, config):
        """
        Initialize the base forecaster.
        
        Args:
            model_name: Name of the model
            config: Model configuration object
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_loaded = False
        
        logger.info(f"Initialized {model_name} forecaster")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from disk."""
        pass
    
    @abstractmethod
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using the model.
        
        Args:
            context: Historical time series data
            forecast_horizon: Number of steps to predict ahead
            quantiles: Quantiles to compute for probabilistic forecasts
            
        Returns:
            Dictionary containing predictions with keys:
                - 'mean': Point forecasts (mean/median)
                - 'quantiles': Dictionary mapping quantile levels to values
                - 'lower_bound': Lower prediction interval (optional)
                - 'upper_bound': Upper prediction interval (optional)
        """
        pass
    
    @abstractmethod
    def batch_predict(
        self,
        contexts: List[Union[np.ndarray, pd.Series, pd.DataFrame]],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Make predictions for multiple time series.
        
        Args:
            contexts: List of historical time series data
            forecast_horizon: Number of steps to predict ahead
            quantiles: Quantiles to compute for probabilistic forecasts
            
        Returns:
            List of prediction dictionaries
        """
        pass
    
    def validate_input(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """
        Validate and convert input to numpy array.
        
        Args:
            context: Input time series data
            
        Returns:
            Numpy array of values
        """
        if isinstance(context, pd.DataFrame):
            # Extract numeric column
            numeric_cols = context.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in DataFrame")
            return context[numeric_cols[0]].values
        
        elif isinstance(context, pd.Series):
            return context.values
        
        elif isinstance(context, np.ndarray):
            if context.ndim > 1:
                raise ValueError(f"Expected 1D array, got {context.ndim}D")
            return context
        
        else:
            raise TypeError(f"Unsupported input type: {type(context)}")
    
    def get_prediction_intervals(
        self,
        quantiles_dict: Dict[float, np.ndarray],
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9
    ) -> tuple:
        """
        Extract prediction intervals from quantile predictions.
        
        Args:
            quantiles_dict: Dictionary mapping quantile levels to values
            lower_quantile: Lower bound quantile level
            upper_quantile: Upper bound quantile level
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        lower_bound = quantiles_dict.get(lower_quantile)
        upper_bound = quantiles_dict.get(upper_quantile)
        
        if lower_bound is None or upper_bound is None:
            logger.warning(
                f"Quantiles {lower_quantile} or {upper_quantile} not available"
            )
            return None, None
        
        return lower_bound, upper_bound
    
    def save_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: str,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        Save predictions to CSV file.
        
        Args:
            predictions: Dictionary of predictions
            output_path: Path to save the CSV file
            timestamps: Optional timestamps for the forecast horizon
        """
        import os
        from pathlib import Path
        
        # Create output directory if needed
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        
        # Build DataFrame
        data = {'forecast_mean': predictions['mean']}
        
        if 'quantiles' in predictions:
            for q_level, q_values in predictions['quantiles'].items():
                data[f'quantile_{q_level}'] = q_values
        
        if 'lower_bound' in predictions and predictions['lower_bound'] is not None:
            data['lower_bound'] = predictions['lower_bound']
        
        if 'upper_bound' in predictions and predictions['upper_bound'] is not None:
            data['upper_bound'] = predictions['upper_bound']
        
        df = pd.DataFrame(data)
        
        if timestamps is not None:
            df.index = timestamps
            df.index.name = 'timestamp'
        
        df.to_csv(output_path)
        logger.info(f"Saved predictions to {output_path}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, loaded={self.is_loaded})"
