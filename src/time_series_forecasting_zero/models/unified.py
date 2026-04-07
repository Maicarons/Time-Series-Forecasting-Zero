"""Unified forecasting interface that supports multiple models."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger

from .chronos2 import Chronos2Forecaster
from .timesfm import TimesFMForecaster
from .tirex import TiRexForecaster
from ..configs.config import (
    Chronos2Config,
    TimesFMConfig,
    TiRexConfig,
    get_chronos2_config,
    get_timesfm_config,
    get_tirex_config
)


class UnifiedForecaster:
    """
    Unified interface for time series forecasting with multiple models.
    
    Supports:
    - Chronos-2 (Amazon)
    - TimesFM-2.5 (Google)
    - TiRex (NX-AI)
    """
    
    SUPPORTED_MODELS = {
        "chronos2": Chronos2Forecaster,
        "timesfm": TimesFMForecaster,
        "tirex": TiRexForecaster
    }
    
    def __init__(
        self,
        model_name: str = "chronos2",
        config: Optional[Union[Chronos2Config, TimesFMConfig, TiRexConfig]] = None,
        **config_kwargs
    ):
        """
        Initialize the unified forecaster.
        
        Args:
            model_name: Name of the model to use ('chronos2', 'timesfm', 'tirex')
            config: Model configuration object (optional)
            **config_kwargs: Additional configuration parameters
            
        Example:
            >>> forecaster = UnifiedForecaster('chronos2')
            >>> forecaster = UnifiedForecaster('tirex', forecast_horizon=64)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.config = config
        self.config_kwargs = config_kwargs
        
        # Create forecaster instance
        self.forecaster = self._create_forecaster()
        
        logger.info(f"UnifiedForecaster initialized with model={model_name}")
    
    def _create_forecaster(self):
        """Create the appropriate forecaster based on model name."""
        forecaster_class = self.SUPPORTED_MODELS[self.model_name]
        
        if self.config is not None:
            return forecaster_class(config=self.config)
        else:
            # Create default config with kwargs
            if self.model_name == "chronos2":
                config = get_chronos2_config(**self.config_kwargs)
            elif self.model_name == "timesfm":
                config = get_timesfm_config(**self.config_kwargs)
            elif self.model_name == "tirex":
                config = get_tirex_config(**self.config_kwargs)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            return forecaster_class(config=config)
    
    def load_model(self) -> None:
        """Load the model from disk."""
        logger.info(f"Loading {self.model_name} model")
        self.forecaster.load_model()
        logger.info(f"Model {self.model_name} loaded successfully")
    
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using the selected model.
        
        Args:
            context: Historical time series data
            forecast_horizon: Number of steps to predict ahead
            quantiles: Quantiles to compute for probabilistic forecasts
            
        Returns:
            Dictionary containing:
                - 'mean': Point forecasts
                - 'quantiles': Dictionary of quantile predictions
                - 'lower_bound': Lower prediction interval
                - 'upper_bound': Upper prediction interval
                
        Example:
            >>> import pandas as pd
            >>> data = pd.read_csv('data.csv')
            >>> forecaster = UnifiedForecaster('chronos2')
            >>> predictions = forecaster.predict(data['value'], forecast_horizon=24)
            >>> print(predictions['mean'])
        """
        logger.info(f"Making prediction with {self.model_name}")
        
        try:
            predictions = self.forecaster.predict(
                context=context,
                forecast_horizon=forecast_horizon,
                quantiles=quantiles
            )
            
            logger.info("Prediction completed successfully")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
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
            quantiles: Quantiles to compute
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Making batch predictions with {self.model_name}")
        
        try:
            predictions = self.forecaster.batch_predict(
                contexts=contexts,
                forecast_horizon=forecast_horizon,
                quantiles=quantiles
            )
            
            logger.info(f"Batch prediction completed for {len(contexts)} series")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def compare_models(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compare predictions from multiple models.
        
        Args:
            context: Historical time series data
            forecast_horizon: Number of steps to predict ahead
            quantiles: Quantiles to compute
            models: List of models to compare (default: all supported models)
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        if models is None:
            models = list(self.SUPPORTED_MODELS.keys())
        
        logger.info(f"Comparing models: {models}")
        
        results = {}
        
        for model_name in models:
            try:
                logger.info(f"Running {model_name}")
                
                # Create temporary forecaster
                if model_name == "chronos2":
                    config = get_chronos2_config(**self.config_kwargs)
                    forecaster = Chronos2Forecaster(config=config)
                elif model_name == "timesfm":
                    config = get_timesfm_config(**self.config_kwargs)
                    forecaster = TimesFMForecaster(config=config)
                elif model_name == "tirex":
                    config = get_tirex_config(**self.config_kwargs)
                    forecaster = TiRexForecaster(config=config)
                else:
                    continue
                
                # Load and predict
                forecaster.load_model()
                predictions = forecaster.predict(
                    context=context,
                    forecast_horizon=forecast_horizon,
                    quantiles=quantiles
                )
                
                results[model_name] = predictions
                
                logger.info(f"{model_name} completed successfully")
                
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                results[model_name] = {"error": str(e)}
        
        logger.info("Model comparison completed")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return self.forecaster.get_model_info()
    
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
        self.forecaster.save_predictions(predictions, output_path, timestamps)
    
    def __repr__(self) -> str:
        return f"UnifiedForecaster(model={self.model_name})"
