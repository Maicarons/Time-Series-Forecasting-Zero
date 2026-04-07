"""TimesFM-2.5 model interface for time series forecasting.

Official API: https://github.com/google-research/timesfm
Uses timesfm.TimesFM_2p5_200M_torch.from_pretrained()
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add timesfm-github to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "timesfm-github" / "src"))

from .base import BaseForecaster
from ..configs.config import TimesFMConfig


class TimesFMForecaster(BaseForecaster):
    """Forecasting interface for TimesFM-2.5 using official API."""
    
    def __init__(self, config: Optional[TimesFMConfig] = None):
        """
        Initialize TimesFM-2.5 forecaster.
        
        Args:
            config: Configuration for TimesFM model
        """
        self.config = config or TimesFMConfig()
        super().__init__(model_name="TimesFM-2.5", config=self.config)
    
    def load_model(self) -> None:
        """Load the TimesFM-2.5 model using official from_pretrained."""
        if self.is_loaded:
            logger.info("TimesFM-2.5 model already loaded")
            return
        
        logger.info(f"Loading TimesFM-2.5 model from {self.config.model_path}")
        
        try:
            # Import official TimesFM
            import timesfm
            
            # Load model using official from_pretrained
            # Note: from_pretrained may pass extra kwargs that cause issues
            try:
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.config.model_path,
                    torch_compile=False  # Disable compilation for stability
                )
            except TypeError as e:
                # If from_pretrained fails due to unexpected kwargs, load manually
                logger.warning(f"from_pretrained failed ({e}), loading manually...")
                self.model = timesfm.TimesFM_2p5_200M_torch(
                    torch_compile=False
                )
                # Load checkpoint manually
                from pathlib import Path
                model_path = Path(self.config.model_path)
                weights_file = model_path / "model.safetensors"
                if weights_file.exists():
                    self.model.model.load_checkpoint(str(weights_file), torch_compile=False)
                else:
                    raise FileNotFoundError(f"Weights not found at {weights_file}")
            
            self.is_loaded = True
            logger.info(f"TimesFM-2.5 model loaded successfully on {self.config.device}")
            
        except ImportError as e:
            logger.error(
                f"Failed to import timesfm package. Make sure timesfm-github is in the path.\n"
                f"Original error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load TimesFM-2.5 model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using TimesFM-2.5 official API.
        
        Args:
            context: Historical time series data
            forecast_horizon: Number of steps to predict ahead
            quantiles: Quantiles to compute (uses config default if None)
            
        Returns:
            Dictionary with 'mean', 'quantiles', 'lower_bound', 'upper_bound'
        """
        if not self.is_loaded:
            self.load_model()
        
        # Validate and prepare input
        context_values = self.validate_input(context)
        
        # Use config defaults if not provided
        if forecast_horizon is None:
            forecast_horizon = self.config.forecast_horizon
        if quantiles is None:
            quantiles = self.config.quantiles
        
        logger.info(
            f"Making prediction with context length={len(context_values)}, "
            f"horizon={forecast_horizon}"
        )
        
        try:
            # Compile model if not already compiled
            if self.model.compiled_decode is None:
                logger.info("Compiling TimesFM model for first use...")
                from timesfm import configs
                forecast_config = configs.ForecastConfig(
                    max_context=max(512, len(context_values)),  # Must be >= context length
                    max_horizon=forecast_horizon,
                    per_core_batch_size=1,
                )
                self.model.compile(forecast_config)
                logger.info("Model compiled successfully")
            
            # TimesFM expects list of arrays
            inputs = [context_values]
            
            # Make prediction using official forecast API
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=forecast_horizon,
                inputs=inputs
            )
            
            # Process outputs
            # point_forecast shape: (batch_size, horizon)
            # quantile_forecast shape: (batch_size, horizon, num_quantiles)
            predictions = self._process_outputs(
                point_forecast, 
                quantile_forecast, 
                quantiles, 
                forecast_horizon
            )
            
            logger.info("Prediction completed successfully")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
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
        if not self.is_loaded:
            self.load_model()
        
        logger.info(f"Making batch predictions for {len(contexts)} time series")
        
        # Validate and prepare all inputs
        context_values = [self.validate_input(ctx) for ctx in contexts]
        
        # Use config defaults if not provided
        if forecast_horizon is None:
            forecast_horizon = self.config.forecast_horizon
        if quantiles is None:
            quantiles = self.config.quantiles
        
        try:
            # TimesFM accepts list of arrays directly
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=forecast_horizon,
                inputs=context_values
            )
            
            # Process all outputs
            predictions = []
            for i in range(len(context_values)):
                single_point = point_forecast[i:i+1]
                single_quantile = quantile_forecast[i:i+1] if quantile_forecast.ndim > 2 else quantile_forecast
                
                pred = self._process_single_output(
                    single_point,
                    single_quantile,
                    quantiles,
                    forecast_horizon
                )
                predictions.append(pred)
            
            logger.info("Batch prediction completed")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _process_outputs(
        self,
        point_forecast: np.ndarray,
        quantile_forecast: np.ndarray,
        quantiles: List[float],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Process model outputs into prediction format.
        
        Args:
            point_forecast: Point predictions, shape: (batch_size, horizon)
            quantile_forecast: Quantile predictions, shape: (batch_size, horizon, num_quantiles)
            quantiles: Requested quantile levels
            forecast_horizon: Forecast horizon length
            
        Returns:
            Dictionary of processed predictions
        """
        return self._process_single_output(
            point_forecast, 
            quantile_forecast, 
            quantiles, 
            forecast_horizon
        )
    
    def _process_single_output(
        self,
        point_forecast: np.ndarray,
        quantile_forecast: np.ndarray,
        quantiles: List[float],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Process a single forecast output.
        
        Args:
            point_forecast: Single point forecast
            quantile_forecast: Single quantile forecast
            quantiles: Requested quantile levels
            forecast_horizon: Forecast horizon length
            
        Returns:
            Dictionary of processed predictions
        """
        # Extract mean prediction
        if point_forecast.ndim == 2:
            mean_pred = point_forecast[0][:forecast_horizon]
        else:
            mean_pred = point_forecast[:forecast_horizon]
        
        # Extract quantiles
        quantile_preds = {}
        if quantile_forecast is not None and quantile_forecast.size > 0:
            # quantile_forecast shape: (batch, horizon, num_quantiles) or (horizon, num_quantiles)
            if quantile_forecast.ndim == 3:
                q_values = quantile_forecast[0]  # Shape: (horizon, num_quantiles)
            elif quantile_forecast.ndim == 2:
                q_values = quantile_forecast  # Shape: (horizon, num_quantiles)
            else:
                q_values = quantile_forecast
            
            # Map quantiles - assuming they match the order in config
            # TimesFM typically returns 10 quantiles: [0.1, 0.2, ..., 0.9]
            default_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            for i, q in enumerate(quantiles):
                if i < q_values.shape[-1]:
                    quantile_preds[q] = q_values[:forecast_horizon, i]
                else:
                    # Fallback to mean if quantile index out of range
                    quantile_preds[q] = mean_pred.copy()
        else:
            # If no quantiles available, use mean as placeholder
            logger.warning("Quantiles not available, using mean as placeholder")
            for q in quantiles:
                quantile_preds[q] = mean_pred.copy()
        
        # Get prediction intervals
        lower_bound, upper_bound = self.get_prediction_intervals(quantile_preds)
        
        result = {
            'mean': mean_pred,
            'quantiles': quantile_preds,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_path": self.config.model_path,
            "context_length": self.config.context_length,
            "horizon_length": self.config.horizon_length,
            "patch_length": self.config.patch_length,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
            "description": "TimesFM-2.5: Time Series Foundation Model by Google"
        }
        
        return info
