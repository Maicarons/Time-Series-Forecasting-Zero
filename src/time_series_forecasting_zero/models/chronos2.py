"""Chronos-2 model interface for time series forecasting.

Official API: https://huggingface.co/amazon/chronos-2
Uses Chronos2Pipeline from chronos-forecasting package.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from loguru import logger

from .base import BaseForecaster
from ..configs.config import Chronos2Config


class Chronos2Forecaster(BaseForecaster):
    """Forecasting interface for Chronos-2 using official Chronos2Pipeline."""
    
    def __init__(self, config: Optional[Chronos2Config] = None):
        """
        Initialize Chronos-2 forecaster.
        
        Args:
            config: Configuration for Chronos-2 model
        """
        self.config = config or Chronos2Config()
        super().__init__(model_name="Chronos-2", config=self.config)
        self.pipeline = None
    
    def load_model(self) -> None:
        """Load the Chronos-2 model using official Chronos2Pipeline."""
        if self.is_loaded:
            logger.info("Chronos-2 model already loaded")
            return
        
        logger.info(f"Loading Chronos-2 model from {self.config.model_path}")
        
        try:
            # Import official Chronos2Pipeline
            from chronos import Chronos2Pipeline
            
            # Load model using official pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.config.model_path,
                device_map=self.config.device if self.config.device == "cuda" else "cpu"
            )
            
            self.is_loaded = True
            logger.info(f"Chronos-2 model loaded successfully on {self.config.device}")
            
        except ImportError as e:
            logger.error(
                f"Failed to import chronos package. Please install it with:\n"
                f"  pip install chronos-forecasting\n"
                f"Original error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Chronos-2 model: {e}")
            raise
    
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using Chronos-2 official pipeline API.
        
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
            # Convert to tensor - Chronos expects 3D shape: (n_series, n_variates, seq_len)
            # For univariate time series: (1, 1, seq_len)
            context_tensor = torch.tensor(
                context_values,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len)
            
            # Generate forecasts using official pipeline API
            # Chronos2Pipeline.predict returns samples
            with torch.no_grad():
                samples = self.pipeline.predict(
                    inputs=context_tensor,  # Use 'inputs' parameter
                    prediction_length=forecast_horizon,
                    # Note: num_samples, temperature, top_k not supported in this version
                )
            
            # Process outputs - samples is a list of tensors
            # Each tensor shape: (batch_size, num_samples, prediction_length)
            if isinstance(samples, list) and len(samples) > 0:
                samples_tensor = samples[0]  # Take first element
            else:
                samples_tensor = samples
            
            predictions = self._process_outputs(samples_tensor, quantiles, forecast_horizon)
            
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
            # Stack all contexts into a single batch
            # Pad sequences to the same length
            max_len = max(len(ctx) for ctx in context_values)
            padded_contexts = []
            for ctx in context_values:
                pad_length = max_len - len(ctx)
                if pad_length > 0:
                    padded = np.pad(ctx, (pad_length, 0), mode='constant')
                else:
                    padded = ctx
                padded_contexts.append(padded)
            
            # Create batch tensor
            context_tensor = torch.tensor(
                np.array(padded_contexts),
                dtype=torch.float32
            )
            
            # Make batch prediction
            with torch.no_grad():
                samples = self.pipeline.predict(
                    inputs=context_tensor,
                    prediction_length=forecast_horizon,
                )
            
            # Process all outputs
            predictions = []
            for i in range(len(context_values)):
                single_samples = samples[i:i+1]  # Keep batch dimension
                pred = self._process_outputs(single_samples, quantiles, forecast_horizon)
                predictions.append(pred)
            
            logger.info("Batch prediction completed")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _process_outputs(
        self,
        samples: torch.Tensor,
        quantiles: List[float],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Process model outputs into prediction format.
        
        Args:
            samples: Model output samples, shape: (batch_size, num_samples, prediction_length)
            quantiles: Requested quantile levels
            forecast_horizon: Forecast horizon length
            
        Returns:
            Dictionary of processed predictions
        """
        # Convert to numpy
        if isinstance(samples, torch.Tensor):
            samples_np = samples.cpu().numpy()
        else:
            samples_np = samples
        
        # Take first item from batch
        if samples_np.ndim == 3:
            samples_np = samples_np[0]  # Shape: (num_samples, prediction_length)
        elif samples_np.ndim == 2:
            # If only 2D, add sample dimension
            samples_np = samples_np[np.newaxis, :]
        
        # Calculate mean (average of samples)
        mean_pred = np.mean(samples_np, axis=0)
        
        # Calculate quantiles
        quantile_preds = {}
        for q in quantiles:
            quantile_preds[q] = np.percentile(samples_np, q * 100, axis=0)
        
        # Get prediction intervals
        lower_bound, upper_bound = self.get_prediction_intervals(quantile_preds)
        
        result = {
            'mean': mean_pred[:forecast_horizon],
            'quantiles': {q: quantile_preds[q][:forecast_horizon] for q in quantiles},
            'lower_bound': lower_bound[:forecast_horizon] if lower_bound is not None else None,
            'upper_bound': upper_bound[:forecast_horizon] if upper_bound is not None else None
        }
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded or self.pipeline is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_path": self.config.model_path,
            "context_length": self.config.context_length,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
            "description": "Chronos-2: Pre-trained time series forecasting model by Amazon"
        }
        
        return info
