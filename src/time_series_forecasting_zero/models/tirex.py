"""TiRex model interface for time series forecasting."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from loguru import logger

from .base import BaseForecaster
from ..configs.config import TiRexConfig


class TiRexForecaster(BaseForecaster):
    """Forecasting interface for TiRex model."""
    
    def __init__(self, config: Optional[TiRexConfig] = None):
        """
        Initialize TiRex forecaster.
        
        Args:
            config: Configuration for TiRex model
        """
        self.config = config or TiRexConfig()
        super().__init__(model_name="TiRex", config=self.config)
    
    def load_model(self) -> None:
        """Load the TiRex model using official load_model."""
        if self.is_loaded:
            logger.info("TiRex model already loaded")
            return
        
        # Determine model source: local path or HuggingFace ID
        import os
        from pathlib import Path
        
        # Check if model_path is provided and exists (local model)
        if self.config.model_path and Path(self.config.model_path).exists():
            model_source = self.config.model_path
            logger.info(f"Loading TiRex model from local path: {model_source}")
        else:
            # Use HuggingFace model ID
            model_source = self.config.model_name
            logger.info(f"Loading TiRex model from HuggingFace: {model_source}")
        
        try:
            # Import official TiRex
            from tirex import load_model, ForecastModel
            
            # Load model using official API
            # load_model supports both HF model ID and local path
            self.model: ForecastModel = load_model(model_source)
            
            self.is_loaded = True
            logger.info(f"TiRex model loaded successfully on {self.config.device}")
            
        except ImportError as e:
            logger.error(
                f"Failed to import tirex package. Please install it with:\n"
                f"  pip install tirex xlstm\n"
                f"Original error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load TiRex model: {e}")
            raise
    
    def predict(
        self,
        context: Union[np.ndarray, pd.Series, pd.DataFrame],
        forecast_horizon: Optional[int] = None,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using TiRex.
        
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
            # Prepare input tensor
            # TiRex expects input shape: (batch_size, seq_len)
            context_tensor = torch.tensor(
                context_values,
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
            # Move to device
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.config.device
            if isinstance(device, str):
                device = torch.device(device)
            context_tensor = context_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                forecast = self.model.forecast(
                    context=context_tensor,
                    prediction_length=forecast_horizon,
                    output_type='torch'  # Return as torch.Tensor [batch, horizon, quantiles]
                )
            
            # Process outputs
            predictions = self._process_outputs(forecast, quantiles, forecast_horizon)
            
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
            # Find max sequence length for padding
            max_seq_len = max(len(ctx) for ctx in context_values)
            
            # Pad sequences and create batch
            padded_contexts = []
            for ctx in context_values:
                pad_length = max_seq_len - len(ctx)
                if pad_length > 0:
                    padded = np.pad(ctx, (pad_length, 0), mode='constant')
                else:
                    padded = ctx
                padded_contexts.append(padded)
            
            # Convert to tensor
            context_tensor = torch.tensor(
                np.array(padded_contexts),
                dtype=torch.float32
            )
            
            # Move to device
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.config.device
            if isinstance(device, str):
                device = torch.device(device)
            context_tensor = context_tensor.to(device)
            
            # Make batch prediction
            with torch.no_grad():
                forecasts = self.model.forecast(
                    context=context_tensor,
                    prediction_length=forecast_horizon
                )
            
            # Process all outputs
            predictions = []
            for i in range(len(context_values)):
                # Extract single sample from batch
                if isinstance(forecasts, dict):
                    single_forecast = {
                        key: val[i:i+1] if isinstance(val, torch.Tensor) else val
                        for key, val in forecasts.items()
                    }
                elif isinstance(forecasts, (list, tuple)):
                    single_forecast = forecasts[i]
                else:
                    single_forecast = forecasts[i:i+1] if isinstance(forecasts, torch.Tensor) else forecasts
                
                pred = self._process_single_output(single_forecast, quantiles, forecast_horizon)
                predictions.append(pred)
            
            logger.info("Batch prediction completed")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _process_outputs(
        self,
        forecast,
        quantiles: List[float],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Process model outputs into prediction format.
        
        Args:
            forecast: Raw model forecast output
            quantiles: Requested quantile levels
            forecast_horizon: Forecast horizon length
            
        Returns:
            Dictionary of processed predictions
        """
        return self._process_single_output(forecast, quantiles, forecast_horizon)
    
    def _process_single_output(
        self,
        forecast,
        quantiles: List[float],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """
        Process a single forecast output.
        
        Args:
            forecast: Single forecast object
            quantiles: Requested quantile levels
            forecast_horizon: Forecast horizon length
            
        Returns:
            Dictionary of processed predictions
        """
        # Extract predictions based on output type
        if isinstance(forecast, tuple):
            # TiRex returns (quantiles_tensor, mean_tensor)
            # quantiles shape: (batch, horizon, num_quantiles)
            # mean shape: (batch, horizon)
            if len(forecast) == 2:
                quantiles_tensor, mean_tensor = forecast
                
                # Extract mean
                if isinstance(mean_tensor, torch.Tensor):
                    if mean_tensor.ndim == 2:
                        mean_pred = mean_tensor[0]  # Take first batch
                    else:
                        mean_pred = mean_tensor
                else:
                    mean_pred = mean_tensor
                
                # Extract quantiles
                quantile_preds = {}
                if isinstance(quantiles_tensor, torch.Tensor):
                    if quantiles_tensor.ndim == 3:
                        q_single = quantiles_tensor[0]  # Shape: (horizon, num_q)
                        num_q = q_single.shape[-1]
                        default_quantiles = np.linspace(0.1, 0.9, num_q)
                        
                        for i, q in enumerate(quantiles):
                            closest_idx = np.argmin(np.abs(default_quantiles - q))
                            if closest_idx < num_q:
                                quantile_preds[q] = q_single[:, closest_idx]
                            else:
                                quantile_preds[q] = mean_pred.clone()
                    else:
                        quantile_preds = {q: mean_pred.clone() for q in quantiles}
                else:
                    quantile_preds = {q: mean_pred.clone() for q in quantiles}
            else:
                raise ValueError(f"Unexpected tuple length: {len(forecast)}")
        
        elif isinstance(forecast, dict):
            # Dictionary output with mean and quantiles
            if 'mean' in forecast:
                mean_pred = forecast['mean']
            elif 'predictions' in forecast:
                mean_pred = forecast['predictions']
            else:
                raise ValueError(f"Unknown output keys: {forecast.keys()}")
            
            # Extract quantiles if available
            quantile_preds = {}
            if 'quantiles' in forecast:
                for q in quantiles:
                    if q in forecast['quantiles']:
                        quantile_preds[q] = forecast['quantiles'][q]
            elif 'q' in forecast:
                # Alternative quantile format
                for q in quantiles:
                    quantile_preds[q] = forecast['q'].get(q, mean_pred)
            else:
                # If no quantiles, use mean as placeholder
                logger.warning("Quantiles not available, using mean as placeholder")
                for q in quantiles:
                    quantile_preds[q] = mean_pred.copy()
        
        elif isinstance(forecast, torch.Tensor):
            # Tensor output - TiRex returns [batch, horizon, num_quantiles]
            if forecast.ndim == 3:
                # Shape: (batch_size, horizon, num_quantiles)
                # Take first batch
                forecast_single = forecast[0]  # Shape: (horizon, num_quantiles)
                
                # Median (0.5 quantile) is typically at index 4 or 5
                # Assuming 10 quantiles: [0.1, 0.2, ..., 0.9]
                num_q = forecast_single.shape[-1]
                median_idx = num_q // 2  # Middle index
                
                mean_pred = forecast_single[:, median_idx]  # Use median as mean
                
                # Extract quantiles
                quantile_preds = {}
                default_quantiles = np.linspace(0.1, 0.9, num_q)
                for i, q in enumerate(quantiles):
                    # Find closest quantile index
                    closest_idx = np.argmin(np.abs(default_quantiles - q))
                    if closest_idx < num_q:
                        quantile_preds[q] = forecast_single[:, closest_idx]
                    else:
                        quantile_preds[q] = mean_pred.clone()
            else:
                # Simple tensor - assume it's the mean prediction
                mean_pred = forecast
                quantile_preds = {q: mean_pred.clone() for q in quantiles}
        
        elif hasattr(forecast, 'mean'):
            # Object with mean attribute
            mean_pred = forecast.mean
            if hasattr(forecast, 'quantiles'):
                quantile_preds = {}
                for q in quantiles:
                    if q in forecast.quantiles:
                        quantile_preds[q] = forecast.quantiles[q]
            else:
                quantile_preds = {q: mean_pred.clone() for q in quantiles}
        
        else:
            raise ValueError(f"Unexpected forecast type: {type(forecast)}")
        
        # Convert to numpy
        if isinstance(mean_pred, torch.Tensor):
            mean_pred = mean_pred.cpu().numpy()
        
        # Ensure correct shape
        if mean_pred.ndim > 1:
            mean_pred = mean_pred.squeeze()
        
        # Ensure correct horizon length
        mean_pred = mean_pred[:forecast_horizon]
        
        # Process quantiles
        final_quantiles = {}
        for q, q_val in quantile_preds.items():
            if isinstance(q_val, torch.Tensor):
                q_val = q_val.cpu().numpy()
            if q_val.ndim > 1:
                q_val = q_val.squeeze()
            final_quantiles[q] = q_val[:forecast_horizon]
        
        # Get prediction intervals
        lower_bound, upper_bound = self.get_prediction_intervals(final_quantiles)
        
        result = {
            'mean': mean_pred,
            'quantiles': final_quantiles,
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
            "model_id": self.config.model_name,
            "context_length": self.config.context_length,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
            "description": "TiRex: Zero-Shot Forecasting with xLSTM architecture"
        }
        
        return info
