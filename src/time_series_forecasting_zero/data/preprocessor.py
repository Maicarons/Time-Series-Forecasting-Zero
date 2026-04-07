"""Data preprocessing utilities for time series."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger


class TimeSeriesPreprocessor:
    """Preprocess time series data for forecasting models."""
    
    def __init__(
        self,
        scaler_type: str = "standard",
        normalize: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaling ('standard', 'minmax', or 'none')
            normalize: Whether to apply normalization
        """
        self.scaler_type = scaler_type
        self.normalize = normalize
        self.scaler = None
        self.is_fitted = False
        
        if normalize and scaler_type != "none":
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        logger.info(f"TimeSeriesPreprocessor initialized with scaler_type={scaler_type}")
    
    def fit(self, data: np.ndarray) -> 'TimeSeriesPreprocessor':
        """
        Fit the scaler on training data.
        
        Args:
            data: 1D or 2D array of time series values
            
        Returns:
            Self for method chaining
        """
        if not self.normalize or self.scaler is None:
            logger.info("Skipping normalization")
            self.is_fitted = True
            return self
        
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.scaler.fit(data)
        self.is_fitted = True
        
        logger.info("Scaler fitted on training data")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted scaler.
        
        Args:
            data: 1D or 2D array of time series values
            
        Returns:
            Transformed data
        """
        if not self.normalize or self.scaler is None:
            logger.debug("Skipping transformation")
            return data
        
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Reshape if 1D
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        transformed = self.scaler.transform(data)
        
        # Restore original shape
        if len(original_shape) == 1:
            transformed = transformed.flatten()
        
        logger.debug(f"Transformed data with shape {transformed.shape}")
        
        return transformed
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            data: 1D or 2D array of time series values
            
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: 1D or 2D array of scaled values
            
        Returns:
            Data in original scale
        """
        if not self.normalize or self.scaler is None:
            logger.debug("Skipping inverse transformation")
            return data
        
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Reshape if 1D
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        inverse_transformed = self.scaler.inverse_transform(data)
        
        # Restore original shape
        if len(original_shape) == 1:
            inverse_transformed = inverse_transformed.flatten()
        
        logger.debug(f"Inverse transformed data with shape {inverse_transformed.shape}")
        
        return inverse_transformed
    
    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int,
        forecast_horizon: int,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for model training/prediction.
        
        Args:
            data: 1D array of time series values
            seq_length: Length of input sequence (lookback window)
            forecast_horizon: Number of steps to predict ahead
            stride: Step size between sequences
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        if len(data) < seq_length + forecast_horizon:
            raise ValueError(
                f"Data length ({len(data)}) is less than required "
                f"(seq_length={seq_length} + forecast_horizon={forecast_horizon})"
            )
        
        sequences_x = []
        sequences_y = []
        
        for i in range(0, len(data) - seq_length - forecast_horizon + 1, stride):
            x = data[i:i + seq_length]
            y = data[i + seq_length:i + seq_length + forecast_horizon]
            
            sequences_x.append(x)
            sequences_y.append(y)
        
        X = np.array(sequences_x)
        Y = np.array(sequences_y)
        
        logger.info(f"Created {len(X)} sequences with shape X={X.shape}, Y={Y.shape}")
        
        return X, Y
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "interpolate"
    ) -> pd.DataFrame:
        """
        Handle missing values in time series data.
        
        Args:
            data: DataFrame with time series data
            method: Method for handling missing values 
                   ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        
        Returns:
            DataFrame with handled missing values
        """
        n_missing = data.isnull().sum().sum()
        if n_missing == 0:
            logger.info("No missing values found")
            return data
        
        logger.info(f"Handling {n_missing} missing values using method: {method}")
        
        if method == "interpolate":
            # Linear interpolation (works well for time series)
            data = data.interpolate(method='linear')
        elif method == "forward_fill":
            data = data.ffill()
        elif method == "backward_fill":
            data = data.bfill()
        elif method == "drop":
            data = data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Drop any remaining NaN at boundaries
        data = data.dropna()
        
        n_remaining = data.isnull().sum().sum()
        logger.info(f"Missing values after handling: {n_remaining}")
        
        return data
    
    def detect_outliers(
        self,
        data: np.ndarray,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outliers in time series data.
        
        Args:
            data: 1D array of time series values
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        if method == "zscore":
            # Z-score method
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)
            outliers = z_scores > threshold
            
        elif method == "iqr":
            # IQR method
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_outliers = np.sum(outliers)
        logger.info(f"Detected {n_outliers} outliers using {method} method")
        
        return outliers
    
    def remove_outliers(
        self,
        data: np.ndarray,
        method: str = "zscore",
        threshold: float = 3.0,
        replacement: str = "interpolate"
    ) -> np.ndarray:
        """
        Remove outliers from time series data.
        
        Args:
            data: 1D array of time series values
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            replacement: How to replace outliers ('interpolate', 'mean', 'median')
            
        Returns:
            Cleaned data array
        """
        outliers = self.detect_outliers(data, method, threshold)
        
        if not np.any(outliers):
            logger.info("No outliers to remove")
            return data
        
        cleaned_data = data.copy()
        
        if replacement == "interpolate":
            # Replace outliers with interpolated values
            cleaned_data[outliers] = np.nan
            indices = np.arange(len(cleaned_data))
            mask = ~np.isnan(cleaned_data)
            cleaned_data = np.interp(indices, indices[mask], cleaned_data[mask])
        
        elif replacement == "mean":
            mean_val = np.mean(data[~outliers])
            cleaned_data[outliers] = mean_val
        
        elif replacement == "median":
            median_val = np.median(data[~outliers])
            cleaned_data[outliers] = median_val
        
        else:
            raise ValueError(f"Unknown replacement method: {replacement}")
        
        logger.info(f"Removed {np.sum(outliers)} outliers using {replacement} replacement")
        
        return cleaned_data
