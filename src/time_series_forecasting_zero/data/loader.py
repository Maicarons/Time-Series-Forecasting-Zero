"""Data loading utilities for time series data."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger


class TimeSeriesDataLoader:
    """Load and manage time series data from various sources."""
    
    def __init__(
        self,
        data_dir: str = "./data",
        time_column: str = "timestamp",
        value_column: str = "value",
        freq: str = "H"
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
            time_column: Name of the timestamp column
            value_column: Name of the value column
            freq: Frequency of the time series (H, D, W, M, etc.)
        """
        self.data_dir = Path(data_dir)
        self.time_column = time_column
        self.value_column = value_column
        self.freq = freq
        self.data = None
        
        logger.info(f"TimeSeriesDataLoader initialized with data_dir={data_dir}")
    
    def load_csv(
        self,
        filename: str,
        time_column: Optional[str] = None,
        value_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load time series data from a CSV file.
        
        Args:
            filename: Name of the CSV file
            time_column: Override default time column name
            value_column: Override default value column name
            
        Returns:
            DataFrame with parsed timestamps
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Use provided column names or defaults
        time_col = time_column or self.time_column
        value_col = value_column or self.value_column
        
        # Parse timestamp
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            df.index.name = 'timestamp'
        else:
            raise ValueError(f"Time column '{time_col}' not found in data")
        
        # Ensure value column exists
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in data")
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(df)} records from {filename}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        self.data = df
        return df
    
    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        time_column: Optional[str] = None,
        value_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load time series data from an existing DataFrame.
        
        Args:
            df: Input DataFrame
            time_column: Name of the timestamp column
            value_column: Name of the value column
            
        Returns:
            Processed DataFrame with timestamp index
        """
        time_col = time_column or self.time_column
        value_col = value_column or self.value_column
        
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Parse and set timestamp as index
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df.index.name = 'timestamp'
        
        # Sort and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(df)} records from DataFrame")
        
        self.data = df
        return df
    
    def train_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_from_dataframe() first.")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(self.data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = self.data.iloc[:train_end]
        val_df = self.data.iloc[train_end:val_end]
        test_df = self.data.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_values(self) -> np.ndarray:
        """
        Get the time series values as a numpy array.
        
        Returns:
            Numpy array of values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_from_dataframe() first.")
        
        return self.data[self.value_column].values
    
    def get_timestamps(self) -> pd.DatetimeIndex:
        """
        Get the timestamps.
        
        Returns:
            DatetimeIndex of timestamps
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_from_dataframe() first.")
        
        return self.data.index
    
    def resample(self, freq: str) -> pd.DataFrame:
        """
        Resample the time series to a different frequency.
        
        Args:
            freq: Target frequency (H, D, W, M, etc.)
            
        Returns:
            Resampled DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_from_dataframe() first.")
        
        logger.info(f"Resampling data to frequency: {freq}")
        
        # Use mean aggregation for numerical values
        resampled = self.data.resample(freq).mean()
        
        # Drop rows with all NaN
        resampled = resampled.dropna(subset=[self.value_column])
        
        self.data = resampled
        self.freq = freq
        
        logger.info(f"Resampled to {len(resampled)} records")
        
        return resampled
    
    def save_processed_data(
        self,
        output_path: str,
        filename: str = "processed_data.csv"
    ) -> None:
        """
        Save processed data to CSV.
        
        Args:
            output_path: Directory to save the file
            filename: Output filename
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() or load_from_dataframe() first.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        self.data.to_csv(filepath)
        
        logger.info(f"Saved processed data to {filepath}")
