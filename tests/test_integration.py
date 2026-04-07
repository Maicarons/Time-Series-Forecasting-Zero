"""
End-to-end integration tests.

Tests complete workflows including data loading, preprocessing, forecasting, and evaluation.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.unified import UnifiedForecaster
from src.data.loader import TimeSeriesDataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.utils.evaluator import MetricsEvaluator


class TestEndToEnd(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Generate synthetic dataset
        np.random.seed(42)
        n_points = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
        t = np.arange(n_points)
        trend = 0.01 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
        noise = np.random.normal(0, 2, n_points)
        
        values = 50 + trend + seasonal + noise
        
        cls.df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        # Split data
        cls.train_data = values[:800]
        cls.test_data = values[800:]
        cls.forecast_horizon = len(cls.test_data)
    
    def test_unified_forecaster_workflow(self):
        """Test complete workflow with UnifiedForecaster."""
        
        # Initialize forecaster
        forecaster = UnifiedForecaster(
            model_name="chronos2",
            forecast_horizon=self.forecast_horizon,
            device=self.device
        )
        
        # Load model
        forecaster.load_model()
        
        # Make prediction
        predictions = forecaster.predict(
            context=self.train_data,
            forecast_horizon=self.forecast_horizon,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        # Verify output format
        self.assertIn('mean', predictions)
        self.assertIn('quantiles', predictions)
        self.assertEqual(predictions['mean'].shape, (self.forecast_horizon,))
        
        # Evaluate
        evaluator = MetricsEvaluator()
        metrics = evaluator.evaluate(self.test_data, predictions['mean'])
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_batch_prediction(self):
        """Test batch prediction for multiple time series."""
        
        # Create multiple time series
        contexts = [
            self.train_data,
            self.train_data * 1.1,  # Slightly different series
            self.train_data * 0.9
        ]
        
        forecaster = UnifiedForecaster(
            model_name="tirex",
            forecast_horizon=48,
            device=self.device
        )
        forecaster.load_model()
        
        # Batch predict
        predictions_list = forecaster.batch_predict(
            contexts=contexts,
            forecast_horizon=48
        )
        
        # Verify
        self.assertEqual(len(predictions_list), len(contexts))
        for pred in predictions_list:
            self.assertIn('mean', pred)
            self.assertEqual(pred['mean'].shape, (48,))
    
    def test_data_loader_and_preprocessor(self):
        """Test data loading and preprocessing pipeline."""
        
        # Save test data to CSV
        csv_path = Path(__file__).parent.parent / "test_data.csv"
        self.df.to_csv(csv_path, index=False)
        
        try:
            # Load data
            loader = TimeSeriesDataLoader(data_dir=str(Path(__file__).parent.parent))
            df_loaded = loader.load_csv("test_data.csv")
            
            self.assertEqual(len(df_loaded), len(self.df))
            self.assertIn('timestamp', df_loaded.columns)
            self.assertIn('value', df_loaded.columns)
            
            # Preprocess
            preprocessor = TimeSeriesPreprocessor()
            processed = preprocessor.fit_transform(df_loaded, value_column='value')
            
            self.assertIsNotNone(processed)
            self.assertEqual(len(processed), len(df_loaded))
            
        finally:
            # Clean up
            if csv_path.exists():
                csv_path.unlink()
    
    def test_all_models_consistency(self):
        """Test that all models produce consistent output format."""
        
        models = ["chronos2", "timesfm", "tirex"]
        required_keys = ['mean', 'quantiles', 'lower_bound', 'upper_bound']
        
        for model_name in models:
            with self.subTest(model=model_name):
                forecaster = UnifiedForecaster(
                    model_name=model_name,
                    forecast_horizon=48,
                    device=self.device
                )
                forecaster.load_model()
                
                predictions = forecaster.predict(
                    context=self.train_data[:350],  # Use subset for speed
                    forecast_horizon=48,
                    quantiles=[0.1, 0.5, 0.9]
                )
                
                # Check all required keys present
                for key in required_keys:
                    self.assertIn(key, predictions, 
                                f"{model_name} missing key: {key}")
                
                # Check shapes
                self.assertEqual(predictions['mean'].shape, (48,))
                self.assertIsInstance(predictions['quantiles'], dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
