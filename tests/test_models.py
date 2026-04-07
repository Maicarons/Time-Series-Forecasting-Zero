"""
Unit tests for model interfaces.

Tests individual model loading and prediction functionality.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.chronos2 import Chronos2Forecaster
from src.models.timesfm import TimesFMForecaster
from src.models.tirex import TiRexForecaster
from src.configs.config import (
    get_chronos2_config,
    get_timesfm_config,
    get_tirex_config
)


class TestChronos2(unittest.TestCase):
    """Test Chronos-2 model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.config = get_chronos2_config(device=cls.device)
        cls.model = Chronos2Forecaster(cls.config)
        
        # Generate test data
        np.random.seed(42)
        cls.train_data = np.sin(np.linspace(0, 10, 350)) * 10 + 50 + np.random.randn(350) * 2
        cls.test_data = np.sin(np.linspace(10, 12, 48)) * 10 + 50 + np.random.randn(48) * 2
    
    def test_model_loading(self):
        """Test model loads successfully."""
        self.model.load_model()
        self.assertTrue(self.model.is_loaded)
    
    def test_prediction(self):
        """Test prediction returns correct format."""
        self.model.load_model()
        predictions = self.model.predict(self.train_data, forecast_horizon=48)
        
        # Check output format
        self.assertIn('mean', predictions)
        self.assertIn('quantiles', predictions)
        self.assertIn('lower_bound', predictions)
        self.assertIn('upper_bound', predictions)
        
        # Check shapes
        self.assertEqual(predictions['mean'].shape, (48,))
    
    def test_prediction_values_reasonable(self):
        """Test prediction values are in reasonable range."""
        self.model.load_model()
        predictions = self.model.predict(self.train_data, forecast_horizon=48)
        
        # Predictions should be in similar range as training data
        train_mean = np.mean(self.train_data)
        train_std = np.std(self.train_data)
        
        pred_mean = np.mean(predictions['mean'])
        self.assertLess(abs(pred_mean - train_mean), 3 * train_std)


class TestTimesFM(unittest.TestCase):
    """Test TimesFM-2.5 model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.config = get_timesfm_config(device=cls.device)
        cls.model = TimesFMForecaster(cls.config)
        
        # Generate test data
        np.random.seed(42)
        cls.train_data = np.sin(np.linspace(0, 10, 350)) * 10 + 50 + np.random.randn(350) * 2
    
    def test_model_loading(self):
        """Test model loads successfully."""
        self.model.load_model()
        self.assertTrue(self.model.is_loaded)
    
    def test_prediction(self):
        """Test prediction returns correct format."""
        self.model.load_model()
        predictions = self.model.predict(self.train_data, forecast_horizon=48)
        
        # Check output format
        self.assertIn('mean', predictions)
        self.assertIn('quantiles', predictions)
        
        # Check shapes
        self.assertEqual(predictions['mean'].shape, (48,))


class TestTiRex(unittest.TestCase):
    """Test TiRex model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.config = get_tirex_config(device=cls.device)
        cls.model = TiRexForecaster(cls.config)
        
        # Generate test data
        np.random.seed(42)
        cls.train_data = np.sin(np.linspace(0, 10, 350)) * 10 + 50 + np.random.randn(350) * 2
    
    def test_model_loading(self):
        """Test model loads successfully."""
        self.model.load_model()
        self.assertTrue(self.model.is_loaded)
    
    def test_prediction(self):
        """Test prediction returns correct format."""
        self.model.load_model()
        predictions = self.model.predict(self.train_data, forecast_horizon=48)
        
        # Check output format
        self.assertIn('mean', predictions)
        self.assertIn('quantiles', predictions)
        
        # Check shapes
        self.assertEqual(predictions['mean'].shape, (48,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
