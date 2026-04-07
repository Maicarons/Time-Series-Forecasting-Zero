"""Configuration management for the forecasting framework.

Supports three usage modes:
1. Direct execution: Parameters from function arguments
2. CLI: Parameters from CLI arguments or INI file
3. PyPI package: Parameters from function arguments
"""

import os
import configparser
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    model_path: str = ""
    device: str = "cuda"
    forecast_horizon: int = 128
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    batch_size: int = 32
    
    def __post_init__(self):
        # Convert model_path to absolute path if provided
        if self.model_path and not os.path.isabs(self.model_path):
            self.model_path = str(Path(self.model_path).resolve())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Chronos2Config(ModelConfig):
    """Configuration specific to Chronos-2 model."""
    model_path: str = ""
    context_length: int = 8192
    input_patch_size: int = 16
    output_patch_size: int = 16
    max_output_patches: int = 64
    use_arcsinh: bool = True
    temperature: float = 1.0
    num_samples: int = 20
    
    def __post_init__(self):
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = os.getenv("CHRONOS2_MODEL_PATH", "./models/chronos-2")
        super().__post_init__()


@dataclass
class TimesFMConfig(ModelConfig):
    """Configuration specific to TimesFM-2.5 model."""
    model_path: str = ""
    context_length: int = 16384
    patch_length: int = 32
    horizon_length: int = 128
    freq: str = "H"  # Frequency: H=hourly, D=daily, W=weekly, M=monthly
    normalize: bool = True
    
    def __post_init__(self):
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = os.getenv("TIMESFM_MODEL_PATH", "./models/timesfm-2.5-200m-pytorch")
        super().__post_init__()


@dataclass
class TiRexConfig(ModelConfig):
    """Configuration specific to TiRex model."""
    model_name: str = "NX-AI/TiRex"  # HuggingFace model ID
    model_path: str = ""  # Not used for TiRex, kept for compatibility
    device: str = "cuda"
    forecast_horizon: int = 128
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    batch_size: int = 32
    context_length: int = 512  # Default context length for TiRex


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = ""
    output_dir: str = ""
    time_column: str = "timestamp"
    value_column: str = "value"
    freq: str = "H"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
        # Set defaults if not provided
        if not self.data_dir:
            self.data_dir = os.getenv("DATA_DIR", "./data")
        if not self.output_dir:
            self.output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        
        # Create directories if they don't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "logs")).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "./outputs/logs/app.log")
    )
    console_output: bool = True
    file_output: bool = True


def get_chronos2_config(**kwargs) -> Chronos2Config:
    """Get Chronos-2 configuration with optional overrides."""
    config = Chronos2Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_timesfm_config(**kwargs) -> TimesFMConfig:
    """Get TimesFM configuration with optional overrides."""
    config = TimesFMConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_tirex_config(**kwargs) -> TiRexConfig:
    """Get TiRex configuration with optional overrides."""
    config = TiRexConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_data_config(**kwargs) -> DataConfig:
    """Get data configuration with optional overrides."""
    config = DataConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_logging_config(**kwargs) -> LoggingConfig:
    """Get logging configuration with optional overrides."""
    config = LoggingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def load_config_from_ini(ini_path: str, section: str = "DEFAULT") -> Dict[str, Any]:
    """
    Load configuration from INI file.
    
    Args:
        ini_path: Path to INI file
        section: Section name in INI file
        
    Returns:
        Dictionary of configuration values
        
    Example INI file:
        [DEFAULT]
        model_name = chronos2
        forecast_horizon = 128
        device = cuda
        
        [chronos2]
        model_path = ./models/chronos-2
        context_length = 8192
        
        [data]
        data_dir = ./data
        time_column = timestamp
        value_column = value
    """
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    result = {}
    if section in config:
        result.update(dict(config[section]))
    
    # Also merge DEFAULT section
    if "DEFAULT" in config and section != "DEFAULT":
        result.update(dict(config["DEFAULT"]))
    
    # Convert types
    for key, value in result.items():
        # Try to convert to int
        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ['true', 'yes', '1']:
            result[key] = True
        elif value.lower() in ['false', 'no', '0']:
            result[key] = False
    
    return result
