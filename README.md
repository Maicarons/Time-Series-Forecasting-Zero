# Time-Series-Forecasting-Zero

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A production-ready time series forecasting framework supporting state-of-the-art foundation models with zero-shot prediction capabilities.

**Three Usage Modes:**
1. 📝 **Examples**: Run example scripts directly
2. 💻 **CLI**: Use command-line tool `tsforecast`  
3. 📦 **PyPI Package**: Import as a Python library

## 🚀 Features

- **Multiple Foundation Models**: Chronos-2 (Amazon), TimesFM-2.5 (Google), TiRex (NX-AI)
- **Zero-Shot Forecasting**: Make predictions without training on your data
- **Built-in Evaluation Tools**: RMSE, MAE, MAPE, Coverage metrics + visualization
- **Flexible Configuration**: CLI args, INI files, environment variables, or code parameters
- **Unified Interface**: Simple API for all models
- **Probabilistic Forecasts**: Prediction intervals and quantiles
- **Batch Processing**: Forecast multiple time series simultaneously
- **CUDA Acceleration**: Automatic GPU detection
- **Local Model Support**: Use downloaded models or HuggingFace

## 📦 Supported Models

| Model | Organization | Parameters | Best For | Local Path |
|-------|-------------|------------|----------|------------|
| **Chronos-2** | Amazon | 120M | General purpose | `./models/chronos-2/` |
| **TimesFM-2.5** | Google | 200M | Long sequences | `./models/timesfm-2.5-200m-pytorch/` |
| **TiRex** | NX-AI | 35M | Fast inference | `./models/tirex-model/` |

## 🛠️ Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/Time-Series-Forecasting-Zero.git
cd Time-Series-Forecasting-Zero

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package (enables CLI)
pip install -e .
```

### Download Models

Models are stored in `models/` directory:

```bash
# Option 1: Git submodules
git submodule update --init --recursive

# Option 2: Manual download from HuggingFace/ModelScope
# Place in: models/chronos-2/, models/timesfm-2.5-200m-pytorch/, models/tirex-model/
```

## 🚦 Quick Start

### Mode 1: Examples (Learning)

```bash
# Basic forecasting
python examples/01_basic_forecasting.py

# Specify model paths
python examples/01_basic_forecasting.py \
    --chronos2-path ./models/chronos-2 \
    --timesfm-path ./models/timesfm-2.5

# Load from CSV
python examples/02_load_from_csv.py

# Batch forecasting
python examples/03_batch_forecasting.py

# Model comparison
python examples/04_model_comparison.py

# Forecast utilities (metrics + visualization)
python examples/05_forecast_utilities.py
```

### Mode 2: CLI Tool (Daily Use)

```bash
# Single model prediction
tsforecast predict \
    --model chronos2 \
    --data data/test_data.csv \
    --horizon 128 \
    --device cuda

# Specify model path
tsforecast predict \
    --model timesfm \
    --model-path ./models/timesfm-2.5 \
    --data data/test.csv

# Use configuration file
tsforecast predict --config config.ini

# Compare models
tsforecast compare \
    --data data/test.csv \
    --models chronos2 timesfm tirex
```

### Mode 3: PyPI Package (Integration)

```python
from time_series_forecasting_zero import UnifiedForecaster

# Initialize with model path from parameter
forecaster = UnifiedForecaster(
    model_name='chronos2',
    model_path='./models/chronos-2',  # ← From parameter
    forecast_horizon=128,
    device='cuda'
)

# Load and predict
forecaster.load_model()
predictions = forecaster.predict(
    context=train_data,
    forecast_horizon=128,
    quantiles=[0.1, 0.5, 0.9]
)

# Access results
print(f"Mean: {predictions['mean']}")
print(f"Interval: [{predictions['lower_bound']}, {predictions['upper_bound']}]")
```

## 🛠️ Forecast Utilities

Built-in tools for evaluation and visualization:

### Metrics Calculation

```python
from time_series_forecasting_zero import (
    compute_all_metrics,
    print_metrics,
    quick_evaluate
)

# Compute all metrics at once
metrics = compute_all_metrics(
    y_true=test_data,
    y_pred=predictions['mean'],
    lower_bound=predictions['lower_bound'],
    upper_bound=predictions['upper_bound']
)
# Returns: {'rmse': ..., 'mae': ..., 'mape': ..., 'coverage_80pct': ...}

# Pretty print
print_metrics(metrics, title="Model Performance")

# One-call evaluation (metrics + plots)
quick_evaluate(
    train_data=train_data,
    test_data=test_data,
    predictions=predictions,
    model_name='Chronos-2',
    save_plots=True,
    output_dir='./outputs'
)
```

### Visualization

```python
from time_series_forecasting_zero import (
    plot_forecast,
    plot_residuals,
    compare_models_plot
)

# Forecast plot with prediction intervals
plot_forecast(
    train_data=train_data,
    predictions=predictions,
    test_data=test_data,
    save_path='forecast.png'
)

# Residual analysis (3 subplots)
plot_residuals(y_true, y_pred, save_path='residuals.png')

# Multi-model comparison
compare_models_plot(results, test_data, save_path='comparison.png')
```

**Available Functions:**
- `compute_rmse()` - Root Mean Squared Error
- `compute_mae()` - Mean Absolute Error
- `compute_mape()` - Mean Absolute Percentage Error (%)
- `compute_coverage()` - Prediction interval coverage
- `compute_all_metrics()` - All metrics at once
- `print_metrics()` - Formatted table output
- `plot_forecast()` - Forecast visualization
- `plot_residuals()` - Residual analysis
- `compare_models_plot()` - Model comparison
- `save_predictions_to_csv()` - Save to CSV
- `quick_evaluate()` - One-call evaluation + plots

## ⚙️ Configuration

### Model Path Priority

Model locations can be specified in multiple ways (highest to lowest priority):

1. **Command-line arguments**
   ```bash
   python examples/01_basic_forecasting.py --chronos2-path ./models/chronos-2
   tsforecast predict --model-path ./models/chronos-2
   ```

2. **Function parameters**
   ```python
   UnifiedForecaster('chronos2', model_path='./models/chronos-2')
   ```

3. **INI configuration file**
   ```ini
   [chronos2]
   model_path = ./models/chronos-2
   ```

4. **Environment variables**
   ```bash
   export CHRONOS2_MODEL_PATH=./models/chronos-2
   export TIMESFM_MODEL_PATH=./models/timesfm-2.5
   export TIREX_MODEL_PATH=./models/tirex-model
   ```

5. **Auto-detection** (TiRex only)
   - Checks `./models/tirex-model/` first
   - Falls back to HuggingFace `NX-AI/TiRex`

### Configuration File

Create `config.ini` from template:

```bash
cp config.ini.example config.ini
```

Example:
```ini
[DEFAULT]
device = cuda
forecast_horizon = 128

[chronos2]
model_path = ./models/chronos-2

[timesfm]
model_path = ./models/timesfm-2.5-200m-pytorch

[data]
data_dir = ./data
time_column = timestamp
value_column = value
```

Use with CLI:
```bash
tsforecast predict --config config.ini --model chronos2 --data data/test.csv
```

## 📁 Project Structure

```
Time-Series-Forecasting-Zero/
├── examples/                      # Usage examples
│   ├── 01_basic_forecasting.py   # Basic usage
│   ├── 02_load_from_csv.py       # CSV loading
│   ├── 03_batch_forecasting.py   # Batch processing
│   ├── 04_model_comparison.py    # Model comparison
│   └── 05_forecast_utilities.py  # Utilities demo
├── src/time_series_forecasting_zero/  # Main package
│   ├── __init__.py               # Package entry
│   ├── cli.py                    # CLI tool
│   ├── models/                   # Model implementations
│   │   ├── unified.py            # Unified interface
│   │   ├── chronos2.py           # Chronos-2
│   │   ├── timesfm.py            # TimesFM-2.5
│   │   └── tirex.py              # TiRex
│   ├── configs/                  # Configuration
│   ├── data/                     # Data loading
│   └── utils/                    # Utilities
│       ├── forecast_utils.py     # Metrics & visualization
│       ├── evaluator.py          # Evaluation
│       └── visualizer.py         # Plotting
├── models/                        # Pre-trained models
│   ├── chronos-2/
│   ├── timesfm-2.5-200m-pytorch/
│   └── tirex-model/
├── data/                          # Your data
├── outputs/                       # Results & logs
├── config.ini.example             # Config template
└── requirements.txt               # Dependencies
```

## 📊 Output Format

All models return standardized dictionary:

```python
{
    'mean': np.ndarray,              # Point forecasts
    'quantiles': {                   # Quantile predictions
        0.1: np.ndarray,
        0.5: np.ndarray,
        0.9: np.ndarray
    },
    'lower_bound': np.ndarray,       # Lower prediction interval
    'upper_bound': np.ndarray        # Upper prediction interval
}
```

## 🚀 CI/CD & Release

### Automated Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **Tests**: Run on Ubuntu & Windows with Python 3.9, 3.10, 3.11
- **Code Quality**: Black, Flake8, MyPy checks
- **Build**: Create distribution packages (wheel + sdist)
- **Publish**: Auto-deploy to PyPI on tagged releases

### Release Process

#### Option 1: Using Release Script (Recommended)

**Linux/Mac:**
```bash
chmod +x scripts/release.sh
./scripts/release.sh 0.1.0
```

**Windows:**
```bash
scripts\release.bat 0.1.0
```

The script will:
1. ✅ Run all tests
2. ✅ Update version numbers
3. ✅ Create git tag
4. ✅ Push to trigger CI/CD pipeline

#### Option 2: Manual Release

```bash
# 1. Update version in setup.py and __init__.py
# 2. Build package
python -m build

# 3. Check package
twine check dist/*

# 4. Upload to PyPI
twine upload dist/*

# 5. Create git tag
git tag v0.1.0
git push origin v0.1.0
```

### Test on TestPyPI

Before publishing to production PyPI, test on TestPyPI:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ time-series-forecasting-zero
```

See `RELEASE_CHECKLIST.md` for detailed release steps.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Verify installation
python verify_installation.py
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
forecaster = UnifiedForecaster(model_name="chronos2", device="cpu")
```

### Model Loading Issues
Ensure model files exist:
```bash
ls models/chronos-2/  # Should contain: config.json, model.safetensors
```

### TiRex Using HuggingFace Instead of Local
TiRex auto-detects local models. To force local:
```python
forecaster = UnifiedForecaster(
    'tirex',
    model_path='./models/tirex-model'  # Explicit local path
)
```

### Slow Inference
- Use CUDA: `device='cuda'`
- Try TiRex (fastest)
- Reduce forecast horizon

## 🤝 Contributing

Contributions welcome! Please submit Pull Requests.

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Chronos-2**: Amazon's time series foundation model
- **TimesFM-2.5**: Google's time series foundation model  
- **TiRex**: NX-AI's efficient forecasting model
- **HuggingFace Transformers**: Model hosting

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Happy Forecasting! 🎯**
