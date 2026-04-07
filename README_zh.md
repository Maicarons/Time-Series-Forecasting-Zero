# Time-Series-Forecasting-Zero

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

一个支持最先进基础模型的生产级时间序列预测框架，具备零样本预测能力。

**三种使用方式：**
1. 📝 **示例脚本**：直接运行examples代码
2. 💻 **命令行工具**：使用 `tsforecast` 命令
3. 📦 **PyPI包**：作为Python库导入使用

## 🚀 核心特性

- **多模型支持**：Chronos-2 (Amazon)、TimesFM-2.5 (Google)、TiRex (NX-AI)
- **零样本预测**：无需在您的数据上训练即可预测
- **内置评估工具**：RMSE、MAE、MAPE、覆盖率指标 + 可视化
- **灵活配置**：命令行参数、INI文件、环境变量或代码参数
- **统一接口**：所有模型使用简单一致的API
- **概率预测**：提供预测区间和分位数
- **批量处理**：同时预测多个时间序列
- **CUDA加速**：自动GPU检测和优化
- **本地模型支持**：使用下载的模型或HuggingFace

## 📦 支持的模型

| 模型 | 机构 | 参数量 | 适用场景 | 本地路径 |
|------|------|--------|----------|----------|
| **Chronos-2** | Amazon | 1.2亿 | 通用场景，稳健性能 | `./models/chronos-2/` |
| **TimesFM-2.5** | Google | 2亿 | 长序列，高精度 | `./models/timesfm-2.5-200m-pytorch/` |
| **TiRex** | NX-AI | 3500万 | 快速推理，轻量级 | `./models/tirex-model/` |

## 🛠️ 安装

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/Time-Series-Forecasting-Zero.git
cd Time-Series-Forecasting-Zero

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装包（启用CLI）
pip install -e .
```

### 下载模型

模型存放在 `models/` 目录：

```bash
# 方式1：Git子模块
git submodule update --init --recursive

# 方式2：手动从HuggingFace/ModelScope下载
# 放置到：models/chronos-2/, models/timesfm-2.5-200m-pytorch/, models/tirex-model/
```

## 🚦 快速开始

### 方式1：示例脚本（学习用）

```bash
# 基础预测
python examples/01_basic_forecasting.py

# 指定模型路径
python examples/01_basic_forecasting.py \
    --chronos2-path ./models/chronos-2 \
    --timesfm-path ./models/timesfm-2.5

# 从CSV加载
python examples/02_load_from_csv.py

# 批量预测
python examples/03_batch_forecasting.py

# 模型对比
python examples/04_model_comparison.py

# 预测工具（指标+可视化）
python examples/05_forecast_utilities.py
```

### 方式2：命令行工具（日常使用）

```bash
# 单模型预测
tsforecast predict \
    --model chronos2 \
    --data data/test_data.csv \
    --horizon 128 \
    --device cuda

# 指定模型路径
tsforecast predict \
    --model timesfm \
    --model-path ./models/timesfm-2.5 \
    --data data/test.csv

# 使用配置文件
tsforecast predict --config config.ini

# 模型对比
tsforecast compare \
    --data data/test.csv \
    --models chronos2 timesfm tirex
```

### 方式3：PyPI包（集成开发）

```python
from time_series_forecasting_zero import UnifiedForecaster

# 通过参数指定模型路径
forecaster = UnifiedForecaster(
    model_name='chronos2',
    model_path='./models/chronos-2',  # ← 从参数获取
    forecast_horizon=128,
    device='cuda'
)

# 加载并预测
forecaster.load_model()
predictions = forecaster.predict(
    context=train_data,
    forecast_horizon=128,
    quantiles=[0.1, 0.5, 0.9]
)

# 获取结果
print(f"均值: {predictions['mean']}")
print(f"区间: [{predictions['lower_bound']}, {predictions['upper_bound']}]")
```

## 🛠️ 预测工具

内置评估和可视化工具：

### 指标计算

```python
from time_series_forecasting_zero import (
    compute_all_metrics,
    print_metrics,
    quick_evaluate
)

# 一次性计算所有指标
metrics = compute_all_metrics(
    y_true=test_data,
    y_pred=predictions['mean'],
    lower_bound=predictions['lower_bound'],
    upper_bound=predictions['upper_bound']
)
# 返回: {'rmse': ..., 'mae': ..., 'mape': ..., 'coverage_80pct': ...}

# 格式化打印
print_metrics(metrics, title="模型性能")

# 一键评估（指标+图表）
quick_evaluate(
    train_data=train_data,
    test_data=test_data,
    predictions=predictions,
    model_name='Chronos-2',
    save_plots=True,
    output_dir='./outputs'
)
```

### 可视化

```python
from time_series_forecasting_zero import (
    plot_forecast,
    plot_residuals,
    compare_models_plot
)

# 预测图（含置信区间）
plot_forecast(
    train_data=train_data,
    predictions=predictions,
    test_data=test_data,
    save_path='forecast.png'
)

# 残差分析（3个子图）
plot_residuals(y_true, y_pred, save_path='residuals.png')

# 多模型对比
compare_models_plot(results, test_data, save_path='comparison.png')
```

**可用函数：**
- `compute_rmse()` - 均方根误差
- `compute_mae()` - 平均绝对误差
- `compute_mape()` - 平均绝对百分比误差 (%)
- `compute_coverage()` - 预测区间覆盖率
- `compute_all_metrics()` - 一次性计算所有指标
- `print_metrics()` - 格式化表格输出
- `plot_forecast()` - 预测可视化
- `plot_residuals()` - 残差分析
- `compare_models_plot()` - 模型对比图
- `save_predictions_to_csv()` - 保存到CSV
- `quick_evaluate()` - 一键评估+绘图

## ⚙️ 配置说明

### 模型路径优先级

模型位置可通过多种方式指定（从高到低优先级）：

1. **命令行参数**
   ```bash
   python examples/01_basic_forecasting.py --chronos2-path ./models/chronos-2
   tsforecast predict --model-path ./models/chronos-2
   ```

2. **函数参数**
   ```python
   UnifiedForecaster('chronos2', model_path='./models/chronos-2')
   ```

3. **INI配置文件**
   ```ini
   [chronos2]
   model_path = ./models/chronos-2
   ```

4. **环境变量**
   ```bash
   export CHRONOS2_MODEL_PATH=./models/chronos-2
   export TIMESFM_MODEL_PATH=./models/timesfm-2.5
   export TIREX_MODEL_PATH=./models/tirex-model
   ```

5. **自动检测**（仅TiRex）
   - 优先检查 `./models/tirex-model/`
   - 回退到 HuggingFace `NX-AI/TiRex`

### 配置文件

从模板创建 `config.ini`：

```bash
cp config.ini.example config.ini
```

示例：
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

CLI使用：
```bash
tsforecast predict --config config.ini --model chronos2 --data data/test.csv
```

## 📁 项目结构

```
Time-Series-Forecasting-Zero/
├── examples/                      # 使用示例
│   ├── 01_basic_forecasting.py   # 基础用法
│   ├── 02_load_from_csv.py       # CSV加载
│   ├── 03_batch_forecasting.py   # 批量处理
│   ├── 04_model_comparison.py    # 模型对比
│   └── 05_forecast_utilities.py  # 工具演示
├── src/time_series_forecasting_zero/  # 主包
│   ├── __init__.py               # 包入口
│   ├── cli.py                    # CLI工具
│   ├── models/                   # 模型实现
│   │   ├── unified.py            # 统一接口
│   │   ├── chronos2.py           # Chronos-2
│   │   ├── timesfm.py            # TimesFM-2.5
│   │   └── tirex.py              # TiRex
│   ├── configs/                  # 配置管理
│   ├── data/                     # 数据加载
│   └── utils/                    # 工具函数
│       ├── forecast_utils.py     # 指标和可视化
│       ├── evaluator.py          # 评估
│       └── visualizer.py         # 绘图
├── models/                        # 预训练模型
│   ├── chronos-2/
│   ├── timesfm-2.5-200m-pytorch/
│   └── tirex-model/
├── data/                          # 您的数据
├── outputs/                       # 结果和日志
├── config.ini.example             # 配置模板
└── requirements.txt               # 依赖
```

## 📊 输出格式

所有模型返回标准化字典：

```python
{
    'mean': np.ndarray,              # 点预测
    'quantiles': {                   # 分位数预测
        0.1: np.ndarray,
        0.5: np.ndarray,
        0.9: np.ndarray
    },
    'lower_bound': np.ndarray,       # 预测区间下界
    'upper_bound': np.ndarray        # 预测区间上界
}
```

## 🚀 CI/CD 与发布

### 自动化流水线

项目使用 GitHub Actions 进行持续集成和部署：

- **测试**：在 Ubuntu & Windows 上运行，支持 Python 3.9, 3.10, 3.11
- **代码质量**：Black、Flake8、MyPy 检查
- **构建**：创建分发包（wheel + sdist）
- **发布**：打标签后自动发布到 PyPI

### 发布流程

#### 方式1：使用发布脚本（推荐）

**Linux/Mac：**
```bash
chmod +x scripts/release.sh
./scripts/release.sh 0.1.0
```

**Windows：**
```bash
scripts\release.bat 0.1.0
```

脚本将自动：
1. ✅ 运行所有测试
2. ✅ 更新版本号
3. ✅ 创建 git 标签
4. ✅ 推送触发 CI/CD 流水线

#### 方式2：手动发布

```bash
# 1. 更新 setup.py 和 __init__.py 中的版本号
# 2. 构建包
python -m build

# 3. 检查包
twine check dist/*

# 4. 上传到 PyPI
twine upload dist/*

# 5. 创建 git 标签
git tag v0.1.0
git push origin v0.1.0
```

### 在 TestPyPI 上测试

发布到正式 PyPI 前，先在 TestPyPI 上测试：

```bash
# 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 从 TestPyPI 安装
pip install --index-url https://test.pypi.org/simple/ time-series-forecasting-zero
```

详细发布步骤请参见 `RELEASE_CHECKLIST.md`。

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 验证安装
python verify_installation.py
```

## 🔧 常见问题

### CUDA内存不足
```python
forecaster = UnifiedForecaster(model_name="chronos2", device="cpu")
```

### 模型加载失败
确保模型文件存在：
```bash
ls models/chronos-2/  # 应包含: config.json, model.safetensors
```

### TiRex使用HuggingFace而非本地模型
TiRex会自动检测本地模型。强制使用本地：
```python
forecaster = UnifiedForecaster(
    'tirex',
    model_path='./models/tirex-model'  # 显式指定本地路径
)
```

### 推理速度慢
- 使用CUDA: `device='cuda'`
- 尝试TiRex（最快）
- 减少预测步长

## 🤝 贡献

欢迎贡献！请提交Pull Request。

1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 📄 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Chronos-2**: Amazon的时间序列基础模型
- **TimesFM-2.5**: Google的时间序列基础模型
- **TiRex**: NX-AI的高效预测模型
- **HuggingFace Transformers**: 模型托管

## 📧 联系

如有问题或需要支持，请在GitHub上开启Issue。

---

**祝预测顺利！🎯**
