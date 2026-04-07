# PyPI 发布快速指南

## 🚀 首次设置

### 1. 配置 PyPI 凭证

**方式A：使用 API Token（推荐）**

1. 在 https://pypi.org/manage/account/token/ 创建 API Token
2. 在 GitHub Repository Settings → Secrets and variables → Actions 中添加：
   - `PYPI_API_TOKEN`: 你的 PyPI API token

**方式B：使用 .pypirc 文件（本地测试）**

```ini
[distutils]
index-servers = pypi testpypi

[pypi]
username: __token__
password: pypi-xxxxxxxxxxxxx

[testpypi]
username: __token__
password: pypi-xxxxxxxxxxxxx
```

### 2. 安装发布工具

```bash
pip install build twine
```

---

## 📦 发布流程

### 快速发布（推荐）

```bash
# Linux/Mac
./scripts/release.sh 0.1.0

# Windows
scripts\release.bat 0.1.0
```

### 手动发布

```bash
# 1. 更新版本号
# - setup.py: version="0.1.0"
# - src/time_series_forecasting_zero/__init__.py: __version__ = "0.1.0"

# 2. 运行测试
pytest tests/ -v
python verify_installation.py

# 3. 构建包
python -m build

# 4. 检查包
twine check dist/*

# 5. 上传到 PyPI
twine upload dist/*

# 6. 创建 git 标签
git tag v0.1.0
git push origin v0.1.0
```

---

## 🧪 测试发布

### 上传到 TestPyPI

```bash
# 上传
twine upload --repository testpypi dist/*

# 测试安装
pip install --index-url https://test.pypi.org/simple/ time-series-forecasting-zero

# 验证
python -c "import time_series_forecasting_zero; print(time_series_forecasting_zero.__version__)"
```

---

## 🔧 常见问题

### 版本号已存在

PyPI 不允许重复的版本号。解决方案：

```bash
# 增加版本号
# 0.1.0 → 0.1.1
# 或
# 0.1.0 → 0.2.0
```

### 上传失败：认证错误

检查 API Token 是否正确：

```bash
# 重新上传
twine upload -u __token__ -p pypi-your-token dist/*
```

### 构建失败

确保所有依赖都已列出：

```bash
# 检查 requirements.txt
cat requirements.txt

# 清理后重新构建
rm -rf build/ dist/ *.egg-info
python -m build
```

### CI/CD 流水线失败

1. 检查 GitHub Actions 日志
2. 确保 Tests 和 Build 任务通过
3. 检查 PyPI Token 是否正确配置

---

## 📋 版本命名规范

遵循 [Semantic Versioning](https://semver.org/)：

- **MAJOR.MINOR.PATCH**
  - `0.1.0` - 初始开发版本
  - `0.2.0` - 新功能
  - `0.1.1` - Bug 修复
  - `1.0.0` - 稳定版本

预发布版本：
- `0.1.0a1` - Alpha
- `0.1.0b1` - Beta
- `0.1.0rc1` - Release Candidate

---

## 🔗 有用链接

- PyPI: https://pypi.org/project/time-series-forecasting-zero/
- TestPyPI: https://test.pypi.org/project/time-series-forecasting-zero/
- GitHub Actions: https://github.com/yourusername/Time-Series-Forecasting-Zero/actions
- Semantic Versioning: https://semver.org/
