# Release Checklist

## Before Release

### 1. Update Version
- [ ] Update version in `setup.py`
- [ ] Update version in `src/time_series_forecasting_zero/__init__.py`
- [ ] Update `CHANGELOG.md` with release notes

### 2. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Verify installation
python verify_installation.py

# Test examples
python examples/01_basic_forecasting.py --help
python examples/05_forecast_utilities.py --help
```

### 3. Code Quality
```bash
# Format code
black src/ examples/

# Lint
flake8 src/

# Type check
mypy src/
```

### 4. Build Package
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build package
python -m build

# Check package
twine check dist/*
```

### 5. Test on TestPyPI
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ time-series-forecasting-zero
```

## Release Process

### Option 1: Automatic (Recommended)

1. Create and push a tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```

2. GitHub Actions will automatically:
   - Run tests on multiple platforms
   - Build distribution packages
   - Publish to PyPI
   - Create GitHub Release

### Option 2: Manual

1. Build package:
```bash
python -m build
```

2. Upload to PyPI:
```bash
twine upload dist/*
```

3. Create GitHub Release manually

## After Release

### 1. Verify PyPI Package
- [ ] Check package page on PyPI
- [ ] Test installation: `pip install time-series-forecasting-zero`
- [ ] Test basic usage

### 2. Update Documentation
- [ ] Update README if needed
- [ ] Announce release on relevant channels

### 3. Prepare Next Version
- [ ] Bump version to next dev version (e.g., 0.1.1.dev0)
- [ ] Create new milestone for next release

## Troubleshooting

### Build Fails
- Ensure all dependencies are listed in `requirements.txt`
- Check that `setup.py` is correctly configured
- Verify package structure

### Upload Fails
- Check PyPI credentials
- Ensure version number is unique
- Verify `.pypirc` configuration

### Installation Issues
- Clear pip cache: `pip cache purge`
- Use `--no-cache-dir` flag
- Check Python version compatibility
