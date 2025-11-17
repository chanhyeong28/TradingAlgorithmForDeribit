# Publishing to PyPI

This guide explains how to publish the `deribit-trading-toolkit` package to PyPI.

## Prerequisites

1. Create accounts:
   - [PyPI](https://pypi.org/account/register/) (for production)
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)

2. Install build tools:
   ```bash
   pip install build twine
   ```

3. Configure credentials (create `~/.pypirc`):
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-api-token-here

   [testpypi]
   username = __token__
   password = pypi-your-test-api-token-here
   ```

## Building the Package

1. Clean previous builds:
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```

2. Build the package:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/deribit_trading_toolkit-1.0.0.tar.gz` (source distribution)
   - `dist/deribit_trading_toolkit-1.0.0-py3-none-any.whl` (wheel)

## Testing on TestPyPI

1. Upload to TestPyPI:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. Test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ deribit-trading-toolkit
   ```

## Publishing to PyPI

1. Verify the package:
   ```bash
   python -m twine check dist/*
   ```

2. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

3. Verify on PyPI:
   - Visit: https://pypi.org/project/deribit-trading-toolkit/

## Version Management

Update version in:
- `setup.py`: `version="1.0.0"`
- `deribit_trading_toolkit/__init__.py`: `__version__ = "1.0.0"`
- `CHANGELOG.md`: Add new version entry

## Checklist Before Publishing

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Update README.md if needed
- [ ] Run tests: `pytest`
- [ ] Check code style: `black --check .` and `flake8`
- [ ] Verify `MANIFEST.in` includes all necessary files
- [ ] Test installation from TestPyPI
- [ ] Update documentation if API changed

## After Publishing

1. Create a GitHub release with the version tag
2. Update documentation links if needed
3. Announce the release (if applicable)

