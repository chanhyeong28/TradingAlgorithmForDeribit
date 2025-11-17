# Contributing to Deribit Trading Toolkit

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/TradingAlgorithmForDeribit.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install dev dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write or update tests
4. Ensure all tests pass: `pytest`
5. Check code style: `black --check .` and `flake8`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black .`
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for all public functions and classes

## Testing

- Write tests for new features
- Ensure all existing tests pass
- Aim for good test coverage
- Use `pytest` for running tests

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update examples if API changes

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md (if exists)
5. Request review from maintainers

## Questions?

Open an issue for questions or discussions about contributions.

