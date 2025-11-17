"""
Setup script for deribit-trading-toolkit package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deribit-trading-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive Python package for real-time volatility monitoring, options backtesting, and trading on Deribit",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TradingAlgorithmForDeribit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deribit-toolkit=deribit_trading_toolkit.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deribit_trading_toolkit": [
            "examples/*.py",
            "examples/*.ipynb",
            "data/*.json",
        ],
    },
    keywords="deribit, trading, options, volatility, quantitative finance, algorithmic trading",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TradingAlgorithmForDeribit/issues",
        "Source": "https://github.com/yourusername/TradingAlgorithmForDeribit",
        "Documentation": "https://github.com/yourusername/TradingAlgorithmForDeribit#readme",
    },
)
