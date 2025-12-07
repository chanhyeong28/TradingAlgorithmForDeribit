# Project Structure

This document describes the organization of the Deribit Trading Toolkit project.

## Directory Structure

```
TradingAlgorithmForDeribit/
├── deribit_trading_toolkit/     # Main package
│   ├── __init__.py              # Package initialization and exports
│   ├── core/                    # Core API client and authentication
│   │   ├── client.py            # Deribit REST and WebSocket client
│   │   └── session_manager.py  # Multi-session support with fork_token
│   ├── analytics/              # Volatility and options analytics
│   │   ├── volatility.py        # IV curve construction (SVI/cubic spline)
│   │   ├── svi.py              # SVI parameterization
│   │   ├── pnl_decomposition.py # Model-free PnL decomposition
│   │   ├── min_var_delta.py    # Minimum variance delta hedging
│   │   ├── ssr.py              # Skew Stickiness Ratio calculation
│   │   └── backtesting.py      # Historical data collection
│   ├── backtesting/            # Backtesting framework
│   │   ├── environment.py      # Backtesting environment builder
│   │   └── simple_option.py    # Options backtesting simulator
│   ├── apps/                   # Real-time applications
│   │   └── realtime_iv.py      # Real-time IV monitoring app
│   ├── models/                 # Data models
│   │   └── market_data.py      # Market data structures
│   ├── visualization/          # Plotting and visualization
│   │   ├── term_structure.py  # Term structure visualization
│   │   └── vol_curve.py       # Volatility curve visualization
│   ├── utils/                  # Utilities
│   │   ├── config.py          # Configuration management
│   │   └── expiration_selector.py  # Expiration selection logic
│   ├── risk/                   # Risk management
│   │   └── manager.py         # Risk management system
│   ├── strategies/             # Trading strategies
│   │   └── base.py            # Base strategy framework
│   └── main.py                # Main application controller
│
├── tests/                      # Test files
│   ├── test_backtesting.py     # Backtesting environment builder test
│   ├── test_simple_option.py   # Options backtesting simulation test
│   └── test_execute.py         # Execution tests
│
├── examples/                    # Example scripts
│   ├── realtime_iv.py          # Real-time IV monitoring entry point
│   ├── example_usage.py        # General usage examples
│   └── visualize_vol_curves.py # Volatility curve visualization
│
├── key/                         # Private key storage (not in repo)
│   ├── private.pem             # RSA private key (required)
│   └── generate_key.py        # Key generation utility
│
├── .env                         # Environment variables (not in repo)
├── .env.example                 # Environment variables template
│
├── sql/                         # Database schema
│   ├── create_tables.sql       # Initial database schema
│   └── migrate_add_svi_columns.sql  # SVI migration
│
├── reference/                  # Reference materials
│   ├── varianceSwap.pdf
│   ├── skewSwap.pdf
│   └── ...
│
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup for PyPI
├── MANIFEST.in                # Files to include in distribution
├── README.md                  # Main documentation
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Contribution guidelines
├── CHANGELOG.md               # Version history
└── PUBLISHING.md              # PyPI publishing guide
```

## Main Entry Points

### 1. Real-time IV Monitoring
- **Script**: `examples/realtime_iv.py`
- **Module**: `deribit_trading_toolkit.apps.realtime_iv.RealTimeIVApp`
- **Purpose**: Live volatility monitoring with SVI fitting and term structure

### 2. Backtesting Environment Builder
- **Script**: `tests/test_backtesting.py`
- **Module**: `deribit_trading_toolkit.backtesting.environment.BacktestingEnvironment`
- **Purpose**: Build historical database for backtesting

### 3. Options Backtesting
- **Script**: `tests/test_simple_option.py`
- **Module**: `deribit_trading_toolkit.backtesting.simple_option.SimpleOptionBacktester`
- **Purpose**: Simulate options trading with PnL decomposition

## Key Modules

### Core (`core/`)
- **client.py**: Deribit API wrapper (REST and WebSocket)
- **session_manager.py**: Multi-session management using fork_token

### Analytics (`analytics/`)
- **volatility.py**: Volatility curve construction with SVI/cubic spline
- **svi.py**: SVI parameterization and fitting
- **pnl_decomposition.py**: Model-free PnL decomposition
- **min_var_delta.py**: Minimum variance delta calculation
- **ssr.py**: Skew Stickiness Ratio calculation
- **backtesting.py**: Historical data collection

### Backtesting (`backtesting/`)
- **environment.py**: Backtesting environment builder
- **simple_option.py**: Options backtesting simulator with plotting

### Apps (`apps/`)
- **realtime_iv.py**: Real-time IV monitoring application with Dash dashboard

## Data Flow

1. **Real-time Monitoring**:
   - Deribit API → Market Data → Volatility Analyzer → SVI Fitting → Dashboard

2. **Backtesting Environment**:
   - Historical Data Collection → Database Storage → Curve Building → SVI Fitting

3. **Options Backtesting**:
   - Load Historical Data → Establish Positions → Calculate Greeks → Apply Hedges → PnL Decomposition → Visualization

## Configuration

Configuration is managed through a `.env` file (excluded from git):
- Copy `.env.example` to `.env` and fill in your credentials
- All API credentials, database settings, and optional configurations
- The RSA private key should be placed at the path specified in `DERIBIT_PRIVATE_KEY_PATH` (default: `key/private.pem`)

See `.env.example` for all available configuration options.

## Database Schema

See `sql/create_tables.sql` for database schema:
- `btc_historical_futures_prices`: Futures price history
- `btc_historical_option_prices`: Option price history
- `btc_historical_volatility_curves`: Volatility curves with SVI parameters

