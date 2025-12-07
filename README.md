# Deribit Trading Toolkit

A comprehensive Python package for implementing trading algorithms on Deribit exchange. This toolkit provides tools for real-time volatility monitoring, options backtesting, and systematic trading strategies.

## Features

### 🎯 Core Functionality

- **Real-time IV Monitoring**: Live implied volatility curve visualization using SVI (Stochastic Volatility Inspired) parameterization
- **Term Structure Analysis**: ATM volatility term structure and SSR (Skew Stickiness Ratio) calculation
- **Options Backtesting**: Comprehensive backtesting environment for options trading strategies
- **PnL Decomposition**: Model-free PnL decomposition using Taylor expansion (Greeks-based)
- **Delta Hedging**: Black-Scholes and Minimum Variance delta hedging strategies
- **Deribit API Wrapper**: Clean interface for Deribit REST and WebSocket APIs

### 📊 Analytics

- **SVI Volatility Fitting**: Advanced SVI parameterization for implied volatility curves
- **Volatility Surface**: Real-time volatility surface construction and visualization
- **SSR Calculation**: Skew Stickiness Ratio for volatility regime detection
- **PnL Decomposition**: Breakdown of option PnL into Funding, IR Theta, Delta, Gamma, Vol Block, and Vanna Block
- **Minimum Variance Delta**: Advanced delta hedging using minimum variance methodology

### 🔧 Tools

- **Backtesting Environment Builder**: Automated historical data collection and curve building
- **Real-time Dashboard**: Web-based dashboard for live volatility monitoring
- **Risk Management**: Position sizing, margin management, and portfolio risk assessment
- **Market Making Strategies**: Simple and sophisticated market making with volatility-based spreads
- **Multi-session Support**: Run multiple trading strategies simultaneously with session management

## Installation

### From PyPI(On its way)

```bash
pip install deribit-trading-toolkit
```

### From Source

```bash
git clone https://github.com/chanhyeong28/TradingAlgorithmForDeribit.git
cd TradingAlgorithmForDeribit
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Deribit API Wrapper

The Deribit API client is the foundation of the toolkit. It provides a clean interface for both REST and WebSocket APIs:

```python
import asyncio
from deribit_trading_toolkit import DeribitClient, DeribitAuth
from deribit_trading_toolkit.utils.config import ConfigManager

async def main():
    config = ConfigManager().get_config()
    auth = DeribitAuth(
        config.deribit.effective_client_id,
        config.deribit.effective_private_key_path,
        config.deribit.effective_private_key
    )
    
    client = DeribitClient(config.deribit, auth)
    await client.connect()
    
    try:
        # Get market data
        ticker = await client.get_ticker("BTC-PERPETUAL")
        print(f"BTC Price: ${ticker.mark_price}")
        
        # Get option chain
        instruments = await client.get_instruments("BTC", "option")
        print(f"Found {len(instruments)} options")
        
        # Subscribe to mark price updates
        await client.subscribe_to_mark_price()
        
        # Register handler
        def on_mark_price(channel, msg):
            print(f"Mark price update: {msg}")
        
        client.register_message_handler(
            r"markprice\.options\.btc_usd",
            on_mark_price
        )
        
        # Listen for messages
        await client.listen_for_messages()
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

**Features:**
- REST API wrapper for all Deribit endpoints
- WebSocket support for real-time data streaming
- Automatic authentication with RSA or secret key
- Message handler registration for WebSocket channels
- Support for both mainnet and testnet
- Rate limiting and connection management
- Multi-session support with fork tokens

**Key Methods:**
- `get_ticker(instrument)`: Get current ticker data
- `get_instruments(currency, kind)`: Get list of instruments
- `get_orderbook(instrument)`: Get orderbook data
- `subscribe_to_mark_price()`: Subscribe to mark price updates
- `subscribe_to_ticker(channels)`: Subscribe to ticker channels
- `register_message_handler(pattern, handler)`: Register WebSocket message handlers
- `listen_for_messages()`: Start listening for WebSocket messages

### 2. Real-time IV Monitoring

Monitor implied volatility curves in real-time with SVI fitting and term structure analysis:

```python
import asyncio
from deribit_trading_toolkit import RealTimeIVApp

async def main():
    app = RealTimeIVApp(
        refresh_seconds=2,
        futures_refresh_seconds=15,
        use_auto_expirations=True
    )
    try:
        await app.start()
        # Dashboard available at http://127.0.0.1:8050
    finally:
        await app.stop()

asyncio.run(main())
```

Or use the example script:

```bash
python examples/realtime_iv.py
```

**Features:**
- Real-time IV curve visualization with SVI fitting
- ATM volatility term structure
- SSR (Skew Stickiness Ratio) calculation
- Automatic expiration selection (daily, weekly, monthly, quarterly)
- Web-based dashboard with auto-refresh

### 3. Building Backtesting Environment

Create a backtesting environment with historical data:

```python
import asyncio
from deribit_trading_toolkit import BacktestingEnvironment

async def main():
    async with BacktestingEnvironment() as env:
        results = await env.build_environment(
            expirations=["26DEC25", "27MAR26"],
            days_back=30,
            resolution=60,  # 1-minute bars
            time_window_seconds=3600,  # Build curve every hour
            save_prices=True,
            save_curves=True
        )
        
        for result in results:
            if result.success:
                print(f"{result.expiration_str}: "
                      f"{result.futures_records} futures, "
                      f"{result.option_records} options, "
                      f"{result.curves_saved} curves")

asyncio.run(main())
```

Or use the example script:

```bash
python examples/backtestEnv_option.py
```

**Features:**
- Historical data collection from Deribit
- Volatility curve building with SVI or cubic spline
- Database storage for futures, options, and curves
- Configurable time windows and resolution

### 4. Options Backtesting

Simulate options trading strategies with PnL decomposition:

```python
import asyncio
from deribit_trading_toolkit import SimpleOptionBacktester, OptionSpec
from datetime import datetime, timedelta

async def main():
    # Define options portfolio
    options = [
        OptionSpec(
            expiration_str="26DEC25",
            strike=120000,
            option_type="call",
            quantity=1.0  # Long 1 call
        ),
        OptionSpec(
            expiration_str="26DEC25",
            strike=100000,
            option_type="put",
            quantity=-1.0  # Short 1 put
        )
    ]
    
    # Calculate timestamps
    end_timestamp = int(datetime.now().timestamp() * 1000)
    start_timestamp = int((datetime.now() - timedelta(days=40)).timestamp() * 1000)
    
    # Run backtest with minimum variance delta hedging
    with SimpleOptionBacktester(
        options=options,
        use_delta_hedge=True,
        hedge_method="min_var",  # or "bs" for Black-Scholes
        risk_free_rate=0.05
    ) as backtester:
        result = backtester.run_backtest(start_timestamp, end_timestamp)
        
        print(f"Total PnL: ${result.total_pnl:,.2f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: ${result.max_drawdown:,.2f}")
        
        # Generate plots
        backtester.plot_cumulative_pnl(result, save_path="cumulative_pnl.png")
        backtester.plot_daily_pnl_decomposition(result, save_path="pnl_decomposition.png")

asyncio.run(main())
```

Or use the example script:

```bash
python examples/option_backtest.py
```

**Features:**
- Support for long and short options (positive/negative quantities)
- Delta hedging with Black-Scholes or Minimum Variance methods
- PnL decomposition into Greeks (Delta, Gamma, Vol Block, Vanna Block, etc.)
- Comparison of real PnL vs. Taylor expansion approximation
- Comprehensive visualization with cumulative PnL and decomposition charts

### 5. Market Making Strategies

The toolkit includes two market making strategies:

#### Simple Market Maker

A basic market making strategy that places limit orders on both sides of the orderbook:

```python
import asyncio
from deribit_trading_toolkit import DeribitClient, DeribitAuth, ConfigManager
from deribit_trading_toolkit.strategies.market_maker import SimpleMarketMaker, MarketMakerConfig
from deribit_trading_toolkit.risk.manager import RiskManager, RiskLimits

async def main():
    config = ConfigManager().get_config()
    auth = DeribitAuth(
        config.deribit.effective_client_id,
        config.deribit.effective_private_key_path,
        config.deribit.effective_private_key
    )
    
    client = DeribitClient(config.deribit, auth)
    await client.connect()
    
    try:
        # Setup risk manager
        risk_limits = RiskLimits(
            max_position_size=0.5,
            max_portfolio_risk=0.2
        )
        risk_manager = RiskManager(client, risk_limits)
        
        # Create market maker
        mm_config = MarketMakerConfig(
            instrument="BTC-PERPETUAL",
            spread_bps=10,  # 0.1% spread
            order_size=0.01,
            max_position=0.5
        )
        
        market_maker = SimpleMarketMaker(
            config=mm_config,
            client=client,
            risk_manager=risk_manager
        )
        
        await market_maker.start()
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

Or use the example script:

```bash
python examples/market_maker.py
```

#### Sophisticated Market Maker

An advanced market making strategy with volatility-based spread adjustment and inventory management:

```python
import asyncio
from deribit_trading_toolkit import DeribitClient, DeribitAuth, ConfigManager
from deribit_trading_toolkit.strategies.sophisticated_mm import SophisticatedMarketMaker, SophisticatedMMConfig
from deribit_trading_toolkit.risk.manager import RiskManager, RiskLimits

async def main():
    config = ConfigManager().get_config()
    auth = DeribitAuth(
        config.deribit.effective_client_id,
        config.deribit.effective_private_key_path,
        config.deribit.effective_private_key
    )
    
    client = DeribitClient(config.deribit, auth)
    await client.connect()
    
    try:
        # Setup risk manager
        risk_limits = RiskLimits(
            max_position_size=0.5,
            max_portfolio_risk=0.2
        )
        risk_manager = RiskManager(client, risk_limits)
        
        # Create sophisticated market maker
        mm_config = SophisticatedMMConfig(
            instrument="BTC-PERPETUAL",
            base_spread_bps=10,
            volatility_multiplier=1.5,
            inventory_skew_factor=0.1,
            order_size=0.01,
            max_position=0.5
        )
        
        market_maker = SophisticatedMarketMaker(
            config=mm_config,
            client=client,
            risk_manager=risk_manager
        )
        
        await market_maker.start()
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

Or use the example script:

```bash
python examples/sophisticated_mm.py
```

**Features:**
- Real-time orderbook subscription via WebSocket
- Volatility-based spread adjustment
- Inventory-based quote skewing
- Risk management with position limits
- Automatic order cancellation and replacement


## Configuration

### Required Setup

1. **Create `.env` file**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and fill in your credentials:
     - **Mainnet credentials:**
       - `DERIBIT_CLIENT_ID`: Your Deribit mainnet client ID
       - **Option 1 (Recommended):** `DERIBIT_PRIVATE_KEY_PATH`: Path to your RSA private key file (default: `key/private.pem`)
       - **Option 2:** `DERIBIT_PRIVATE_KEY`: Direct secret key string (leave `DERIBIT_PRIVATE_KEY_PATH` empty if using this)
     - **Testnet credentials (optional, but recommended for testing):**
       - `DERIBIT_CLIENT_ID_TESTNET`: Your Deribit testnet client ID
       - **Option 1:** `DERIBIT_PRIVATE_KEY_PATH_TESTNET`: Path to your RSA private key file (default: `key/private_testnet.pem`)
       - **Option 2 (Recommended for testnet):** `DERIBIT_PRIVATE_KEY_TESTNET`: Direct secret key string (leave `DERIBIT_PRIVATE_KEY_PATH_TESTNET` empty if using this)
     - Place your RSA private key files at the specified paths (if using Option 1)
     - Set `DERIBIT_TESTNET=false` for mainnet or `DERIBIT_TESTNET=true` for testnet

2. **Database Configuration** (for backtesting):
   - Set up MySQL database
   - Configure in `.env` file:
     ```bash
     DB_HOST=localhost
     DB_USER=root
     DB_PASSWORD=your_password
     DB_NAME=btc_options_db
     DB_PORT=3306
     ```

3. **Optional: Telegram Notifications**:
   - Add to `.env`:
     ```bash
     TELEGRAM_BOT_TOKEN=your_bot_token
     TELEGRAM_CHAT_ID=your_chat_id
     ```

### Environment Variables

All configuration is managed through the `.env` file. The following variables are available:

**Deribit API - Mainnet:**
- `DERIBIT_CLIENT_ID`: Your Deribit mainnet client ID (required for mainnet)
- `DERIBIT_PRIVATE_KEY_PATH`: Path to RSA private key for mainnet (default: `key/private.pem`)
- `DERIBIT_PRIVATE_KEY`: Direct secret key string for mainnet (alternative to `DERIBIT_PRIVATE_KEY_PATH`)

**Deribit API - Testnet:**
- `DERIBIT_CLIENT_ID_TESTNET`: Your Deribit testnet client ID (required for testnet)
- `DERIBIT_PRIVATE_KEY_PATH_TESTNET`: Path to RSA private key for testnet (default: `key/private_testnet.pem`)
- `DERIBIT_PRIVATE_KEY_TESTNET`: Direct secret key string for testnet (alternative to `DERIBIT_PRIVATE_KEY_PATH_TESTNET`)

**Deribit API Settings:**
- `DERIBIT_TESTNET`: Use testnet (`true`/`false`, default: `false`)
  - When `true`, uses testnet credentials and testnet API endpoints
  - When `false`, uses mainnet credentials and mainnet API endpoints
- `DERIBIT_RATE_LIMIT`: API rate limit (default: `20`)
- `DERIBIT_TIMEOUT`: Request timeout in seconds (default: `30`)

> **Note:** You can use either a `.pem` file path OR a direct secret key string. If both are provided, the secret key string takes precedence. For mainnet, using RSA key files (`.pem`) is recommended for better security. For testnet, you can use direct secret keys for convenience. You can configure both mainnet and testnet credentials in `.env`. The system will automatically use the appropriate credentials based on the `DERIBIT_TESTNET` setting. If testnet credentials are not provided, it will fall back to mainnet credentials.

**Database:**
- `DB_HOST`: Database host (default: `localhost`)
- `DB_USER`: Database user (default: `root`)
- `DB_PASSWORD`: Database password
- `DB_NAME`: Database name (default: `btc_options_db`)
- `DB_PORT`: Database port (default: `3306`)

**Telegram (Optional):**
- `TELEGRAM_BOT_TOKEN`: Telegram bot token
- `TELEGRAM_CHAT_ID`: Telegram chat ID

**Trading:**
- `TRADING_NEAR_EXPIRATION`: Near expiration date
- `TRADING_FAR_EXPIRATION`: Far expiration date
- `TRADING_SPREAD_WAY`: Spread direction (`SHORT` or `LONG`, default: `SHORT`)
- `TRADING_EXECUTION_ENABLED`: Enable trading execution (default: `false`)
- `TRADING_POSITION_SIZE`: Position size (default: `0.1`)
- `TRADING_MAX_POSITIONS`: Maximum positions (default: `4`)

**Risk Management:**
- `RISK_MAX_POSITION_SIZE`: Maximum position size (default: `0.1`)
- `RISK_MAX_PORTFOLIO_RISK`: Maximum portfolio risk (default: `0.2`)
- `RISK_MARGIN_BUFFER`: Margin buffer (default: `1.2`)
- `RISK_MAX_DAILY_LOSS`: Maximum daily loss (default: `0.1`)
- `RISK_STOP_LOSS_PERCENTAGE`: Stop loss percentage (default: `0.05`)

**Logging:**
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `LOG_FILE_PATH`: Path to log file (optional)
- `LOG_MAX_FILE_SIZE`: Maximum log file size in bytes (default: `10485760`)
- `LOG_BACKUP_COUNT`: Number of backup log files (default: `5`)

> **Note:** The `.env` file is automatically loaded when the package is imported. You can also set these as system environment variables, which will take precedence over the `.env` file.

> **Legacy Support:** The code still supports reading from `key/*.txt` files for backward compatibility, but using `.env` is the recommended approach.

## Project Structure

```
deribit_trading_toolkit/
├── core/                    # Deribit API client and authentication
│   ├── client.py           # REST and WebSocket client
│   └── session_manager.py  # Multi-session support
├── analytics/              # Volatility and options analytics
│   ├── volatility.py       # IV curve construction (SVI/cubic spline)
│   ├── svi.py             # SVI parameterization
│   ├── pnl_decomposition.py  # Model-free PnL decomposition
│   ├── min_var_delta.py   # Minimum variance delta hedging
│   ├── ssr.py             # Skew Stickiness Ratio calculation
│   └── backtesting.py     # Historical data collection
├── backtesting/           # Backtesting framework
│   ├── environment.py     # Backtesting environment builder
│   └── simple_option.py   # Options backtesting simulator
├── apps/                  # Real-time applications
│   └── realtime_iv.py     # Real-time IV monitoring app
├── models/                # Data models
│   └── market_data.py     # Market data structures
├── visualization/         # Plotting and visualization
│   ├── term_structure.py  # Term structure visualization
│   └── vol_curve.py       # Volatility curve visualization
├── strategies/            # Trading strategies
│   ├── base.py            # Base strategy framework
│   ├── market_maker.py    # Simple market making strategy
│   └── sophisticated_mm.py # Advanced market making strategy
├── utils/                 # Utilities
│   ├── config.py          # Configuration management
│   └── expiration_selector.py  # Expiration selection logic
├── risk/                  # Risk management
│   └── manager.py         # Risk management system
└── main.py                # Main application controller
```

## Key Features Explained

### SVI Volatility Fitting

The toolkit uses SVI (Stochastic Volatility Inspired) parameterization for fitting implied volatility curves:

```python
from deribit_trading_toolkit import VolatilityAnalyzer

analyzer = VolatilityAnalyzer(use_svi=True)  # Default: True
curve = analyzer.build_volatility_curve(options_data, underlying_price, expiration)

# SVI parameters are stored in curve.svi_params
if hasattr(curve, 'svi_params') and curve.svi_params:
    print(f"SVI params: {curve.svi_params}")
```

**Quality Control**: The toolkit automatically evaluates SVI fit quality (R², RMSE, max error) and falls back to cubic spline interpolation if the fit is poor.

### PnL Decomposition

Model-free PnL decomposition breaks down option PnL into:

- **Funding**: Interest rate component
- **IR Theta**: Time decay
- **Delta**: Price movement impact
- **Gamma**: Convexity effect
- **Vol Block**: Volatility changes
- **Vanna Block**: Volatility skew changes

```python
from deribit_trading_toolkit import PnLDecomposer

decomposer = PnLDecomposer(risk_free_rate=0.05)
result = decomposer.decompose(
    F_t=prev_price,
    F_next=current_price,
    K=strike,
    tau_years=time_to_expiry,
    V_t=option_value,
    # ... other parameters
)

print(f"Delta PnL: {result.delta}")
print(f"Gamma PnL: {result.gamma}")
print(f"Vol Block: {result.vol_block}")
```

### Minimum Variance Delta

Advanced delta hedging using minimum variance methodology:

```python
from deribit_trading_toolkit import MinimumVarianceDeltaCalculator

calc = MinimumVarianceDeltaCalculator(risk_free_rate=0.05)
delta_min = calc.calculate_delta_min(
    F_t=underlying_price,
    K=strike,
    tau_years=time_to_expiry,
    sqrtw=sqrt_total_variance,
    d_sqrtw_dk=slope_at_moneyness,
    option_type="call"  # or "put"
)
```

## Examples

The `examples/` directory contains comprehensive example scripts:

### Real-time Monitoring
- **`realtime_iv.py`**: Real-time IV monitoring with SVI fitting and term structure visualization
- **`visualize_vol_curves.py`**: Volatility curve visualization and analysis

### Backtesting
- **`backtestEnv_option.py`**: Build backtesting environment with historical data collection
- **`option_backtest.py`**: Options backtesting simulation with PnL decomposition

### Trading Strategies
- **`market_maker.py`**: Simple market making strategy example
- **`sophisticated_mm.py`**: Advanced market making with volatility-based spreads

### General Usage
- **`example_usage.py`**: General toolkit usage examples and patterns

All examples can be run directly:

```bash
# Real-time IV monitoring
python examples/realtime_iv.py

# Build backtesting environment
python examples/backtestEnv_option.py

# Run options backtest
python examples/option_backtest.py

# Run market maker (testnet)
python examples/market_maker.py

# Run sophisticated market maker (testnet)
python examples/sophisticated_mm.py

# Visualize volatility curves
python examples/visualize_vol_curves.py
```

> **Note**: Market making examples require `DERIBIT_TESTNET=true` in your `.env` file for safety.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=deribit_trading_toolkit --cov-report=html

# Run specific test file
pytest tests/test_backtesting.py
pytest tests/test_simple_option.py
```

> **Note**: Test files are located in the `tests/` directory (if present). Example scripts in `examples/` can also serve as integration tests.

## Documentation

- **API Reference**: See docstrings in source code
- **Examples**: Check `examples/` directory for comprehensive usage examples
- **Project Structure**: See `PROJECT_STRUCTURE.md` for detailed architecture documentation
- **Tutorials**: Coming soon

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: This software is provided for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use at your own risk.

## Support

- **Issues**: [GitHub Issues](https://github.com/chanhyeong28/TradingAlgorithmForDeribit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chanhyeong28/TradingAlgorithmForDeribit/discussions)

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{deribit_trading_toolkit,
  title = {Deribit Trading Toolkit},
  author = {Chanhyeong28},
  year = {2024},
  url = {https://github.com/chanhyeong28/TradingAlgorithmForDeribit}
}
```

## Acknowledgments

- Deribit for providing the exchange API
- The quantitative finance community for research and methodologies
- Contributors and users of this toolkit
