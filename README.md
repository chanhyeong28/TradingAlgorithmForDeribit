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

## Installation

### From PyPI

```bash
pip install deribit-trading-toolkit
```

### From Source

```bash
git clone https://github.com/yourusername/TradingAlgorithmForDeribit.git
cd TradingAlgorithmForDeribit
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Real-time IV Monitoring

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

Or use the command-line entry point:

```bash
python realtime_iv.py
```

**Features:**
- Real-time IV curve visualization with SVI fitting
- ATM volatility term structure
- SSR (Skew Stickiness Ratio) calculation
- Automatic expiration selection (daily, weekly, monthly, quarterly)
- Web-based dashboard with auto-refresh

### 2. Building Backtesting Environment

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

Or use the test script:

```bash
python test_backtesting.py
```

**Features:**
- Historical data collection from Deribit
- Volatility curve building with SVI or cubic spline
- Database storage for futures, options, and curves
- Configurable time windows and resolution

### 3. Options Backtesting

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

Or use the test script:

```bash
python test_simple_option.py
```

**Features:**
- Support for long and short options (positive/negative quantities)
- Delta hedging with Black-Scholes or Minimum Variance methods
- PnL decomposition into Greeks (Delta, Gamma, Vol Block, Vanna Block, etc.)
- Comparison of real PnL vs. Taylor expansion approximation
- Comprehensive visualization with cumulative PnL and decomposition charts

### 4. Deribit API Wrapper

Basic usage of the Deribit API client:

```python
import asyncio
from deribit_trading_toolkit import DeribitClient, DeribitAuth
from deribit_trading_toolkit.utils.config import ConfigManager

async def main():
    config = ConfigManager().get_config()
    auth = DeribitAuth(
        config.deribit.client_id,
        config.deribit.private_key_path
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

## Configuration

### Required Setup

1. **Deribit API Credentials**:
   - Create a file `key/client_id.txt` with your Deribit client ID
   - Place your RSA private key at `key/private.pem`

2. **Database Configuration** (for backtesting):
   - Set up MySQL database
   - Configure in `deribit_trading_toolkit/utils/config.py` or use environment variables:
     ```bash
     export DB_HOST=localhost
     export DB_USER=root
     export DB_PASSWORD=your_password
     export DB_NAME=btc_options_db
     ```

3. **Optional: Telegram Notifications**:
   - `key/bot_token.txt`: Telegram bot token
   - `key/chat_id.txt`: Telegram chat ID

### Environment Variables

```bash
export DERIBIT_TESTNET=false
export TRADING_EXECUTION_ENABLED=false
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=btc_options_db
```

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
├── utils/                 # Utilities
│   ├── config.py          # Configuration management
│   └── expiration_selector.py  # Expiration selection logic
└── risk/                  # Risk management
    └── manager.py         # Risk management system
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

### Example Scripts

- **`realtime_iv.py`**: Real-time IV monitoring entry point
- **`test_backtesting.py`**: Backtesting environment builder
- **`test_simple_option.py`**: Options backtesting simulation

### Advanced Usage

See the `examples/` directory for more detailed examples and tutorials.

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

# Run specific test
pytest test_backtesting.py
pytest test_simple_option.py
```

## Documentation

- **API Reference**: See docstrings in source code
- **Examples**: Check `test_*.py` files for usage examples
- **Tutorials**: Coming soon

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: This software is provided for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use at your own risk.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/TradingAlgorithmForDeribit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/TradingAlgorithmForDeribit/discussions)

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{deribit_trading_toolkit,
  title = {Deribit Trading Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/TradingAlgorithmForDeribit}
}
```

## Acknowledgments

- Deribit for providing the exchange API
- The quantitative finance community for research and methodologies
- Contributors and users of this toolkit
