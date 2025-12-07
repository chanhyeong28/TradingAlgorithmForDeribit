#!/usr/bin/env python3
"""
Simple Option Buy-and-Hold Backtesting Example

Demonstrates how to use SimpleOptionBacktester to backtest buying and holding
specific options with optional delta hedging.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deribit_trading_toolkit import SimpleOptionBacktester, OptionSpec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run simple option buy-and-hold backtest"""
    try:
        logger.info("=" * 60)
        logger.info("Simple Option Buy-and-Hold Backtest")
        logger.info("=" * 60)
        
        # Define options to trade
        # Quantity: positive for long, negative for short
        options = [
            OptionSpec(
                expiration_str="26DEC25",
                strike=150000,
                option_type="call",
                quantity=-1.0  # Short 1 call
            ),
            OptionSpec(
                expiration_str="26DEC25",
                strike=80000,
                option_type="put",
                quantity=1.0  # Long 1 put 
            ),
            OptionSpec(
                expiration_str="26DEC25",
                strike=120000,
                option_type="call",
                quantity=-1.0  # Short 1 call
            ),
            OptionSpec(
                expiration_str="26DEC25",
                strike=100000,
                option_type="put",
                quantity=1.0  # Long 1 put
            )
        ]
        
        days_back = 40
        
        # Calculate timestamps
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        logger.info(f"Options to trade:")
        for opt in options:
            position_type = "Long" if opt.quantity > 0 else "Short"
            logger.info(f"  - {position_type} {abs(opt.quantity):.2f} {opt.expiration_str} {opt.strike} {opt.option_type} (quantity: {opt.quantity:+.2f})")
        logger.info(f"Period: {datetime.fromtimestamp(start_timestamp/1000)} to {datetime.fromtimestamp(end_timestamp/1000)}")
        
        # Test without delta hedging
        logger.info("\n" + "-" * 60)
        logger.info("Test 1: Without Delta Hedging")
        logger.info("-" * 60)

        
        
        with SimpleOptionBacktester(
            options=options,
            use_delta_hedge=False,
            risk_free_rate=0.05
        ) as backtester:
            result_no_hedge = backtester.run_backtest(start_timestamp, end_timestamp)
            
            logger.info("\n" + "=" * 60)
            logger.info("Results (No Hedge)")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${result_no_hedge.total_pnl:,.2f}")
            logger.info(f"Sharpe Ratio: {result_no_hedge.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: ${result_no_hedge.max_drawdown:,.2f}")
        
            ## Plot results using new OOP plotting methods
            # if result_no_hedge.daily_pnl_decomposition:
            #     logger.info("\n" + "-" * 60)
            #     logger.info("Generating plots...")
            #     logger.info("-" * 60)
                
            #     # Plot cumulative PnL
            #     logger.info("Generating cumulative PnL plot...")
            #     fig1 = backtester.plot_cumulative_pnl(result_no_hedge, save_path="simple_option_cumulative_pnl.png")
            #     logger.info("Cumulative PnL plot saved to simple_option_cumulative_pnl.png")
                
            #     # Plot daily PnL decomposition with Greek proportions
            #     logger.info("Generating daily PnL decomposition plot...")
            #     fig2 = backtester.plot_daily_pnl_decomposition(result_no_hedge, save_path="simple_option_pnl_decomposition.png")
            #     logger.info("Daily PnL decomposition plot saved to simple_option_pnl_decomposition.png")
                
            #     plt.show()
        
        # Test with Black-Scholes delta hedging
        logger.info("\n" + "-" * 60)
        logger.info("Test 2: With Black-Scholes Delta Hedging")
        logger.info("-" * 60)
        
        with SimpleOptionBacktester(
            options=options,
            use_delta_hedge=True,
            hedge_method="bs",
            risk_free_rate=0.05
        ) as backtester:
            result_bs_hedge = backtester.run_backtest(start_timestamp, end_timestamp)
            
            logger.info("\n" + "=" * 60)
            logger.info("Results (BS Delta Hedge)")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${result_bs_hedge.total_pnl:,.2f}")
            logger.info(f"Total Hedges: {result_bs_hedge.total_hedges}")
            logger.info(f"Sharpe Ratio: {result_bs_hedge.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: ${result_bs_hedge.max_drawdown:,.2f}")

            # Plot results using new OOP plotting methods
            if result_bs_hedge.daily_pnl_decomposition:
                logger.info("\n" + "-" * 60)
                logger.info("Generating plots...")
                logger.info("-" * 60)
                
                # Plot cumulative PnL
                logger.info("Generating cumulative PnL plot...")
                fig1 = backtester.plot_cumulative_pnl(result_bs_hedge, save_path="simple_option_cumulative_pnl.png")
                logger.info("Cumulative PnL plot saved to simple_option_cumulative_pnl.png")
                
                # Plot daily PnL decomposition with Greek proportions
                logger.info("Generating daily PnL decomposition plot...")
                fig2 = backtester.plot_daily_pnl_decomposition(result_bs_hedge, save_path="simple_option_pnl_decomposition.png")
                logger.info("Daily PnL decomposition plot saved to simple_option_pnl_decomposition.png")
                
                plt.show()
        
        # Test with Minimum Variance delta hedging
        logger.info("\n" + "-" * 60)
        logger.info("Test 3: With Minimum Variance Delta Hedging")
        logger.info("-" * 60)


        with SimpleOptionBacktester(
            options=options,
            use_delta_hedge=True,
            hedge_method="min_var",
            risk_free_rate=0.05
        ) as backtester:
            result_minvar_hedge = backtester.run_backtest(start_timestamp, end_timestamp)
            
            logger.info("\n" + "=" * 60)
            logger.info("Results (Min Var Delta Hedge)")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${result_minvar_hedge.total_pnl:,.2f}")
            logger.info(f"Total Hedges: {result_minvar_hedge.total_hedges}")
            logger.info(f"Sharpe Ratio: {result_minvar_hedge.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: ${result_minvar_hedge.max_drawdown:,.2f}")
            
            # Plot results using new OOP plotting methods
            if result_minvar_hedge.daily_pnl_decomposition:
                logger.info("\n" + "-" * 60)
                logger.info("Generating plots...")
                logger.info("-" * 60)
                
                # Plot cumulative PnL
                logger.info("Generating cumulative PnL plot...")
                fig1 = backtester.plot_cumulative_pnl(result_minvar_hedge, save_path="simple_option_cumulative_pnl_hedged.png")
                logger.info("Cumulative PnL plot saved to simple_option_cumulative_pnl.png")
                
                # Plot daily PnL decomposition with Greek proportions
                logger.info("Generating daily PnL decomposition plot...")
                fig2 = backtester.plot_daily_pnl_decomposition(result_minvar_hedge, save_path="simple_option_pnl_decomposition_hedged.png")
                logger.info("Daily PnL decomposition plot saved to simple_option_pnl_decomposition.png")
                
                plt.show()
        
        # Comparison summary
        logger.info("\n" + "=" * 60)
        logger.info("Comparison Summary")
        logger.info("=" * 60)
        logger.info(f"{'Metric':<25} {'No Hedge':>15} {'BS Hedge':>15} {'MinVar Hedge':>15}")
        logger.info("-" * 70)
        logger.info(f"{'Total PnL ($)':<25} {result_no_hedge.total_pnl:>15,.2f} {result_bs_hedge.total_pnl:>15,.2f} {result_minvar_hedge.total_pnl:>15,.2f}")
        logger.info(f"{'Sharpe Ratio':<25} {result_no_hedge.sharpe_ratio:>15,.2f} {result_bs_hedge.sharpe_ratio:>15,.2f} {result_minvar_hedge.sharpe_ratio:>15,.2f}")
        logger.info(f"{'Max Drawdown ($)':<25} {result_no_hedge.max_drawdown:>15,.2f} {result_bs_hedge.max_drawdown:>15,.2f} {result_minvar_hedge.max_drawdown:>15,.2f}")
        logger.info(f"{'Total Hedges':<25} {0:>15} {result_bs_hedge.total_hedges:>15} {result_minvar_hedge.total_hedges:>15}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Backtest completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

