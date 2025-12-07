"""
Main Application Controller

Orchestrates the entire trading system including:
- Configuration management
- Service initialization
- Strategy execution
- Risk management
- Data collection
- Performance monitoring
"""

import asyncio
import logging
import signal
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .core.client import DeribitClient, DeribitAuth
from .core.session_manager import SessionManager, MultiSessionTradingApp
from .analytics.volatility import VolatilityAnalyzer
from .risk.manager import RiskManager, RiskLimits
from .strategies.base import RiskReversalStrategy, RiskReversalConfig
from .utils.config import AppConfig, ConfigManager
from .models.market_data import MarketData, OptionContract

logger = logging.getLogger(__name__)


@dataclass
class AppStatus:
    """Application status tracking"""
    running: bool = False
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    errors_count: int = 0
    strategies_active: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0


class TradingApp:
    """
    Main trading application controller
    
    Features:
    - Configuration management
    - Service orchestration
    - Strategy execution
    - Risk management
    - Multi-session support
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or ConfigManager().get_config()
        self.status = AppStatus()
        
        # Core services
        self.client: Optional[DeribitClient] = None
        self.auth: Optional[DeribitAuth] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Multi-session support
        self.session_manager: Optional[SessionManager] = None
        self.multi_session_app: Optional[MultiSessionTradingApp] = None
        
        # Analytics services
        self.volatility_analyzer: Optional[VolatilityAnalyzer] = None
        
        # Strategies
        self.strategies: List[RiskReversalStrategy] = []
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup application logging"""
        try:
            logging.basicConfig(
                level=getattr(logging, self.config.logging.level),
                format=self.config.logging.format,
                datefmt=self.config.logging.date_format,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(self.config.logging.file_path) if self.config.logging.file_path else logging.NullHandler()
                ]
            )
            logger.info("Logging configured successfully")
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialize the trading application
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing trading application...")
            
            # Validate configuration
            errors = self.config.validate()
            if errors:
                logger.error(f"Configuration validation failed: {errors}")
                return False
            
            # Initialize authentication
            self.auth = DeribitAuth(
                client_id=self.config.deribit.effective_client_id,
                private_key_path=self.config.deribit.effective_private_key_path,
                private_key=self.config.deribit.effective_private_key
            )
            
            # Initialize client
            self.client = DeribitClient(self.config.deribit, self.auth)
            
            # Initialize risk manager
            risk_limits = RiskLimits(
                max_position_size=self.config.risk.max_position_size,
                max_portfolio_risk=self.config.risk.max_portfolio_risk,
                margin_buffer=self.config.risk.margin_buffer,
                max_daily_loss=self.config.risk.max_daily_loss,
                stop_loss_percentage=self.config.risk.stop_loss_percentage
            )
            self.risk_manager = RiskManager(self.client, risk_limits)
            
            # Initialize analytics services
            self.volatility_analyzer = VolatilityAnalyzer()
            
            # Initialize multi-session support if enabled
            if self.config.session.enable_multi_session:
                await self._initialize_multi_session()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            logger.info("Trading application initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            return False
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            if not self.config.trading.is_valid:
                logger.warning("Trading configuration is invalid, skipping strategy initialization")
                return
            
            # Create Risk Reversal strategy
            strategy_config = RiskReversalConfig(
                name="Risk Reversal Strategy",
                enabled=self.config.trading.execution_enabled,
                position_size=self.config.trading.position_size,
                max_positions=self.config.trading.max_positions,
                near_expiration=self.config.trading.near_expiration,
                far_expiration=self.config.trading.far_expiration,
                spread_way=self.config.trading.spread_way,
                perpetual_expirations=self.config.trading.perpetual_expirations
            )
            
            strategy = RiskReversalStrategy(
                config=strategy_config,
                client=self.client,
                risk_manager=self.risk_manager
            )
            
            self.strategies.append(strategy)
            logger.info(f"Initialized strategy: {strategy_config.name}")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
    
    async def _initialize_multi_session(self):
        """Initialize multi-session support"""
        try:
            self.session_manager = SessionManager(
                config=self.config.deribit,
                base_client_id=self.config.deribit.effective_client_id,
                private_key_path=self.config.deribit.effective_private_key_path,
                private_key=self.config.deribit.effective_private_key
            )
            
            self.multi_session_app = MultiSessionTradingApp(
                config=self.config.deribit,
                base_client_id=self.config.deribit.effective_client_id,
                private_key_path=self.config.deribit.effective_private_key_path,
                private_key=self.config.deribit.effective_private_key
            )
            
            await self.multi_session_app.initialize()
            logger.info("Multi-session support initialized")
            
        except Exception as e:
            logger.error(f"Error initializing multi-session support: {e}")
    
    async def create_strategy_session(self, session_name: str, strategy_class, strategy_config) -> Optional[str]:
        """
        Create a new session for a trading strategy
        
        Args:
            session_name: Name for the session
            strategy_class: Strategy class to instantiate
            strategy_config: Configuration for the strategy
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if not self.multi_session_app:
                logger.warning("Multi-session support not enabled")
                return None
            
            return await self.multi_session_app.create_strategy_session(
                session_name, strategy_class, strategy_config
            )
            
        except Exception as e:
            logger.error(f"Error creating strategy session: {e}")
            return None
    
    async def run_strategy_in_session(self, session_id: str) -> Dict[str, Any]:
        """
        Run a strategy in a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            Strategy execution result
        """
        try:
            if not self.multi_session_app:
                return {"status": "error", "message": "Multi-session support not enabled"}
            
            return await self.multi_session_app.run_strategy(session_id)
            
        except Exception as e:
            logger.error(f"Error running strategy in session: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        if not self.session_manager:
            return None
        return self.session_manager.get_session_info(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        if not self.session_manager:
            return []
        return self.session_manager.list_sessions()
    
    async def run(self):
        """Main application run loop"""
        try:
            # Initialize application
            if not await self.initialize():
                logger.error("Failed to initialize application")
                return
            
            # Connect to Deribit
            await self.client.connect()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update status
            self.status.running = True
            self.status.start_time = datetime.now()
            
            logger.info("Trading application started successfully")
            
            # Main execution loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error in main application loop: {e}")
            self.status.errors_count += 1
        finally:
            await self.shutdown()
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Data collection task
            self.tasks.append(asyncio.create_task(self._data_collection_loop()))
            
            # Strategy execution task
            self.tasks.append(asyncio.create_task(self._strategy_execution_loop()))
            
            # Analytics task
            self.tasks.append(asyncio.create_task(self._analytics_loop()))
            
            # Health check task
            self.tasks.append(asyncio.create_task(self._health_check_loop()))
            
            logger.info(f"Started {len(self.tasks)} background tasks")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _main_loop(self):
        """Main application loop"""
        try:
            while self.status.running:
                # Update heartbeat
                self.status.last_heartbeat = datetime.now()
                
                # Check for shutdown signal
                if not self.status.running:
                    break
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.status.errors_count += 1
    
    async def _data_collection_loop(self):
        """Background task for collecting market data"""
        while self.status.running:
            try:
                # Subscribe to market data channels
                channels = self._get_subscription_channels()
                await self.client.subscribe_to_ticker(channels)
                await self.client.subscribe_to_mark_price()
                
                # Listen for messages
                await self.client.listen_for_messages()
                
                await asyncio.sleep(60)  # Update subscriptions every minute
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                self.status.errors_count += 1
                await asyncio.sleep(10)
    
    async def _strategy_execution_loop(self):
        """Background task for strategy execution"""
        while self.status.running:
            try:
                for strategy in self.strategies:
                    if strategy.is_enabled():
                        result = await strategy.run_strategy()
                        
                        if result.get("executions", 0) > 0:
                            self.status.total_trades += result["executions"]
                            logger.info(f"Strategy {strategy.config.name} executed {result['executions']} trades")
                
                await asyncio.sleep(60)  # Check strategies every minute
                
            except Exception as e:
                logger.error(f"Error in strategy execution loop: {e}")
                self.status.errors_count += 1
                await asyncio.sleep(10)
    
    async def _analytics_loop(self):
        """Background task for analytics calculations"""
        while self.status.running:
            try:
                # Calculate volatility curves
                await self._calculate_volatility_curves()
                
                await asyncio.sleep(60)  # Calculate analytics every minute
                
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                self.status.errors_count += 1
                await asyncio.sleep(10)
    
    async def _health_check_loop(self):
        """Background task for health monitoring"""
        while self.status.running:
            try:
                # Check client connection
                if not self.client.session or self.client.session.closed:
                    logger.warning("Client connection lost, attempting to reconnect...")
                    await self.client.connect()
                
                # Check risk status
                risk_summary = self.risk_manager.get_risk_summary()
                if risk_summary.get('current_status', {}).get('daily_loss_limit_exceeded', False):
                    logger.warning("Daily loss limit exceeded, disabling strategies")
                    for strategy in self.strategies:
                        strategy.disable()
                
                # Log status
                await self._log_status()
                
                await asyncio.sleep(300)  # Health check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                self.status.errors_count += 1
                await asyncio.sleep(30)
    
    async def _calculate_volatility_curves(self):
        """Calculate volatility curves for all expirations"""
        try:
            # Get options data
            options = await self.client.get_option_chain("BTC")
            
            # Calculate curves for each expiration
            expirations = list(set(opt.expiration_timestamp for opt in options))
            
            for expiration in expirations:
                expiration_options = [opt for opt in options if opt.expiration_timestamp == expiration]
                
                # Get underlying price (simplified)
                underlying_price = 50000  # In practice, get from market data
                
                curve = self.volatility_analyzer.build_volatility_curve(
                    expiration_options, underlying_price, expiration
                )
                
                if curve:
                    logger.debug(f"Calculated volatility curve for expiration {expiration}")
                    
        except Exception as e:
            logger.error(f"Error calculating volatility curves: {e}")
    
    def _get_subscription_channels(self) -> List[str]:
        """Get list of channels to subscribe to"""
        channels = []
        
        # Add expiration channels
        for expiration in [self.config.trading.near_expiration, self.config.trading.far_expiration]:
            if expiration:
                channels.append(f"ticker.BTC-{expiration}.100ms")
        
        # Add general channels
        channels.extend([
            "markprice.options.btc_usd",
            "ticker.BTC-PERPETUAL.100ms"
        ])
        
        return channels
    
    async def _log_status(self):
        """Log current application status"""
        try:
            uptime = datetime.now() - self.status.start_time if self.status.start_time else timedelta(0)
            
            logger.info(f"Application Status - Uptime: {uptime}, "
                       f"Errors: {self.status.errors_count}, "
                       f"Strategies: {len([s for s in self.strategies if s.is_enabled()])}, "
                       f"Total Trades: {self.status.total_trades}")
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    async def shutdown(self):
        """Graceful application shutdown"""
        try:
            logger.info("Initiating application shutdown...")
            
            # Update status
            self.status.running = False
            
            # Cancel all background tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Disconnect client
            if self.client:
                await self.client.disconnect()
            
            # Log final status
            logger.info("Application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current application status"""
        try:
            uptime = datetime.now() - self.status.start_time if self.status.start_time else timedelta(0)
            
            return {
                'running': self.status.running,
                'uptime_seconds': uptime.total_seconds(),
                'start_time': self.status.start_time.isoformat() if self.status.start_time else None,
                'last_heartbeat': self.status.last_heartbeat.isoformat() if self.status.last_heartbeat else None,
                'errors_count': self.status.errors_count,
                'strategies_active': len([s for s in self.strategies if s.is_enabled()]),
                'total_trades': self.status.total_trades,
                'total_pnl': self.status.total_pnl,
                'strategies': [
                    {
                        'name': strategy.config.name,
                        'enabled': strategy.is_enabled(),
                        'performance': strategy.get_performance_metrics()
                    }
                    for strategy in self.strategies
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}


async def main():
    """Main entry point"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Create and run application
        app = TradingApp(config)
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
