"""
Configuration Management

Centralized configuration management for the trading system.
Handles loading credentials, settings, and environment-specific configurations.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    user: str = "root"
    password: str = "1234"
    database: str = "btc_options_db"
    port: int = 3306
    
    @property
    def connection_string(self) -> str:
        """Get database connection string"""
        return f"mysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DeribitConfig:
    """Deribit API configuration"""
    ws_url: str = "wss://www.deribit.com/ws/api/v2"
    api_url: str = "https://www.deribit.com/api/v2"
    testnet: bool = False
    client_id: str = ""
    private_key_path: str = "key/private.pem"
    rate_limit: int = 20  # requests per second
    timeout: int = 30
    
    @property
    def ws_url_testnet(self) -> str:
        """Get testnet WebSocket URL"""
        return "wss://test.deribit.com/ws/api/v2"
    
    @property
    def api_url_testnet(self) -> str:
        """Get testnet API URL"""
        return "https://test.deribit.com/api/v2"
    
    @property
    def effective_ws_url(self) -> str:
        """Get effective WebSocket URL based on testnet setting"""
        return self.ws_url_testnet if self.testnet else self.ws_url
    
    @property
    def effective_api_url(self) -> str:
        """Get effective API URL based on testnet setting"""
        return self.api_url_testnet if self.testnet else self.api_url


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str = ""
    chat_id: Optional[int] = None
    enabled: bool = False
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return self.enabled and bool(self.bot_token) and self.chat_id is not None


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    near_expiration: str = ""
    far_expiration: str = ""
    spread_way: str = "SHORT"  # "SHORT" or "LONG"
    execution_enabled: bool = False
    perpetual_expirations: List[str] = field(default_factory=list)
    position_size: float = 0.1
    max_positions: int = 4
    
    @property
    def is_valid(self) -> bool:
        """Check if trading configuration is valid"""
        return bool(self.near_expiration and self.far_expiration and self.spread_way)


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.2
    margin_buffer: float = 1.2
    max_daily_loss: float = 0.1
    stop_loss_percentage: float = 0.05
    
    @property
    def is_valid(self) -> bool:
        """Check if risk configuration is valid"""
        return all([
            self.max_position_size > 0,
            self.max_portfolio_risk > 0,
            self.margin_buffer > 1.0,
            self.max_daily_loss > 0,
            self.stop_loss_percentage > 0
        ])


@dataclass
class SessionConfig:
    """Session management configuration"""
    enable_multi_session: bool = False
    max_concurrent_sessions: int = 5
    session_cleanup_hours: int = 24
    auto_refresh_tokens: bool = True
    session_health_check_interval: int = 300  # seconds
    
    @property
    def is_valid(self) -> bool:
        """Check if session configuration is valid"""
        return all([
            self.max_concurrent_sessions > 0,
            self.session_cleanup_hours > 0,
            self.session_health_check_interval > 0
        ])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    deribit: DeribitConfig = field(default_factory=DeribitConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def load_from_files(cls, config_dir: str = ".") -> 'AppConfig':
        """
        Load configuration from files
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            AppConfig instance
        """
        config_path = Path(config_dir)
        
        # Initialize with defaults
        config = cls()
        
        # Load database config
        config.database = cls._load_database_config(config_path)
        
        # Load Deribit config
        config.deribit = cls._load_deribit_config(config_path)
        
        # Load Telegram config
        config.telegram = cls._load_telegram_config(config_path)
        
        # Load trading config
        config.trading = cls._load_trading_config(config_path)
        
        # Load risk config
        config.risk = cls._load_risk_config(config_path)
        
        # Load logging config
        config.logging = cls._load_logging_config(config_path)
        
        return config
    
    @classmethod
    def load_from_json(cls, config_file: str) -> 'AppConfig':
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            AppConfig instance
        """
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            return cls(
                database=DatabaseConfig(**data.get('database', {})),
                deribit=DeribitConfig(**data.get('deribit', {})),
                telegram=TelegramConfig(**data.get('telegram', {})),
                trading=TradingConfig(**data.get('trading', {})),
                risk=RiskConfig(**data.get('risk', {})),
                logging=LoggingConfig(**data.get('logging', {}))
            )
        except Exception as e:
            logger.error(f"Error loading config from JSON: {e}")
            return cls()
    
    def save_to_json(self, config_file: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            config_file: Path to save configuration
        """
        try:
            data = {
                'database': {
                    'host': self.database.host,
                    'user': self.database.user,
                    'password': self.database.password,
                    'database': self.database.database,
                    'port': self.database.port
                },
                'deribit': {
                    'ws_url': self.deribit.ws_url,
                    'api_url': self.deribit.api_url,
                    'testnet': self.deribit.testnet,
                    'client_id': self.deribit.client_id,
                    'private_key_path': self.deribit.private_key_path,
                    'rate_limit': self.deribit.rate_limit,
                    'timeout': self.deribit.timeout
                },
                'telegram': {
                    'bot_token': self.telegram.bot_token,
                    'chat_id': self.telegram.chat_id,
                    'enabled': self.telegram.enabled
                },
                'trading': {
                    'near_expiration': self.trading.near_expiration,
                    'far_expiration': self.trading.far_expiration,
                    'spread_way': self.trading.spread_way,
                    'execution_enabled': self.trading.execution_enabled,
                    'perpetual_expirations': self.trading.perpetual_expirations,
                    'position_size': self.trading.position_size,
                    'max_positions': self.trading.max_positions
                },
                'risk': {
                    'max_position_size': self.risk.max_position_size,
                    'max_portfolio_risk': self.risk.max_portfolio_risk,
                    'margin_buffer': self.risk.margin_buffer,
                    'max_daily_loss': self.risk.max_daily_loss,
                    'stop_loss_percentage': self.risk.stop_loss_percentage
                },
                'logging': {
                    'level': self.logging.level,
                    'format': self.logging.format,
                    'date_format': self.logging.date_format,
                    'file_path': self.logging.file_path,
                    'max_file_size': self.logging.max_file_size,
                    'backup_count': self.logging.backup_count
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving config to JSON: {e}")
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate database config
        if not self.database.host:
            errors.append("Database host is required")
        
        # Validate Deribit config
        if not self.deribit.client_id:
            errors.append("Deribit client ID is required")
        
        if not os.path.exists(self.deribit.private_key_path):
            errors.append(f"Private key file not found: {self.deribit.private_key_path}")
        
        # Validate trading config
        if not self.trading.is_valid:
            errors.append("Trading configuration is invalid")
        
        # Validate risk config
        if not self.risk.is_valid:
            errors.append("Risk configuration is invalid")
        
        return errors
    
    @staticmethod
    def _load_database_config(config_path: Path) -> DatabaseConfig:
        """Load database configuration"""
        try:
            # Try to load from environment variables first
            return DatabaseConfig(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', '1234'),
                database=os.getenv('DB_NAME', 'btc_options_db'),
                port=int(os.getenv('DB_PORT', '3306'))
            )
        except Exception as e:
            logger.warning(f"Error loading database config: {e}")
            return DatabaseConfig()
    
    @staticmethod
    def _load_deribit_config(config_path: Path) -> DeribitConfig:
        """Load Deribit configuration"""
        config = DeribitConfig()
        
        try:
            # Load client ID
            client_id_file = config_path / 'key' / 'client_id.txt'
            if client_id_file.exists():
                with open(client_id_file, 'r') as f:
                    config.client_id = f.readline().strip()
            
            # Load private key path
            private_key_file = config_path / 'key' / 'private.pem'
            if private_key_file.exists():
                config.private_key_path = str(private_key_file)
            
            # Load from environment variables
            config.testnet = os.getenv('DERIBIT_TESTNET', 'false').lower() == 'true'
            config.rate_limit = int(os.getenv('DERIBIT_RATE_LIMIT', '20'))
            config.timeout = int(os.getenv('DERIBIT_TIMEOUT', '30'))
            
        except Exception as e:
            logger.warning(f"Error loading Deribit config: {e}")
        
        return config
    
    @staticmethod
    def _load_telegram_config(config_path: Path) -> TelegramConfig:
        """Load Telegram configuration"""
        config = TelegramConfig()
        
        try:
            # Load bot token
            bot_token_file = config_path / 'key' / 'bot_token.txt'
            if bot_token_file.exists():
                with open(bot_token_file, 'r') as f:
                    config.bot_token = f.readline().strip()
                    config.enabled = True
            
            # Load chat ID
            chat_id_file = config_path / 'key' / 'chat_id.txt'
            if chat_id_file.exists():
                with open(chat_id_file, 'r') as f:
                    chat_id_str = f.readline().strip()
                    try:
                        config.chat_id = int(chat_id_str)
                    except ValueError:
                        logger.warning(f"Invalid chat ID: {chat_id_str}")
            
        except Exception as e:
            logger.warning(f"Error loading Telegram config: {e}")
        
        return config
    
    @staticmethod
    def _load_trading_config(config_path: Path) -> TradingConfig:
        """Load trading configuration"""
        config = TradingConfig()
        
        try:
            # Load from environment variables
            config.near_expiration = os.getenv('TRADING_NEAR_EXPIRATION', '')
            config.far_expiration = os.getenv('TRADING_FAR_EXPIRATION', '')
            config.spread_way = os.getenv('TRADING_SPREAD_WAY', 'SHORT')
            config.execution_enabled = os.getenv('TRADING_EXECUTION_ENABLED', 'false').lower() == 'true'
            config.position_size = float(os.getenv('TRADING_POSITION_SIZE', '0.1'))
            config.max_positions = int(os.getenv('TRADING_MAX_POSITIONS', '4'))
            
        except Exception as e:
            logger.warning(f"Error loading trading config: {e}")
        
        return config
    
    @staticmethod
    def _load_risk_config(config_path: Path) -> RiskConfig:
        """Load risk configuration"""
        config = RiskConfig()
        
        try:
            # Load from environment variables
            config.max_position_size = float(os.getenv('RISK_MAX_POSITION_SIZE', '0.1'))
            config.max_portfolio_risk = float(os.getenv('RISK_MAX_PORTFOLIO_RISK', '0.2'))
            config.margin_buffer = float(os.getenv('RISK_MARGIN_BUFFER', '1.2'))
            config.max_daily_loss = float(os.getenv('RISK_MAX_DAILY_LOSS', '0.1'))
            config.stop_loss_percentage = float(os.getenv('RISK_STOP_LOSS_PERCENTAGE', '0.05'))
            
        except Exception as e:
            logger.warning(f"Error loading risk config: {e}")
        
        return config
    
    @staticmethod
    def _load_logging_config(config_path: Path) -> LoggingConfig:
        """Load logging configuration"""
        config = LoggingConfig()
        
        try:
            # Load from environment variables
            config.level = os.getenv('LOG_LEVEL', 'INFO')
            config.file_path = os.getenv('LOG_FILE_PATH')
            config.max_file_size = int(os.getenv('LOG_MAX_FILE_SIZE', str(10 * 1024 * 1024)))
            config.backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
            
        except Exception as e:
            logger.warning(f"Error loading logging config: {e}")
        
        return config


class ConfigManager:
    """Configuration manager singleton"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def load_config(self, config_dir: str = ".", config_file: Optional[str] = None) -> AppConfig:
        """
        Load configuration
        
        Args:
            config_dir: Directory containing configuration files
            config_file: Optional JSON configuration file
            
        Returns:
            AppConfig instance
        """
        if config_file and os.path.exists(config_file):
            self._config = AppConfig.load_from_json(config_file)
        else:
            self._config = AppConfig.load_from_files(config_dir)
        
        return self._config
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = AppConfig.load_from_files()
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        if self._config is None:
            self._config = AppConfig.load_from_files()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def save_config(self, config_file: str) -> None:
        """Save current configuration to file"""
        if self._config:
            self._config.save_to_json(config_file)
