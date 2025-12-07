"""
Deribit API Client

Clean interface for interacting with Deribit's REST and WebSocket APIs.
Handles authentication, rate limiting, error management, and data parsing.
"""

import asyncio
import aiohttp
import websockets
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

from ..models.market_data import MarketData, OptionContract, OrderBook, OrderSide, OrderType
from ..utils.config import DeribitConfig

logger = logging.getLogger(__name__)


@dataclass
class DeribitError(Exception):
    """Base exception for Deribit API errors"""
    message: str
    code: Optional[int] = None
    
    def __str__(self):
        if self.code:
            return f"DeribitError {self.code}: {self.message}"
        return f"DeribitError: {self.message}"


@dataclass
class DeribitConnectionError(DeribitError):
    """Connection-related errors"""
    pass


@dataclass
class DeribitAuthError(DeribitError):
    """Authentication errors"""
    pass


class DeribitAuth:
    """Handles Deribit authentication with session management"""
    
    def __init__(self, client_id: str, private_key_path: Optional[str] = None, 
                 private_key: Optional[str] = None, session_name: Optional[str] = None):
        """
        Initialize Deribit authentication
        
        Args:
            client_id: Deribit client ID
            private_key_path: Path to RSA private key .pem file (optional if private_key provided)
            private_key: Direct secret key string (optional if private_key_path provided)
            session_name: Optional session name
        """
        self.client_id = client_id
        self.private_key_path = private_key_path
        self.private_key_string = private_key
        self.session_name = session_name or "default_session"
        self.private_key = None  # RSA key object (for .pem files)
        self.access_token = None
        self.refresh_token = None
        self.refresh_token_expiry = None
        self.token_type = "bearer"
        self.scope = None
        self._load_private_key()
    
    def _load_private_key(self):
        """Load RSA private key from file or use direct secret key"""
        try:
            from cryptography.hazmat.primitives import serialization
            
            # If direct secret key string is provided, use it
            if self.private_key_string:
                logger.info("Using direct secret key for authentication")
                # Store the string for later use in authentication
                self.private_key = self.private_key_string
                return
            
            # Otherwise, load from .pem file
            if not self.private_key_path:
                raise DeribitAuthError("Either private_key_path or private_key must be provided")
            
            if not os.path.exists(self.private_key_path):
                raise DeribitAuthError(f"Private key file not found: {self.private_key_path}")
            
            with open(self.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.info(f"Private key loaded successfully from {self.private_key_path}")
            
        except DeribitAuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise DeribitAuthError(f"Failed to load private key: {e}")
    
    def generate_signature(self) -> Dict[str, Any]:
        """Generate authentication signature"""
        try:
            import secrets
            import base64
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            timestamp = round(datetime.now().timestamp() * 1000)
            nonce = secrets.token_hex(16)
            data = ""
            
            # Prepare data to sign
            data_to_sign = bytes(f'{timestamp}\n{nonce}\n{data}', "latin-1")
            
            # Check if using direct secret key (string) or RSA key (object)
            if isinstance(self.private_key, str):
                # Using client_secret authentication (direct key string)
                import hmac
                import hashlib
                
                # Deribit uses HMAC-SHA256 for secret key authentication
                message = f"{timestamp}\n{nonce}\n{data}"
                signature = hmac.new(
                    self.private_key.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
                encoded_signature = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
            else:
                # Using RSA key authentication (.pem file)
                signature = self.private_key.sign(
                    data_to_sign,
                    padding.PKCS1v15(),
                    hashes.SHA256()
                )
                encoded_signature = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
            
            return {
                'timestamp': timestamp,
                'nonce': nonce,
                'data': data,
                'signature': encoded_signature
            }
        except Exception as e:
            logger.error(f"Failed to generate signature: {e}")
            raise DeribitAuthError(f"Failed to generate signature: {e}")
    
    async def authenticate(self, session: aiohttp.ClientSession, api_url: Optional[str] = None) -> bool:
        """Authenticate with Deribit API"""
        try:
            # Determine grant type based on key type
            is_client_secret = isinstance(self.private_key, str)
            
            if is_client_secret:
                # Client secret authentication (simpler format - no signature needed)
                # For client_secret, Deribit API expects just client_id and client_secret
                payload = {
                    "jsonrpc": "2.0",
                    "id": 9929,
                    "method": "public/auth",
                    "params": {
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.private_key_string
                    }
                }
            else:
                # Client signature authentication (RSA key)
                auth_data = self.generate_signature()
                payload = {
                    "jsonrpc": "2.0",
                    "id": 9929,
                    "method": "public/auth",
                    "params": {
                        "grant_type": "client_signature",
                        "client_id": self.client_id,
                        "timestamp": auth_data['timestamp'],
                        "signature": auth_data['signature'],
                        "nonce": auth_data['nonce'],
                        "data": auth_data['data']
                    }
                }
            
            # Use provided API URL or default to mainnet
            url = api_url or "https://www.deribit.com/api/v2"
            
            async with session.post(
                url,
                json=payload
            ) as response:
                result = await response.json()
                
                if 'error' in result:
                    raise DeribitAuthError(f"Authentication failed: {result['error']}")
                
                auth_result = result.get('result', {})
                self.access_token = auth_result.get('access_token')
                self.refresh_token = auth_result.get('refresh_token')
                
                # Calculate refresh expiry time
                expires_in = auth_result.get('expires_in', 3600)
                self.refresh_token_expiry = datetime.now() + timedelta(seconds=expires_in - 240)
                
                logger.info("Successfully authenticated with Deribit")
                return True
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise DeribitAuthError(f"Authentication failed: {e}")
    
    async def refresh_auth(self, session: aiohttp.ClientSession, api_url: Optional[str] = None) -> bool:
        """Refresh authentication token"""
        try:
            if not self.refresh_token:
                return await self.authenticate(session, api_url)
            
            payload = {
                "jsonrpc": "2.0",
                "id": 9929,
                "method": "public/auth",
                "params": {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token
                }
            }
            
            # Use provided API URL or default to mainnet
            url = api_url or "https://www.deribit.com/api/v2"
            
            async with session.post(
                url,
                json=payload
            ) as response:
                result = await response.json()
                
                if 'error' in result:
                    logger.warning("Token refresh failed, re-authenticating")
                    return await self.authenticate(session, api_url)
                
                auth_result = result.get('result', {})
                self.access_token = auth_result.get('access_token')
                self.refresh_token = auth_result.get('refresh_token')
                
                expires_in = auth_result.get('expires_in', 3600)
                self.refresh_token_expiry = datetime.now() + timedelta(seconds=expires_in - 240)
                
                logger.info("Successfully refreshed authentication token")
                return True
                
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return await self.authenticate(session)
    
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if not self.access_token:
            raise DeribitAuthError("Not authenticated")
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def needs_refresh(self) -> bool:
        """Check if token needs refresh"""
        if not self.refresh_token_expiry:
            return True
        return datetime.now() >= self.refresh_token_expiry
    
    async def fork_token(self, session_name: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Generate a token for a new named session using fork_token API
        
        Args:
            session_name: Name for the new session
            session: HTTP session for making requests
            
        Returns:
            Dictionary containing new token information
        """
        try:
            if not self.refresh_token:
                raise DeribitAuthError("No refresh token available for forking")
            
            payload = {
                "jsonrpc": "2.0",
                "id": 9930,
                "method": "public/fork_token",
                "params": {
                    "refresh_token": self.refresh_token,
                    "session_name": session_name
                }
            }
            
            async with session.post(
                "https://www.deribit.com/api/v2",
                json=payload
            ) as response:
                result = await response.json()
                
                if 'error' in result:
                    raise DeribitAuthError(f"Fork token failed: {result['error']}")
                
                fork_result = result.get('result', {})
                
                # Update current session with new token info
                self.access_token = fork_result.get('access_token')
                self.refresh_token = fork_result.get('refresh_token')
                self.token_type = fork_result.get('token_type', 'bearer')
                self.scope = fork_result.get('scope')
                
                # Calculate refresh expiry time
                expires_in = fork_result.get('expires_in', 3600)
                self.refresh_token_expiry = datetime.now() + timedelta(seconds=expires_in - 240)
                
                logger.info(f"Successfully forked token for session: {session_name}")
                
                return {
                    'access_token': self.access_token,
                    'refresh_token': self.refresh_token,
                    'expires_in': expires_in,
                    'token_type': self.token_type,
                    'scope': self.scope,
                    'session_name': session_name
                }
                
        except Exception as e:
            logger.error(f"Error forking token: {e}")
            raise DeribitAuthError(f"Fork token failed: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            'session_name': self.session_name,
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'token_type': self.token_type,
            'scope': self.scope,
            'expires_at': self.refresh_token_expiry.isoformat() if self.refresh_token_expiry else None,
            'needs_refresh': self.needs_refresh()
        }


class DeribitClient:
    """
    Main client for interacting with Deribit API
    
    Provides methods for:
    - Market data retrieval
    - Options chain analysis
    - Order management
    - Portfolio operations
    """
    
    def __init__(self, config: DeribitConfig, auth: DeribitAuth):
        self.config = config
        self.auth = auth
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
        self._message_handlers: Dict[str, Callable] = {}
        self._listening = False
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session and authenticate"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        await self.auth.authenticate(self.session, self.config.effective_api_url)
        logger.info("Deribit client connected successfully")
    
    async def disconnect(self):
        """Clean up connections"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
        logger.info("Deribit client disconnected")
    
    async def _make_jsonrpc_request(self, method: str, params: Optional[Dict] = None, 
                                    request_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Make JSON-RPC 2.0 request to Deribit API
        
        Args:
            method: JSON-RPC method name (e.g., "public/get_instruments")
            params: Method parameters
            request_id: Optional request ID (auto-generated if not provided)
            
        Returns:
            Result from the API response
        """
        if request_id is None:
            import random
            request_id = random.randint(1, 1000000)
        
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        async with self._rate_limiter:
            # Check if we need to refresh auth
            if self.auth.needs_refresh():
                await self.auth.refresh_auth(self.session, self.config.effective_api_url)
            
            url = self.config.effective_api_url
            headers = self.auth.get_headers()
            
            try:
                async with self.session.post(url, json=payload, headers=headers) as response:
                    result = await response.json()
                    
                    # Check for HTTP errors
                    if response.status != 200:
                        error_info = result.get('error', {})
                        error_msg = error_info.get('message', 'Unknown error')
                        error_data = error_info.get('data', {})
                        full_error = f"HTTP {response.status}: {error_msg}"
                        if error_data:
                            full_error += f" (data: {error_data})"
                        logger.error(f"HTTP error: {full_error}")
                        raise DeribitError(full_error, response.status)
                    
                    # Check for JSON-RPC errors
                    if 'error' in result:
                        error_info = result['error']
                        error_msg = error_info.get('message', 'Unknown error')
                        error_code = error_info.get('code')
                        error_data = error_info.get('data', {})
                        logger.error(f"Deribit API error: {error_code} - {error_msg} (data: {error_data})")
                        raise DeribitError(error_msg, error_code)
                    
                    return result.get('result', {})
                    
            except DeribitError:
                raise
            except aiohttp.ClientError as e:
                logger.error(f"JSON-RPC request failed: {e}")
                raise DeribitConnectionError(f"Request failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in JSON-RPC request: {e}")
                raise DeribitError(f"Unexpected error: {e}")
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None,
                          data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated HTTP request with rate limiting (legacy method)
        
        Note: Deribit API v2 uses JSON-RPC format. This method converts REST-style
        calls to JSON-RPC format for backward compatibility.
        """
        # Convert REST-style endpoint to JSON-RPC method
        jsonrpc_method = endpoint.replace("/", "/")
        if not jsonrpc_method.startswith("public/") and not jsonrpc_method.startswith("private/"):
            # Assume it's a public method if not specified
            jsonrpc_method = f"public/{jsonrpc_method}"
        
        # Use params if provided, otherwise use data
        request_params = params if params else (data if data else {})
        
        return await self._make_jsonrpc_request(jsonrpc_method, request_params)
    
    # Market Data Methods
    async def get_instruments(self, currency: str = "BTC", 
                           kind: str = "option") -> List[Dict]:
        """Get available instruments"""
        return await self._make_jsonrpc_request(
            "public/get_instruments",
            {"currency": currency, "kind": kind}
        )
    
    async def get_option_chain(self, currency: str = "BTC", 
                             expiration_date: Optional[str] = None) -> List[OptionContract]:
        """Get option chain for given currency and expiration"""
        params = {"currency": currency, "kind": "option"}
        if expiration_date:
            params["expiration_date"] = expiration_date
            
        instruments = await self._make_jsonrpc_request(
            "public/get_instruments", params
        )
        
        return [self._parse_option_contract(instr) for instr in instruments]
    
    async def get_ticker(self, instrument_name: str) -> MarketData:
        """Get current ticker data for instrument"""
        data = await self._make_jsonrpc_request(
            "public/ticker",
            {"instrument_name": instrument_name}
        )
        return self._parse_market_data(data)
    
    async def get_order_book(self, instrument_name: str, depth: int = 20) -> OrderBook:
        """Get order book for instrument"""
        data = await self._make_jsonrpc_request(
            "public/get_order_book",
            {"instrument_name": instrument_name, "depth": depth}
        )
        return self._parse_order_book(data)
    
    async def get_historical_volatility(self, currency: str = "BTC", 
                                     resolution: str = "1D",
                                     start_timestamp: Optional[int] = None,
                                     end_timestamp: Optional[int] = None) -> List[Dict]:
        """Get historical volatility data"""
        params = {
            "currency": currency,
            "resolution": resolution
        }
        if start_timestamp:
            params["start_timestamp"] = start_timestamp
        if end_timestamp:
            params["end_timestamp"] = end_timestamp
            
        return await self._make_jsonrpc_request(
            "public/get_historical_volatility", params
        )
    
    async def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        resolution: int = 10
    ) -> Dict[str, Any]:
        """
        Get TradingView chart data for an instrument (futures or options).
        
        Args:
            instrument_name: Instrument name (e.g., "BTC-25SEP26-150000-C" or "BTC-25SEP26")
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            resolution: Resolution in seconds (default: 10)
            
        Returns:
            Dictionary containing OHLCV data with keys:
            - close: List of closing prices
            - high: List of high prices
            - low: List of low prices
            - open: List of open prices
            - volume: List of volumes
            - cost: List of costs
            - ticks: List of timestamps in milliseconds
            - status: Status string
        """
        # Deribit API expects timestamps as strings
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": str(start_timestamp),
            "end_timestamp": str(end_timestamp),
            "resolution": resolution
        }
        
        return await self._make_jsonrpc_request("public/get_tradingview_chart_data", params)
    
    # Note: get_option_greeks method doesn't exist in Deribit API v2
    # Greeks are included in ticker data (get_ticker method)
    
    async def get_implied_volatility(self, instrument_name: str) -> float:
        """Get implied volatility for option"""
        ticker = await self.get_ticker(instrument_name)
        return ticker.mark_iv or 0.0
    
    # Portfolio methods (require authentication)
    async def get_positions(self, currency: str = "BTC") -> List[Dict]:
        """Get current positions"""
        return await self._make_jsonrpc_request(
            "private/get_positions",
            {"currency": currency}
        )
    
    async def get_account_summary(self, currency: str = "BTC") -> Dict:
        """Get account summary"""
        return await self._make_jsonrpc_request(
            "private/get_account_summary",
            {"currency": currency}
        )
    
    async def place_order(self, instrument_name: str, amount: float,
                        order_type: OrderType = OrderType.MARKET, 
                        price: Optional[float] = None,
                        side: OrderSide = OrderSide.BUY,
                        time_in_force: Optional[str] = "good_til_cancelled") -> Dict:
        """
        Place order
        
        Args:
            instrument_name: Instrument to trade
            amount: Order amount (in quote currency USD for perpetuals/futures, 
                    in base currency for options/spot)
            order_type: Order type (MARKET or LIMIT)
            price: Price for limit orders (required for LIMIT orders)
            side: Order side (BUY or SELL)
            time_in_force: Time in force (default: "good_til_cancelled")
            
        Returns:
            Order result dictionary with 'order' and 'trades' keys
        """
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": order_type.value
        }
        
        # Price is required for limit orders
        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Price is required for limit orders")
            params["price"] = price
        
        # Add time_in_force if provided
        if time_in_force:
            params["time_in_force"] = time_in_force
        
        # Use buy or sell method based on side
        method = "private/buy" if side == OrderSide.BUY else "private/sell"
        result = await self._make_jsonrpc_request(method, params)
        
        # Result should already have 'order' and 'trades' structure
        return result
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel order"""
        return await self._make_jsonrpc_request(
            "private/cancel",
            {"order_id": order_id}
        )
    
    async def simulate_portfolio(self, positions: Dict[str, float], 
                               currency: str = "BTC") -> Dict:
        """Simulate portfolio with given positions"""
        return await self._make_jsonrpc_request(
            "private/simulate_portfolio",
            {
                "currency": currency,
                "add_positions": "false",
                "simulated_positions": positions
            }
        )
    
    async def fork_token(self, session_name: str) -> Dict[str, Any]:
        """
        Generate a token for a new named session
        
        Args:
            session_name: Name for the new session
            
        Returns:
            Dictionary containing new token information
        """
        if not self.session:
            raise DeribitConnectionError("No active session for token forking")
        
        return await self.auth.fork_token(session_name, self.session)
    
    def get_auth_info(self) -> Dict[str, Any]:
        """Get current authentication information"""
        return self.auth.get_session_info()
    
    # Additional Public API Methods
    async def get_currencies(self) -> List[Dict]:
        """Get list of all supported currencies"""
        return await self._make_jsonrpc_request("public/get_currencies")
    
    async def get_book_summary_by_currency(self, currency: str = "BTC", 
                                          kind: Optional[str] = None) -> List[Dict]:
        """Get book summary by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        return await self._make_jsonrpc_request("public/get_book_summary_by_currency", params)
    
    async def get_book_summary_by_instrument(self, instrument_name: str) -> Dict:
        """Get book summary by instrument"""
        return await self._make_jsonrpc_request(
            "public/get_book_summary_by_instrument",
            {"instrument_name": instrument_name}
        )
    
    async def get_index_price(self, index_name: str) -> Dict:
        """Get index price"""
        return await self._make_jsonrpc_request(
            "public/get_index_price",
            {"index_name": index_name}
        )
    
    async def get_index_price_names(self, extended: bool = False) -> List[str]:
        """Get available index price names"""
        return await self._make_jsonrpc_request(
            "public/get_index_price_names",
            {"extended": extended}
        )
    
    async def get_delivery_prices(self, index_name: str, offset: Optional[int] = None,
                                 count: Optional[int] = None) -> Dict:
        """Get delivery prices"""
        params = {"index_name": index_name}
        if offset is not None:
            params["offset"] = offset
        if count is not None:
            params["count"] = count
        return await self._make_jsonrpc_request("public/get_delivery_prices", params)
    
    async def get_last_trades_by_currency(self, currency: str = "BTC",
                                         kind: Optional[str] = None,
                                         start_id: Optional[str] = None,
                                         end_id: Optional[str] = None,
                                         count: Optional[int] = None,
                                         include_old: Optional[bool] = None,
                                         sorting: Optional[str] = None) -> Dict:
        """Get last trades by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if start_id:
            params["start_id"] = start_id
        if end_id:
            params["end_id"] = end_id
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("public/get_last_trades_by_currency", params)
    
    async def get_last_trades_by_currency_and_time(self, currency: str,
                                                   kind: Optional[str] = None,
                                                   start_timestamp: Optional[int] = None,
                                                   end_timestamp: Optional[int] = None,
                                                   count: Optional[int] = None,
                                                   include_old: Optional[bool] = None,
                                                   sorting: Optional[str] = None) -> Dict:
        """Get last trades by currency and time range"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("public/get_last_trades_by_currency_and_time", params)
    
    async def get_last_trades_by_instrument(self, instrument_name: str,
                                          start_seq: Optional[int] = None,
                                          end_seq: Optional[int] = None,
                                          count: Optional[int] = None,
                                          include_old: Optional[bool] = None,
                                          sorting: Optional[str] = None) -> Dict:
        """Get last trades by instrument"""
        params = {"instrument_name": instrument_name}
        if start_seq is not None:
            params["start_seq"] = start_seq
        if end_seq is not None:
            params["end_seq"] = end_seq
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("public/get_last_trades_by_instrument", params)
    
    async def get_last_trades_by_instrument_and_time(self, instrument_name: str,
                                                     start_timestamp: Optional[int] = None,
                                                     end_timestamp: Optional[int] = None,
                                                     count: Optional[int] = None,
                                                     include_old: Optional[bool] = None,
                                                     sorting: Optional[str] = None) -> Dict:
        """Get last trades by instrument and time range"""
        params = {"instrument_name": instrument_name}
        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("public/get_last_trades_by_instrument_and_time", params)
    
    # Note: get_index method doesn't exist in Deribit API v2
    # Use get_index_price instead with index_name parameter
    
    async def get_volatility_index_data(self, currency: str = "BTC",
                                       start_timestamp: Optional[int] = None,
                                       end_timestamp: Optional[int] = None,
                                       resolution: Optional[int] = None) -> Dict:
        """Get volatility index data"""
        params = {"currency": currency}
        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        if resolution is not None:
            params["resolution"] = resolution
        return await self._make_jsonrpc_request("public/get_volatility_index_data", params)
    
    # Note: get_public_trades method doesn't exist in Deribit API v2
    # Use get_last_trades_by_instrument instead
    
    # Note: get_public_trading_statistics method doesn't exist in Deribit API v2
    # Statistics are available in ticker data
    
    async def get_funding_rate_history(self, instrument_name: str,
                                     start_timestamp: Optional[int] = None,
                                     end_timestamp: Optional[int] = None) -> List[Dict]:
        """Get funding rate history"""
        params = {"instrument_name": instrument_name}
        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        return await self._make_jsonrpc_request("public/get_funding_rate_history", params)
    
    async def get_funding_chart_data(self, instrument_name: str, length: str = "24h") -> Dict:
        """Get funding chart data"""
        return await self._make_jsonrpc_request(
            "public/get_funding_chart_data",
            {"instrument_name": instrument_name, "length": length}
        )
    
    async def get_apr_history(self, currency: str) -> Dict:
        """Get APR (Annual Percentage Rate) history"""
        return await self._make_jsonrpc_request(
            "public/get_apr_history",
            {"currency": currency.lower()}
        )
    
    async def get_block_rfq_trades(self, currency: str) -> List[Dict]:
        """Get block RFQ trades"""
        return await self._make_jsonrpc_request(
            "public/get_block_rfq_trades",
            {"currency": currency}
        )
    
    async def get_funding_rate_value(self, instrument_name: str,
                                   start_timestamp: Optional[int] = None,
                                   end_timestamp: Optional[int] = None) -> Dict:
        """Get funding rate value for instrument"""
        params = {"instrument_name": instrument_name}
        if start_timestamp is not None:
            params["start_timestamp"] = str(start_timestamp)
        if end_timestamp is not None:
            params["end_timestamp"] = str(end_timestamp)
        return await self._make_jsonrpc_request("public/get_funding_rate_value", params)
    
    # Note: get_historical_funding_rates method doesn't exist in Deribit API v2
    # Use get_funding_rate_history for a specific instrument instead
    
    # Additional Private API Methods
    async def get_position(self, instrument_name: str) -> Dict:
        """Get single position by instrument"""
        return await self._make_jsonrpc_request(
            "private/get_position",
            {"instrument_name": instrument_name}
        )
    
    async def get_open_orders_by_currency(self, currency: str = "BTC",
                                         kind: Optional[str] = None,
                                         type: Optional[str] = None) -> List[Dict]:
        """Get open orders by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if type:
            params["type"] = type
        return await self._make_jsonrpc_request("private/get_open_orders_by_currency", params)
    
    async def get_open_orders_by_instrument(self, instrument_name: str,
                                           type: Optional[str] = None) -> List[Dict]:
        """Get open orders by instrument"""
        params = {"instrument_name": instrument_name}
        if type:
            params["type"] = type
        return await self._make_jsonrpc_request("private/get_open_orders_by_instrument", params)
    
    async def get_order_history_by_currency(self, currency: str = "BTC",
                                          kind: Optional[str] = None,
                                          count: Optional[int] = None,
                                          offset: Optional[int] = None) -> Dict:
        """Get order history by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset
        return await self._make_jsonrpc_request("private/get_order_history_by_currency", params)
    
    async def get_order_history_by_instrument(self, instrument_name: str,
                                             count: Optional[int] = None,
                                             offset: Optional[int] = None,
                                             include_old: Optional[bool] = None) -> Dict:
        """Get order history by instrument"""
        params = {"instrument_name": instrument_name}
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset
        if include_old is not None:
            params["include_old"] = include_old
        return await self._make_jsonrpc_request("private/get_order_history_by_instrument", params)
    
    async def get_order_state(self, order_id: str) -> Dict:
        """Get order state by order ID"""
        return await self._make_jsonrpc_request(
            "private/get_order_state",
            {"order_id": order_id}
        )
    
    async def get_user_trades_by_currency(self, currency: str = "BTC",
                                        kind: Optional[str] = None,
                                        start_id: Optional[str] = None,
                                        end_id: Optional[str] = None,
                                        count: Optional[int] = None,
                                        include_old: Optional[bool] = None,
                                        sorting: Optional[str] = None) -> Dict:
        """Get user trades by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if start_id:
            params["start_id"] = start_id
        if end_id:
            params["end_id"] = end_id
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("private/get_user_trades_by_currency", params)
    
    async def get_user_trades_by_instrument(self, instrument_name: str,
                                           start_seq: Optional[int] = None,
                                           end_seq: Optional[int] = None,
                                           count: Optional[int] = None,
                                           include_old: Optional[bool] = None,
                                           sorting: Optional[str] = None) -> Dict:
        """Get user trades by instrument"""
        params = {"instrument_name": instrument_name}
        if start_seq is not None:
            params["start_seq"] = start_seq
        if end_seq is not None:
            params["end_seq"] = end_seq
        if count is not None:
            params["count"] = count
        if include_old is not None:
            params["include_old"] = include_old
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("private/get_user_trades_by_instrument", params)
    
    async def get_user_trades_by_order(self, order_id: str,
                                      sorting: Optional[str] = None) -> Dict:
        """Get user trades by order ID"""
        params = {"order_id": order_id}
        if sorting:
            params["sorting"] = sorting
        return await self._make_jsonrpc_request("private/get_user_trades_by_order", params)
    
    async def get_settlement_history_by_currency(self, currency: str = "BTC",
                                               count: Optional[int] = None,
                                               continuation: Optional[str] = None) -> Dict:
        """Get settlement history by currency"""
        params = {"currency": currency}
        if count is not None:
            params["count"] = count
        if continuation:
            params["continuation"] = continuation
        return await self._make_jsonrpc_request("private/get_settlement_history_by_currency", params)
    
    async def get_settlement_history_by_instrument(self, instrument_name: str,
                                                   count: Optional[int] = None,
                                                   continuation: Optional[str] = None) -> Dict:
        """Get settlement history by instrument"""
        params = {"instrument_name": instrument_name}
        if count is not None:
            params["count"] = count
        if continuation:
            params["continuation"] = continuation
        return await self._make_jsonrpc_request("private/get_settlement_history_by_instrument", params)
    
    # Note: get_delivery_history method doesn't exist in Deribit API v2
    # Delivery information is available in settlement history
    
    async def buy(self, instrument_name: str, amount: float,
                 type: str = "market", price: Optional[float] = None,
                 label: Optional[str] = None, time_in_force: str = "good_til_cancelled",
                 max_show: Optional[float] = None, post_only: bool = False,
                 reduce_only: bool = False, stop_price: Optional[float] = None,
                 trigger: Optional[str] = None, advanced: Optional[str] = None) -> Dict:
        """Place buy order (simplified interface)"""
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": type,
            "time_in_force": time_in_force,
            "post_only": post_only,
            "reduce_only": reduce_only
        }
        if price:
            params["price"] = price
        if label:
            params["label"] = label
        if max_show is not None:
            params["max_show"] = max_show
        if stop_price:
            params["stop_price"] = stop_price
        if trigger:
            params["trigger"] = trigger
        if advanced:
            params["advanced"] = advanced
        return await self._make_jsonrpc_request("private/buy", params)
    
    async def sell(self, instrument_name: str, amount: float,
                  type: str = "market", price: Optional[float] = None,
                  label: Optional[str] = None, time_in_force: str = "good_til_cancelled",
                  max_show: Optional[float] = None, post_only: bool = False,
                  reduce_only: bool = False, stop_price: Optional[float] = None,
                  trigger: Optional[str] = None, advanced: Optional[str] = None) -> Dict:
        """Place sell order (simplified interface)"""
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": type,
            "time_in_force": time_in_force,
            "post_only": post_only,
            "reduce_only": reduce_only
        }
        if price:
            params["price"] = price
        if label:
            params["label"] = label
        if max_show is not None:
            params["max_show"] = max_show
        if stop_price:
            params["stop_price"] = stop_price
        if trigger:
            params["trigger"] = trigger
        if advanced:
            params["advanced"] = advanced
        return await self._make_jsonrpc_request("private/sell", params)
    
    async def edit(self, order_id: str, amount: Optional[float] = None,
                  price: Optional[float] = None, post_only: Optional[bool] = None,
                  advanced: Optional[str] = None, stop_price: Optional[float] = None) -> Dict:
        """Edit existing order"""
        params = {"order_id": order_id}
        if amount is not None:
            params["amount"] = amount
        if price is not None:
            params["price"] = price
        if post_only is not None:
            params["post_only"] = post_only
        if advanced:
            params["advanced"] = advanced
        if stop_price is not None:
            params["stop_price"] = stop_price
        return await self._make_jsonrpc_request("private/edit", params)
    
    async def cancel_all(self, currency: Optional[str] = None,
                        kind: Optional[str] = None, type: Optional[str] = None) -> Dict:
        """Cancel all orders"""
        params = {}
        if currency:
            params["currency"] = currency
        if kind:
            params["kind"] = kind
        if type:
            params["type"] = type
        return await self._make_jsonrpc_request("private/cancel_all", params)
    
    async def cancel_all_by_currency(self, currency: str = "BTC",
                                    kind: Optional[str] = None,
                                    type: Optional[str] = None) -> Dict:
        """Cancel all orders by currency"""
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        if type:
            params["type"] = type
        return await self._make_jsonrpc_request("private/cancel_all_by_currency", params)
    
    async def cancel_all_by_instrument(self, instrument_name: str,
                                      type: Optional[str] = None) -> Dict:
        """Cancel all orders by instrument"""
        params = {"instrument_name": instrument_name}
        if type:
            params["type"] = type
        return await self._make_jsonrpc_request("private/cancel_all_by_instrument", params)
    
    async def cancel_by_label(self, label: str) -> Dict:
        """Cancel order by label"""
        return await self._make_jsonrpc_request("private/cancel_by_label", {"label": label})
    
    async def close_position(self, instrument_name: str,
                           type: str = "market", price: Optional[float] = None) -> Dict:
        """Close position"""
        params = {"instrument_name": instrument_name, "type": type}
        if price:
            params["price"] = price
        return await self._make_jsonrpc_request("private/close_position", params)
    
    async def get_margins(self, instrument_name: str, amount: float, price: float) -> Dict:
        """Get margins for position"""
        return await self._make_jsonrpc_request(
            "private/get_margins",
            {
                "instrument_name": instrument_name,
                "amount": amount,
                "price": price
            }
        )
    
    async def change_position_size(self, instrument_name: str, amount: float,
                                  type: str = "market", price: Optional[float] = None) -> Dict:
        """Change position size"""
        params = {"instrument_name": instrument_name, "amount": amount, "type": type}
        if price:
            params["price"] = price
        return await self._make_jsonrpc_request("private/change_position_size", params)
    
    async def get_trade_volumes(self, extended: bool = False) -> Dict:
        """Get trade volumes"""
        return await self._make_jsonrpc_request("public/get_trade_volumes", {"extended": extended})
    
    # Note: The following methods don't exist in Deribit API v2:
    # - get_user_portfolio_margins
    # - get_user_fees  
    # - get_user_delivery_fees
    # Fee and margin information is available in account_summary and positions
    
    async def get_deposits(self, currency: str = "BTC", count: Optional[int] = None,
                          offset: Optional[int] = None) -> Dict:
        """Get deposit history"""
        params = {"currency": currency}
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset
        return await self._make_jsonrpc_request("private/get_deposits", params)
    
    async def get_withdrawals(self, currency: str = "BTC", count: Optional[int] = None,
                             offset: Optional[int] = None) -> Dict:
        """Get withdrawal history"""
        params = {"currency": currency}
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset
        return await self._make_jsonrpc_request("private/get_withdrawals", params)
    
    async def get_transfers(self, currency: str = "BTC", count: Optional[int] = None,
                           offset: Optional[int] = None) -> Dict:
        """Get transfer history"""
        params = {"currency": currency}
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset
        return await self._make_jsonrpc_request("private/get_transfers", params)
    
    # Note: get_currency method doesn't exist in Deribit API v2
    # Currency information is available in account_summary
    
    async def get_subaccounts(self, with_portfolio: bool = False) -> List[Dict]:
        """Get subaccounts"""
        params = {}
        if with_portfolio:
            params["with_portfolio"] = with_portfolio
        return await self._make_jsonrpc_request("private/get_subaccounts", params)
    
    async def get_subaccounts_details(self, currency: str, with_open_orders: bool = False) -> Dict:
        """Get subaccounts details"""
        return await self._make_jsonrpc_request(
            "private/get_subaccounts_details",
            {"currency": currency, "with_open_orders": with_open_orders}
        )
    
    async def create_subaccount(self, name: str) -> Dict:
        """Create subaccount"""
        return await self._make_jsonrpc_request("private/create_subaccount", {"name": name})
    
    async def change_subaccount_name(self, sid: int, name: str) -> Dict:
        """Change subaccount name"""
        return await self._make_jsonrpc_request(
            "private/change_subaccount_name",
            {"sid": sid, "name": name}
        )
    
    # Note: get_subaccounts_summary method doesn't exist in Deribit API v2
    # Use get_subaccounts to get subaccount information
    
    async def get_email_language(self) -> Dict:
        """Get email language setting"""
        return await self._make_jsonrpc_request("private/get_email_language")
    
    async def set_email_language(self, language: str) -> Dict:
        """Set email language"""
        return await self._make_jsonrpc_request("private/set_email_language", {"language": language})
    
    async def get_announcements(self, start_timestamp: Optional[str] = None) -> List[Dict]:
        """Get announcements"""
        params = {}
        if start_timestamp:
            params["start_timestamp"] = start_timestamp
        return await self._make_jsonrpc_request("public/get_announcements", params)
    
    # WebSocket methods
    async def connect_websocket(self):
        """Connect to Deribit WebSocket"""
        try:
            self.websocket = await websockets.connect(
                self.config.effective_ws_url,
                ping_interval=20,
                ping_timeout=10
            )
            logger.info("WebSocket connected")
            
            # Set up heartbeat
            await self._setup_heartbeat()
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise DeribitConnectionError(f"WebSocket connection failed: {e}")
    
    async def _setup_heartbeat(self):
        """Set up WebSocket heartbeat"""
        heartbeat_msg = {
            "jsonrpc": "2.0",
            "id": 9098,
            "method": "public/set_heartbeat",
            "params": {"interval": 10}
        }
        await self.websocket.send(json.dumps(heartbeat_msg))
    
    async def subscribe_to_ticker(self, instruments: List[str]) -> None:
        """Subscribe to ticker updates via WebSocket"""
        channels = [f"ticker.{instrument}.100ms" for instrument in instruments]
        await self._subscribe_to_channels(channels)
    
    async def subscribe_to_order_book(self, instruments: List[str]) -> None:
        """Subscribe to order book updates"""
        channels = [f"book.{instrument}.raw" for instrument in instruments]
        await self._subscribe_to_channels(channels)
    
    async def subscribe_to_mark_price(self) -> None:
        """Subscribe to mark price updates"""
        channels = ["markprice.options.btc_usd"]
        await self._subscribe_to_channels(channels)
    
    async def _subscribe_to_channels(self, channels: List[str]) -> None:
        """Subscribe to WebSocket channels"""
        if not self.websocket:
            await self.connect_websocket()
        
        message = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {"channels": channels},
            "id": 1
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to channels: {channels}")
    
    async def subscribe(self, channels: List[str], private: bool = False) -> None:
        """
        Subscribe to WebSocket channels (public or private)
        
        Args:
            channels: List of channel names (e.g., ["book.BTC_USDC-PERPETUAL.none.10.100ms"])
            private: If True, use private/subscribe (requires authentication)
        """
        if not self.websocket:
            await self.connect_websocket()
        
        method = "private/subscribe" if private else "public/subscribe"
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": {"channels": channels},
            "id": 1
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to {'private' if private else 'public'} channels: {channels}")
    
    def register_message_handler(self, channel_pattern: str, handler: Callable):
        """Register a message handler for specific channel pattern"""
        self._message_handlers[channel_pattern] = handler
    
    async def listen_for_messages(self):
        """Listen for WebSocket messages and route to handlers"""
        if not self.websocket:
            await self.connect_websocket()
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle heartbeat
                    if data.get('method') == 'heartbeat':
                        await self._handle_heartbeat()
                        continue
                    
                    # Route to appropriate handler
                    if 'params' in data and 'channel' in data['params']:
                        channel = data['params']['channel']
                        await self._route_message(channel, data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def _handle_heartbeat(self):
        """Handle heartbeat message"""
        response = {
            "jsonrpc": "2.0",
            "id": 8212,
            "method": "public/test",
            "params": {}
        }
        await self.websocket.send(json.dumps(response))
    
    async def _route_message(self, channel: str, data: Dict):
        """Route message to appropriate handler"""
        for pattern, handler in self._message_handlers.items():
            if re.match(pattern, channel):
                try:
                    await handler(channel, data)

                except Exception as e:
                    logger.error(f"Error in message handler for {pattern}: {e}")
    
    # Data parsing methods
    def _parse_option_contract(self, data: Dict) -> OptionContract:
        """Parse instrument data to OptionContract"""
        from ..models.market_data import InstrumentKind, OptionType
        
        return OptionContract(
            instrument_name=data['instrument_name'],
            base_currency=data['base_currency'],
            quote_currency=data['quote_currency'],
            kind=InstrumentKind(data['kind']),
            is_active=data['is_active'],
            min_trade_amount=data['min_trade_amount'],
            tick_size=data['tick_size'],
            contract_size=data['contract_size'],
            settlement_period=data['settlement_period'],
            expiration_timestamp=data['expiration_timestamp'],
            strike=data.get('strike'),
            option_type=OptionType(data['option_type']) if data.get('option_type') else None
        )
    
    def _parse_market_data(self, data: Dict) -> MarketData:
        """Parse ticker data to MarketData"""
        return MarketData(
            instrument_name=data['instrument_name'],
            timestamp=data['timestamp'],
            stats=data.get('stats', {}),
            state=data.get('state', ''),
            ticker=data.get('ticker', {}),
            mark_price=data.get('mark_price'),
            mark_iv=data.get('mark_iv'),
            best_bid_price=data.get('best_bid_price'),
            best_ask_price=data.get('best_ask_price'),
            best_bid_amount=data.get('best_bid_amount'),
            best_ask_amount=data.get('best_ask_amount'),
            greeks=data.get('greeks', {})
        )
    
    def _parse_order_book(self, data: Dict) -> OrderBook:
        """Parse order book data"""
        return OrderBook(
            instrument_name=data['instrument_name'],
            timestamp=data['timestamp'],
            bids=data.get('bids', []),
            asks=data.get('asks', [])
        )
