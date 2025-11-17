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
    
    def __init__(self, client_id: str, private_key_path: str, session_name: Optional[str] = None):
        self.client_id = client_id
        self.private_key_path = private_key_path
        self.session_name = session_name or "default_session"
        self.private_key = None
        self.access_token = None
        self.refresh_token = None
        self.refresh_token_expiry = None
        self.token_type = "bearer"
        self.scope = None
        self._load_private_key()
    
    def _load_private_key(self):
        """Load RSA private key from file"""
        try:
            from cryptography.hazmat.primitives import serialization
            with open(self.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.info("Private key loaded successfully")
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
            
            # Sign the data
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
    
    async def authenticate(self, session: aiohttp.ClientSession) -> bool:
        """Authenticate with Deribit API"""
        try:
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
            
            async with session.post(
                "https://www.deribit.com/api/v2",
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
    
    async def refresh_auth(self, session: aiohttp.ClientSession) -> bool:
        """Refresh authentication token"""
        try:
            if not self.refresh_token:
                return await self.authenticate(session)
            
            payload = {
                "jsonrpc": "2.0",
                "id": 9929,
                "method": "public/auth",
                "params": {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token
                }
            }
            
            async with session.post(
                "https://www.deribit.com/api/v2",
                json=payload
            ) as response:
                result = await response.json()
                
                if 'error' in result:
                    logger.warning("Token refresh failed, re-authenticating")
                    return await self.authenticate(session)
                
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
        await self.auth.authenticate(self.session)
        logger.info("Deribit client connected successfully")
    
    async def disconnect(self):
        """Clean up connections"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
        logger.info("Deribit client disconnected")
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None,
                          data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated HTTP request with rate limiting"""
        async with self._rate_limiter:
            # Check if we need to refresh auth
            if self.auth.needs_refresh():
                await self.auth.refresh_auth(self.session)
            
            url = f"{self.config.effective_api_url}/{endpoint}"
            headers = self.auth.get_headers()
            
            try:
                async with self.session.request(
                    method, url, params=params, json=data, headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if 'error' in result:
                        raise DeribitError(result['error'].get('message', 'Unknown error'), 
                                         result['error'].get('code'))
                    
                    return result.get('result', {})
                    
            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                raise DeribitConnectionError(f"Request failed: {e}")
    
    # Market Data Methods
    async def get_instruments(self, currency: str = "BTC", 
                           kind: str = "option") -> List[Dict]:
        """Get available instruments"""
        return await self._make_request(
            "GET", "public/get_instruments",
            params={"currency": currency, "kind": kind}
        )
    
    async def get_option_chain(self, currency: str = "BTC", 
                             expiration_date: Optional[str] = None) -> List[OptionContract]:
        """Get option chain for given currency and expiration"""
        params = {"currency": currency, "kind": "option"}
        if expiration_date:
            params["expiration_date"] = expiration_date
            
        instruments = await self._make_request(
            "GET", "public/get_instruments", params=params
        )
        
        return [self._parse_option_contract(instr) for instr in instruments]
    
    async def get_ticker(self, instrument_name: str) -> MarketData:
        """Get current ticker data for instrument"""
        data = await self._make_request(
            "GET", "public/ticker",
            params={"instrument_name": instrument_name}
        )
        return self._parse_market_data(data)
    
    async def get_order_book(self, instrument_name: str, depth: int = 20) -> OrderBook:
        """Get order book for instrument"""
        data = await self._make_request(
            "GET", "public/get_order_book",
            params={"instrument_name": instrument_name, "depth": depth}
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
            
        return await self._make_request(
            "GET", "public/get_historical_volatility", params=params
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
        payload = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "public/get_tradingview_chart_data",
            "params": {
                "instrument_name": instrument_name,
                "start_timestamp": str(start_timestamp),  # Convert to string
                "end_timestamp": str(end_timestamp),  # Convert to string
                "resolution": resolution
            }
        }
        
        # Use POST for JSON-RPC
        async with self._rate_limiter:
            if self.auth.needs_refresh():
                await self.auth.refresh_auth(self.session)
            
            # For JSON-RPC, use the base API URL, not the method-specific endpoint
            url = f"{self.config.effective_api_url}"
            headers = self.auth.get_headers()
            
            try:
                async with self.session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if 'error' in result:
                        error_info = result['error']
                        error_msg = error_info.get('message', 'Unknown error')
                        error_code = error_info.get('code')
                        logger.error(f"Deribit API error: {error_code} - {error_msg}")
                        raise DeribitError(error_msg, error_code)
                    
                    return result.get('result', {})
                    
            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch TradingView chart data: {e}")
                raise DeribitConnectionError(f"Request failed: {e}")
    
    # Options-specific methods
    async def get_option_greeks(self, instrument_name: str) -> Dict:
        """Get Greeks for specific option"""
        return await self._make_request(
            "GET", "public/get_option_greeks",
            params={"instrument_name": instrument_name}
        )
    
    async def get_implied_volatility(self, instrument_name: str) -> float:
        """Get implied volatility for option"""
        ticker = await self.get_ticker(instrument_name)
        return ticker.mark_iv or 0.0
    
    # Portfolio methods (require authentication)
    async def get_positions(self, currency: str = "BTC") -> List[Dict]:
        """Get current positions"""
        return await self._make_request(
            "GET", "private/get_positions",
            params={"currency": currency}
        )
    
    async def get_account_summary(self, currency: str = "BTC") -> Dict:
        """Get account summary"""
        return await self._make_request(
            "GET", "private/get_account_summary",
            params={"currency": currency}
        )
    
    async def place_order(self, instrument_name: str, amount: float,
                        order_type: OrderType = OrderType.MARKET, 
                        price: Optional[float] = None,
                        side: OrderSide = OrderSide.BUY) -> Dict:
        """Place order"""
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": order_type.value,
            "side": side.value
        }
        if price:
            params["price"] = price
            
        return await self._make_request(
            "POST", f"private/{side.value}", data=params
        )
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel order"""
        return await self._make_request(
            "POST", "private/cancel",
            data={"order_id": order_id}
        )
    
    async def simulate_portfolio(self, positions: Dict[str, float], 
                               currency: str = "BTC") -> Dict:
        """Simulate portfolio with given positions"""
        return await self._make_request(
            "POST", "private/simulate_portfolio",
            data={
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
