"""
Session Management for Deribit Trading Toolkit

Handles multiple trading sessions using Deribit's fork_token functionality.
Allows for concurrent trading strategies with separate authentication sessions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

from .client import DeribitClient, DeribitAuth
from ..utils.config import DeribitConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a trading session"""
    session_id: str
    session_name: str
    client: DeribitClient
    auth: DeribitAuth
    created_at: datetime
    last_activity: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session info to dictionary"""
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_active': self.is_active,
            'auth_info': self.auth.get_session_info()
        }


class SessionManager:
    """
    Manages multiple Deribit trading sessions
    
    Features:
    - Create multiple named sessions
    - Session isolation and management
    - Automatic session cleanup
    - Session monitoring and health checks
    """
    
    def __init__(self, config: DeribitConfig, base_client_id: str, 
                 private_key_path: Optional[str] = None, private_key: Optional[str] = None):
        self.config = config
        self.base_client_id = base_client_id
        self.private_key_path = private_key_path
        self.private_key = private_key
        self.sessions: Dict[str, SessionInfo] = {}
        self.base_session: Optional[DeribitClient] = None
        self.base_auth: Optional[DeribitAuth] = None
        
    async def initialize_base_session(self) -> bool:
        """
        Initialize the base session for token forking
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create base authentication
            self.base_auth = DeribitAuth(
                client_id=self.base_client_id,
                private_key_path=self.private_key_path,
                private_key=self.private_key,
                session_name="base_session"
            )
            
            # Create base client
            self.base_session = DeribitClient(self.config, self.base_auth)
            await self.base_session.connect()
            
            logger.info("Base session initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize base session: {e}")
            return False
    
    async def create_session(self, session_name: str) -> Optional[str]:
        """
        Create a new trading session
        
        Args:
            session_name: Name for the new session
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if not self.base_session:
                if not await self.initialize_base_session():
                    return None
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Fork token for new session
            fork_result = await self.base_session.fork_token(session_name)
            
            # Create new authentication for the session
            session_auth = DeribitAuth(
                client_id=self.base_client_id,
                private_key_path=self.private_key_path,
                private_key=self.private_key,
                session_name=session_name
            )
            
            # Update auth with forked token info
            session_auth.access_token = fork_result['access_token']
            session_auth.refresh_token = fork_result['refresh_token']
            session_auth.token_type = fork_result['token_type']
            session_auth.scope = fork_result['scope']
            
            # Create new client for the session
            session_client = DeribitClient(self.config, session_auth)
            await session_client.connect()
            
            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                session_name=session_name,
                client=session_client,
                auth=session_auth,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Store session
            self.sessions[session_id] = session_info
            
            logger.info(f"Created new session: {session_name} (ID: {session_id})")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session {session_name}: {e}")
            return None
    
    async def get_session(self, session_id: str) -> Optional[DeribitClient]:
        """
        Get client for a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            DeribitClient instance or None if not found
        """
        session_info = self.sessions.get(session_id)
        if not session_info or not session_info.is_active:
            return None
        
        # Update last activity
        session_info.last_activity = datetime.now()
        return session_info.client
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close a specific session
        
        Args:
            session_id: Session ID to close
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_info = self.sessions.get(session_id)
            if not session_info:
                return False
            
            # Disconnect client
            await session_info.client.disconnect()
            
            # Mark as inactive
            session_info.is_active = False
            
            logger.info(f"Closed session: {session_info.session_name} (ID: {session_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False
    
    async def close_all_sessions(self):
        """Close all active sessions"""
        try:
            for session_id in list(self.sessions.keys()):
                await self.close_session(session_id)
            
            # Close base session
            if self.base_session:
                await self.base_session.disconnect()
            
            logger.info("All sessions closed")
            
        except Exception as e:
            logger.error(f"Error closing all sessions: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information dictionary or None if not found
        """
        session_info = self.sessions.get(session_id)
        if not session_info:
            return None
        
        return session_info.to_dict()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions
        
        Returns:
            List of session information dictionaries
        """
        return [session_info.to_dict() for session_info in self.sessions.values()]
    
    async def refresh_session_token(self, session_id: str) -> bool:
        """
        Refresh token for a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_info = self.sessions.get(session_id)
            if not session_info or not session_info.is_active:
                return False
            
            # Use the session's auth to refresh
            await session_info.auth.refresh_auth(session_info.client.session)
            
            logger.info(f"Refreshed token for session: {session_info.session_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing token for session {session_id}: {e}")
            return False
    
    async def health_check_sessions(self) -> Dict[str, Any]:
        """
        Perform health check on all sessions
        
        Returns:
            Dictionary with health check results
        """
        try:
            health_results = {}
            
            for session_id, session_info in self.sessions.items():
                if not session_info.is_active:
                    health_results[session_id] = {
                        'status': 'inactive',
                        'session_name': session_info.session_name
                    }
                    continue
                
                try:
                    # Test session by making a simple API call
                    await session_info.client.get_account_summary("BTC")
                    
                    health_results[session_id] = {
                        'status': 'healthy',
                        'session_name': session_info.session_name,
                        'last_activity': session_info.last_activity.isoformat()
                    }
                    
                except Exception as e:
                    health_results[session_id] = {
                        'status': 'unhealthy',
                        'session_name': session_info.session_name,
                        'error': str(e)
                    }
            
            return health_results
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {}
    
    async def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """
        Clean up sessions that have been inactive for too long
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            sessions_to_close = []
            
            for session_id, session_info in self.sessions.items():
                if session_info.last_activity < cutoff_time and session_info.is_active:
                    sessions_to_close.append(session_id)
            
            for session_id in sessions_to_close:
                await self.close_session(session_id)
                logger.info(f"Cleaned up inactive session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all sessions
        
        Returns:
            Dictionary with session statistics
        """
        try:
            total_sessions = len(self.sessions)
            active_sessions = len([s for s in self.sessions.values() if s.is_active])
            inactive_sessions = total_sessions - active_sessions
            
            # Calculate average session age
            if self.sessions:
                now = datetime.now()
                total_age = sum(
                    (now - session_info.created_at).total_seconds()
                    for session_info in self.sessions.values()
                )
                avg_age_hours = total_age / len(self.sessions) / 3600
            else:
                avg_age_hours = 0
            
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'inactive_sessions': inactive_sessions,
                'average_session_age_hours': avg_age_hours,
                'base_session_active': self.base_session is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}


class MultiSessionTradingApp:
    """
    Trading application that supports multiple concurrent sessions
    
    This allows running multiple trading strategies simultaneously
    with separate authentication sessions for better isolation.
    """
    
    def __init__(self, config: DeribitConfig, base_client_id: str, 
                 private_key_path: Optional[str] = None, private_key: Optional[str] = None):
        self.config = config
        self.session_manager = SessionManager(config, base_client_id, private_key_path, private_key)
        self.strategies: Dict[str, Any] = {}  # session_id -> strategy
        
    async def initialize(self) -> bool:
        """Initialize the multi-session trading app"""
        try:
            return await self.session_manager.initialize_base_session()
        except Exception as e:
            logger.error(f"Failed to initialize multi-session app: {e}")
            return False
    
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
            # Create session
            session_id = await self.session_manager.create_session(session_name)
            if not session_id:
                return None
            
            # Get session client
            client = await self.session_manager.get_session(session_id)
            if not client:
                return None
            
            # Create strategy instance
            from ..risk.manager import RiskManager, RiskLimits
            risk_manager = RiskManager(client, RiskLimits())
            
            strategy = strategy_class(strategy_config, client, risk_manager)
            self.strategies[session_id] = strategy
            
            logger.info(f"Created strategy session: {session_name} (ID: {session_id})")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create strategy session {session_name}: {e}")
            return None
    
    async def run_strategy(self, session_id: str) -> Dict[str, Any]:
        """
        Run a strategy in a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            Strategy execution result
        """
        try:
            strategy = self.strategies.get(session_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}
            
            return await strategy.run_strategy()
            
        except Exception as e:
            logger.error(f"Error running strategy in session {session_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def shutdown(self):
        """Shutdown all sessions and strategies"""
        try:
            await self.session_manager.close_all_sessions()
            self.strategies.clear()
            logger.info("Multi-session trading app shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
