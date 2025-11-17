#!/usr/bin/env python3
"""
Step 3: Deribit Login with Session Management using fork_token
"""

import asyncio
import logging
from deribit_trading_toolkit import (
    DeribitClient, DeribitAuth, ConfigManager, SessionManager,
    DeribitError, DeribitConnectionError, DeribitAuthError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def session_managed_login():
    """Deribit login with session management"""
    session_manager = None
    
    try:
        # Step 1: Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        logger.info("✅ Configuration loaded")
        
        # Step 2: Create session manager
        logger.info("Creating session manager...")
        session_manager = SessionManager(
            config=config.deribit,
            base_client_id=config.deribit.client_id,
            private_key_path=config.deribit.private_key_path
        )
        
        # Step 3: Initialize base session
        logger.info("Initializing base session...")
        if not await session_manager.initialize_base_session():
            logger.error("❌ Failed to initialize base session")
            return False
        
        logger.info("✅ Base session initialized")
        
        # Step 4: Create additional sessions using fork_token
        logger.info("Creating additional sessions...")
        
        session_names = ["trading_session", "monitoring_session", "risk_session"]
        session_ids = []
        
        for session_name in session_names:
            session_id = await session_manager.create_session(session_name)
            if session_id:
                session_ids.append(session_id)
                logger.info(f"✅ Created session: {session_name} (ID: {session_id})")
            else:
                logger.error(f"❌ Failed to create session: {session_name}")
        
        # Step 5: Test each session
        logger.info("Testing sessions...")
        for session_id in session_ids:
            try:
                client = await session_manager.get_session(session_id)
                if client:
                    account_summary = await client.get_account_summary("BTC")
                    logger.info(f"✅ Session {session_id} working: {account_summary.get('equity', 'N/A')} BTC")
                else:
                    logger.error(f"❌ Session {session_id} not available")
            except Exception as e:
                logger.error(f"❌ Session {session_id} error: {e}")
        
        # Step 6: Get session statistics
        stats = session_manager.get_session_statistics()
        logger.info(f"✅ Session statistics: {stats}")
        
        # Step 7: Health check
        health_results = await session_manager.health_check_sessions()
        logger.info(f"✅ Health check results: {health_results}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Session management error: {e}")
        return False
        
    finally:
        # Clean up
        if session_manager:
            try:
                await session_manager.close_all_sessions()
                logger.info("All sessions closed")
            except Exception as e:
                logger.error(f"Error closing sessions: {e}")


async def main():
    """Main function"""
    print("Deribit Login with Session Management")
    print("=" * 40)
    
    success = await session_managed_login()
    
    if success:
        print("\n🎉 Session management test completed successfully!")
        print("You now have multiple authenticated sessions ready for trading.")
    else:
        print("\n💥 Session management test failed!")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())