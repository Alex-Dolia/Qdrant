"""
Qdrant Connection Health Check Utility

Provides functions to check Qdrant connection status and health.
Can be used to verify connectivity before operations and display status in UI.
"""

import logging
import os
import time
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import Qdrant client
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


def check_qdrant_connection(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    timeout: float = 5.0
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Check if Qdrant is accessible and responding.
    
    Args:
        qdrant_url: Qdrant server URL (defaults to QDRANT_URL env var or http://localhost:6333)
        qdrant_api_key: Optional API key for Qdrant
        timeout: Connection timeout in seconds
    
    Returns:
        Tuple of (is_connected: bool, error_message: Optional[str], info: Optional[Dict])
        - is_connected: True if Qdrant is accessible
        - error_message: Error description if connection failed
        - info: Dictionary with connection info (url, version, collections count, etc.)
    """
    if not QDRANT_AVAILABLE:
        return False, "Qdrant client library not installed. Install with: pip install qdrant-client", None
    
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    if qdrant_api_key is None:
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    
    try:
        # Create client with timeout
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=timeout
        )
        
        # Try to get collections (lightweight operation)
        start_time = time.time()
        collections = client.get_collections()
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Get additional info if possible
        info = {
            "url": qdrant_url,
            "response_time_ms": round(response_time, 2),
            "collections_count": len(collections.collections),
            "collections": [col.name for col in collections.collections],
            "status": "connected"
        }
        
        # Try to get version info (may not be available in all Qdrant versions)
        try:
            # This is a lightweight operation that just checks connectivity
            # We already have collections, so connection is good
            info["version"] = "unknown"  # Qdrant Python client doesn't expose version easily
        except Exception:
            pass
        
        logger.info(f"Qdrant connection check successful: {qdrant_url} (response time: {response_time:.2f}ms)")
        return True, None, info
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Qdrant connection check failed: {error_msg}")
        
        # Provide helpful error messages
        if "Connection refused" in error_msg or "refused" in error_msg.lower():
            return False, f"Qdrant server not accessible at {qdrant_url}. Please ensure Qdrant is running.", None
        elif "timeout" in error_msg.lower():
            return False, f"Connection to Qdrant at {qdrant_url} timed out after {timeout}s.", None
        elif "Name or service not known" in error_msg or "getaddrinfo failed" in error_msg:
            return False, f"Could not resolve Qdrant hostname. Check QDRANT_URL environment variable.", None
        else:
            return False, f"Failed to connect to Qdrant: {error_msg}", None


def get_qdrant_status(qdrant_client: Optional[QdrantClient] = None) -> Dict[str, Any]:
    """
    Get current Qdrant connection status.
    
    Args:
        qdrant_client: Optional Qdrant client instance (will create new if None)
    
    Returns:
        Dictionary with status information
    """
    if qdrant_client is None:
        is_connected, error_msg, info = check_qdrant_connection()
    else:
        try:
            collections = qdrant_client.get_collections()
            info = {
                "url": "connected",
                "collections_count": len(collections.collections),
                "collections": [col.name for col in collections.collections],
                "status": "connected"
            }
            is_connected, error_msg = True, None
        except Exception as e:
            is_connected, error_msg, info = False, str(e), None
    
    return {
        "connected": is_connected,
        "error": error_msg,
        "info": info
    }


@lru_cache(maxsize=1)
def get_cached_qdrant_status(ttl: float = 30.0) -> Dict[str, Any]:
    """
    Get cached Qdrant status (refreshes every TTL seconds).
    
    Args:
        ttl: Time-to-live for cache in seconds
    
    Returns:
        Dictionary with status information
    """
    # Note: lru_cache doesn't support TTL natively, so this is a simple cache
    # For production, consider using a proper TTL cache library
    return check_qdrant_connection()


def verify_qdrant_before_operation(
    qdrant_client: Optional[QdrantClient] = None,
    operation_name: str = "operation"
) -> bool:
    """
    Verify Qdrant connection before performing an operation.
    Raises exception if not connected.
    
    Args:
        qdrant_client: Qdrant client instance
        operation_name: Name of operation for error message
    
    Returns:
        True if connected
    
    Raises:
        ConnectionError: If Qdrant is not accessible
    """
    if qdrant_client is None:
        is_connected, error_msg, _ = check_qdrant_connection()
        if not is_connected:
            raise ConnectionError(f"Cannot perform {operation_name}: {error_msg}")
        return True
    
    try:
        # Quick health check
        qdrant_client.get_collections()
        return True
    except Exception as e:
        error_msg = str(e)
        raise ConnectionError(
            f"Cannot perform {operation_name}: Qdrant connection failed - {error_msg}. "
            f"Please ensure Qdrant is running at the configured URL."
        )


def display_qdrant_status_in_ui(streamlit_module=None):
    """
    Display Qdrant connection status in Streamlit UI.
    
    Args:
        streamlit_module: Streamlit module (usually `st`)
    """
    if streamlit_module is None:
        try:
            import streamlit as st
            streamlit_module = st
        except ImportError:
            logger.warning("Streamlit not available, cannot display status")
            return
    
    is_connected, error_msg, info = check_qdrant_connection()
    
    if is_connected:
        with streamlit_module.container():
            streamlit_module.success(f"✅ Qdrant Connected ({info.get('url', 'unknown')})")
            if info:
                with streamlit_module.expander("Qdrant Status Details"):
                    streamlit_module.write(f"**URL:** {info.get('url', 'unknown')}")
                    streamlit_module.write(f"**Response Time:** {info.get('response_time_ms', 'unknown')}ms")
                    streamlit_module.write(f"**Collections:** {info.get('collections_count', 0)}")
                    if info.get('collections'):
                        streamlit_module.write(f"**Collection Names:** {', '.join(info['collections'][:5])}")
                        if len(info['collections']) > 5:
                            streamlit_module.write(f"... and {len(info['collections']) - 5} more")
    else:
        streamlit_module.error(f"❌ Qdrant Not Connected: {error_msg}")
        streamlit_module.info(
            "💡 **To start Qdrant:**\n"
            "1. Install Docker Desktop\n"
            "2. Run: `docker run -p 6333:6333 qdrant/qdrant`\n"
            "3. Or use the provided `start_qdrant.bat` script"
        )

