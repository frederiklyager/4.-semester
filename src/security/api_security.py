"""
API Authentication & Token Management
Simulates secure API key handling for Energy Forecast system

Even though Energinet API is public, this demonstrates production security practices:
- Secure credential storage
- Token rotation
- Access logging
- Rate limiting
"""

import os
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import logging

# Security logging
logging.basicConfig(
    filename='logs/security.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APITokenManager:
    """
    Manages API tokens with rotation and validation
    
    Security Features:
    - Token generation with cryptographic randomness
    - SHA-256 hashing for stored tokens
    - Automatic expiration after 30 days
    - Access logging
    - Rate limiting
    """
    
    def __init__(self, token_file: str = ".tokens/api_tokens.json"):
        self.token_file = Path(token_file)
        self.token_file.parent.mkdir(exist_ok=True)
        self._load_tokens()
    
    def _load_tokens(self):
        """Load existing tokens from secure storage"""
        if self.token_file.exists():
            with open(self.token_file, 'r') as f:
                self.tokens = json.load(f)
        else:
            self.tokens = {}
    
    def _save_tokens(self):
        """Save tokens to secure storage"""
        with open(self.token_file, 'w') as f:
            json.dump(self.tokens, f, indent=2)
        # Set restrictive permissions (owner only)
        os.chmod(self.token_file, 0o600)
    
    def generate_token(self, service: str = "energinet_api") -> str:
        """
        Generate cryptographically secure API token
        
        Args:
            service: Service name for this token
        
        Returns:
            Secure random token (32 bytes, hex encoded)
        """
        # Generate 32-byte random token
        token = secrets.token_hex(32)
        
        # Hash token for storage (SHA-256)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Store with metadata
        self.tokens[service] = {
            "token_hash": token_hash,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
            "last_used": None,
            "usage_count": 0
        }
        
        self._save_tokens()
        logger.info(f"Generated new API token for service: {service}")
        
        return token
    
    def validate_token(self, token: str, service: str = "energinet_api") -> bool:
        """
        Validate API token
        
        Args:
            token: Token to validate
            service: Service name
        
        Returns:
            True if valid, False otherwise
        """
        if service not in self.tokens:
            logger.warning(f"Token validation failed: Unknown service {service}")
            return False
        
        # Hash provided token
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Compare with stored hash
        stored = self.tokens[service]
        
        if stored["token_hash"] != token_hash:
            logger.warning(f"Token validation failed: Invalid token for {service}")
            return False
        
        # Check expiration
        expires_at = datetime.fromisoformat(stored["expires_at"])
        if datetime.now() > expires_at:
            logger.warning(f"Token validation failed: Expired token for {service}")
            return False
        
        # Update usage metadata
        self.tokens[service]["last_used"] = datetime.now().isoformat()
        self.tokens[service]["usage_count"] += 1
        self._save_tokens()
        
        logger.info(f"Token validated successfully for {service}")
        return True
    
    def rotate_token(self, service: str = "energinet_api") -> str:
        """
        Rotate API token (security best practice)
        
        Args:
            service: Service name
        
        Returns:
            New token
        """
        logger.info(f"Rotating token for service: {service}")
        return self.generate_token(service)
    
    def get_token_info(self, service: str = "energinet_api") -> Optional[Dict]:
        """Get token metadata without exposing actual token"""
        return self.tokens.get(service)


class RateLimiter:
    """
    Rate limiting for API calls
    
    Prevents abuse and demonstrates security awareness
    """
    
    def __init__(self, max_calls: int = 100, window_seconds: int = 3600):
        """
        Args:
            max_calls: Maximum calls allowed
            window_seconds: Time window in seconds (default 1 hour)
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = []
    
    def check_rate_limit(self, identifier: str) -> tuple[bool, Optional[str]]:
        """
        Check if rate limit is exceeded
        
        Args:
            identifier: Unique identifier (e.g., API endpoint)
        
        Returns:
            (allowed: bool, message: str)
        """
        now = datetime.now()
        
        # Remove old calls outside window
        self.calls = [
            (ts, id_) for ts, id_ in self.calls
            if (now - ts).total_seconds() < self.window_seconds
        ]
        
        # Count calls for this identifier
        count = sum(1 for _, id_ in self.calls if id_ == identifier)
        
        if count >= self.max_calls:
            reset_time = min(ts for ts, id_ in self.calls if id_ == identifier) + timedelta(seconds=self.window_seconds)
            remaining = (reset_time - now).total_seconds()
            
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False, f"Rate limit exceeded. Try again in {remaining:.0f} seconds"
        
        # Add this call
        self.calls.append((now, identifier))
        logger.info(f"Rate limit check passed: {count + 1}/{self.max_calls} for {identifier}")
        
        return True, None


# Global instances
token_manager = APITokenManager()
rate_limiter = RateLimiter(max_calls=100, window_seconds=3600)


def secure_api_call(endpoint: str, token: str, **kwargs):
    """
    Wrapper for secure API calls with authentication and rate limiting
    
    Args:
        endpoint: API endpoint name
        token: API token
        **kwargs: Additional arguments for requests
    
    Returns:
        Response or raises exception
    """
    import requests
    
    # Validate token
    if not token_manager.validate_token(token):
        logger.error(f"API call rejected: Invalid token for {endpoint}")
        raise PermissionError("Invalid API token")
    
    # Check rate limit
    allowed, message = rate_limiter.check_rate_limit(endpoint)
    if not allowed:
        logger.warning(f"API call rejected: Rate limit for {endpoint}")
        raise Exception(f"Rate limit exceeded: {message}")
    
    # Make authenticated API call
    logger.info(f"Secure API call to {endpoint}")
    return requests.get(endpoint, **kwargs)


if __name__ == "__main__":
    # Demo: Generate and validate token
    print("🔐 API Token Security Demo\n")
    
    # Generate token
    token = token_manager.generate_token("energinet_api")
    print(f"✅ Generated token: {token[:16]}... (truncated for security)")
    
    # Validate token
    is_valid = token_manager.validate_token(token)
    print(f"✅ Token validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Show token info
    info = token_manager.get_token_info("energinet_api")
    print(f"\n📊 Token Info:")
    print(f"   Created: {info['created_at']}")
    print(f"   Expires: {info['expires_at']}")
    print(f"   Usage: {info['usage_count']} calls")
    
    # Test rate limiting
    print(f"\n⏱️  Rate Limiting Test:")
    for i in range(3):
        allowed, msg = rate_limiter.check_rate_limit("test_endpoint")
        print(f"   Call {i+1}: {'✅ ALLOWED' if allowed else f'❌ BLOCKED - {msg}'}")