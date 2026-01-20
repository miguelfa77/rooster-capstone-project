# proxies/proxies.py
"""
Proxy configuration and management for Idealista scraper.
Supports rotating proxy endpoints that return different IPs on each request.
"""

import requests
from typing import Dict, Optional, Tuple

# Rotating proxy endpoint configuration
# Each request to this endpoint returns a different IP address
# Proxy endpoint: p.webshare.io:80
# Test URL to check IP: https://ipv4.webshare.io/
PROXY_ENDPOINT_CONFIG = {
    'name': 'p.webshare.io',  # Proxy endpoint name/identifier
    'username': 'yxbayudf-rotate',
    'password': 'lm6w1875zt1i',
    'address': 'p.webshare.io',  # Endpoint address (stays constant)
    'port': '80'  # Endpoint port (stays constant)
}


class RotatingProxyManager:
    """
    Manages rotating proxy endpoint.
    The endpoint address/port stays the same, but each request gets a new IP.
    """
    
    def __init__(self, config: Dict[str, str] = None):
        """
        Initialize proxy manager with endpoint configuration.
        
        Args:
            config: Proxy endpoint config dict (default: PROXY_ENDPOINT_CONFIG)
        """
        self.config = config or PROXY_ENDPOINT_CONFIG
        self.current_ip = None
    
    def get_proxy_string(self) -> str:
        """
        Get proxy string for Selenium/Chrome.
        Uses the endpoint address/port (IP rotates automatically).
        
        Returns:
            Proxy string in format: http://user:pass@address:port
        """
        return (
            f"http://{self.config['username']}:{self.config['password']}"
            f"@{self.config['address']}:{self.config['port']}"
        )
    
    def get_proxy_for_requests(self) -> Dict[str, str]:
        """
        Get proxy dict for requests library.
        
        Returns:
            Proxy dict with http and https keys
        """
        proxy_url = self.get_proxy_string()
        return {
            "http": proxy_url,
            "https": proxy_url
        }
    
    def get_current_ip(self) -> Optional[str]:
        """
        Get the current IP address being used by the proxy.
        Makes a test request to determine the actual IP.
        
        Returns:
            IP address string or None if request fails
        """
        try:
            response = requests.get(
                "https://ipv4.webshare.io/",
                proxies=self.get_proxy_for_requests(),
                timeout=10
            )
            if response.status_code == 200:
                self.current_ip = response.text.strip()
                return self.current_ip
        except Exception as e:
            print(f"Error getting current IP: {e}")
        return None
    
    def test_proxy(self) -> bool:
        """
        Test if the proxy endpoint is working.
        
        Returns:
            True if proxy works, False otherwise
        """
        try:
            response = requests.get(
                "https://ipv4.webshare.io/",
                proxies=self.get_proxy_for_requests(),
                timeout=10
            )
            if response.status_code == 200:
                self.current_ip = response.text.strip()
                return True
            return False
        except Exception as e:
            print(f"Proxy test failed: {e}")
            return False
    
    def get_proxy_info(self) -> Dict[str, str]:
        """
        Get proxy information including current IP.
        
        Returns:
            Dict with proxy endpoint info and current IP
        """
        info = {
            'endpoint': f"{self.config['address']}:{self.config['port']}",
            'name': self.config.get('name', 'unknown'),
            'current_ip': self.current_ip or 'unknown'
        }
        return info


# Convenience functions for backward compatibility
def get_proxy_string(config: Dict[str, str] = None) -> str:
    """
    Get proxy string from config.
    
    Args:
        config: Proxy config dict (default: PROXY_ENDPOINT_CONFIG)
    
    Returns:
        Proxy string for Selenium
    """
    manager = RotatingProxyManager(config)
    return manager.get_proxy_string()


def get_proxy_for_requests(config: Dict[str, str] = None) -> Dict[str, str]:
    """
    Get proxy dict for requests library.
    
    Args:
        config: Proxy config dict (default: PROXY_ENDPOINT_CONFIG)
    
    Returns:
        Proxy dict for requests
    """
    manager = RotatingProxyManager(config)
    return manager.get_proxy_for_requests()


def test_proxy(config: Dict[str, str] = None) -> bool:
    """
    Test if proxy endpoint is working.
    
    Args:
        config: Proxy config dict (default: PROXY_ENDPOINT_CONFIG)
    
    Returns:
        True if proxy works, False otherwise
    """
    manager = RotatingProxyManager(config)
    return manager.test_proxy()


def get_config(config: Dict[str, str] = None) -> Optional[str]:
    """
    Legacy function for backward compatibility.
    Tests proxy and returns IP if successful.
    
    Args:
        config: Proxy config dict (default: PROXY_ENDPOINT_CONFIG)
    
    Returns:
        Current IP address or None
    """
    manager = RotatingProxyManager(config)
    return manager.get_current_ip()


def main():
    """Test proxy configuration"""
    manager = RotatingProxyManager()
    
    print("Testing rotating proxy endpoint...")
    print(f"Endpoint: {manager.config['address']}:{manager.config['port']}")
    
    if manager.test_proxy():
        print(f"✓ Proxy is working")
        print(f"Current IP: {manager.current_ip}")
        
        # Store the first IP before getting a new one
        first_ip = manager.current_ip
        
        # Test rotation - get IP again (should be different)
        print("\nTesting rotation (getting IP again)...")
        new_ip = manager.get_current_ip()
        print(f"New IP: {new_ip}")
        
        # Compare with the stored first IP, not the updated current_ip
        if new_ip and new_ip != first_ip:
            print("✓ IP rotation confirmed!")
        else:
            print("Note: IP may be the same or rotation happens on connection")
    else:
        print("✗ Proxy test failed")


if __name__ == "__main__":
    main()