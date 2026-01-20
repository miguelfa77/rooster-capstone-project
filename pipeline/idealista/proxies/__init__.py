# proxies/__init__.py
"""
Proxy management for Idealista scraper.
"""

from proxies.proxies import (
    RotatingProxyManager,
    PROXY_ENDPOINT_CONFIG,
    get_proxy_string,
    get_proxy_for_requests,
    test_proxy,
    get_config
)

__all__ = [
    'RotatingProxyManager',
    'PROXY_ENDPOINT_CONFIG',
    'get_proxy_string',
    'get_proxy_for_requests',
    'test_proxy',
    'get_config'
]

