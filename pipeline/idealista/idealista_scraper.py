# idealista_scraper.py
"""
Idealista.com scraper for Valencia real estate listings.
Scrapes both 'venta' (sale) and 'alquiler' (rental) listings.

Features:
- Rotating proxy endpoint support
- Incremental CSV saves (pipe-separated)
- Robust error handling with retries
- Progress tracking and logging
- Resume capability via checkpoint files
"""

import time
import random
import os
import sys
import json
import tempfile
from typing import Dict, FrozenSet, List, Optional
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from seleniumwire import webdriver as wire_webdriver  # For proxy authentication
from fake_useragent import UserAgent

# Repo root on sys.path when run as `python pipeline/idealista/idealista_scraper.py`
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from pipeline.idealista import config
from pipeline.idealista.proxies import RotatingProxyManager, PROXY_ENDPOINT_CONFIG
from pipeline.idealista.utils import Logger, PropertyExtractor, DataManager, run_session


class IdealistaScraper:
    """Main scraper class"""
    
    def __init__(self):
        """Initialize scraper with rotating proxy manager"""
        self.ua = UserAgent()
        self.driver = None
        self.proxy_manager = RotatingProxyManager(PROXY_ENDPOINT_CONFIG)
        self.pages_scraped_with_current_proxy = 0
        self.extension_dir = None  # Store extension directory for cleanup
        self.user_data_dir = None  # Store user data directory for cleanup
    
    def _get_random_delay(self) -> float:
        """Get random delay between requests"""
        return random.uniform(
            config.DELAY_MIN_SECONDS,
            config.DELAY_MAX_SECONDS
        )
    
    def _setup_driver(self, use_proxy: bool = True) -> bool:
        """
        Setup Chrome driver with proxy and anti-detection measures.
        
        Args:
            use_proxy: Whether to use proxy (default: True)
        
        Returns:
            True if successful, False otherwise
        """
        if self.driver:
            self.driver.quit()
        
        options = Options()
        
        # Enhanced anti-detection options (similar to undetected-chromedriver)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument(f'user-agent={self.ua.random}')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-infobars')
        options.add_argument('--lang=es-ES,es')
        # Don't disable extensions if we need to load proxy auth extension
        # options.add_argument('--disable-extensions')  # Commented out - needed for proxy extension
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        
        # Additional stealth options
        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-features=TranslateUI')
        options.add_argument('--disable-ipc-flooding-protection')
        
        # Create isolated user data directory for scraper
        # This ensures the scraper's Chrome instance is completely separate from your personal Chrome profiles
        self.user_data_dir = tempfile.mkdtemp(prefix='chrome_scraper_')
        options.add_argument(f'--user-data-dir={self.user_data_dir}')
        Logger.info(f"Using isolated Chrome profile: {self.user_data_dir}")
        
        # Set preferences to avoid detection and prevent IP leaks
        prefs = {
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_settings.popups': 0,
            'profile.managed_default_content_settings.images': 1,
            'credentials_enable_service': False,
            'profile.password_manager_enabled': False,
            # Disable WebRTC to prevent IP leaks
            'webrtc.ip_handling_policy': 'disable_non_proxied_udp',
            'webrtc.multiple_routes_enabled': False,
            'webrtc.nonproxied_udp_enabled': False,
        }
        options.add_experimental_option('prefs', prefs)
        
        # Additional flags to prevent IP leaks
        options.add_argument('--disable-webrtc')  # Disable WebRTC completely
        options.add_argument('--disable-webrtc-hw-encoding')
        options.add_argument('--disable-webrtc-hw-decoding')
        options.add_argument('--force-webrtc-ip-permission-check')  # Force WebRTC through proxy
        
        # Headless mode (optional - visible browser is less detectable)
        if config.HEADLESS_MODE:
            options.add_argument('--headless=new')  # New headless mode
        
        # Proxy configuration
        # Store proxy config for use with selenium-wire (which supports authenticated proxies)
        proxy_config = None
        if use_proxy and config.USE_PROXY:
            proxy_config = self.proxy_manager.config
            Logger.info(f"Using proxy endpoint: {proxy_config['address']}:{proxy_config['port']}")
            Logger.info(f"Proxy username: {proxy_config['username']}")
            
            # Get current IP for logging
            current_ip = self.proxy_manager.get_current_ip()
            if current_ip:
                Logger.info(f"Current IP: {current_ip}")
            else:
                Logger.warning("Could not determine current proxy IP")
        
        try:
            # Use undetected-chromedriver with proxy authentication extension
            # This provides the best anti-detection while supporting authenticated proxies
            if proxy_config:
                # Create a temporary proxy auth extension for Chrome
                # This allows undetected-chromedriver to use authenticated proxies
                
                # Create proxy authentication extension
                self.extension_dir = tempfile.mkdtemp()
                extension_dir = self.extension_dir
                manifest = {
                    "version": "1.0.0",
                    "manifest_version": 2,
                    "name": "Proxy Auth",
                    "permissions": [
                        "proxy",
                        "webRequest",
                        "webRequestBlocking",
                        "<all_urls>"
                    ],
                    "background": {
                        "scripts": ["background.js"],
                        "persistent": True
                    }
                }
                
                background_js = f"""
                // Proxy configuration with DNS leak prevention
                var config = {{
                    mode: "fixed_servers",
                    rules: {{
                        singleProxy: {{
                            scheme: "http",
                            host: "{proxy_config['address']}",
                            port: parseInt({proxy_config['port']})
                        }},
                        bypassList: ["localhost", "127.0.0.1"]
                    }}
                }};
                
                // Set proxy for all requests (including DNS)
                chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{
                    console.log("Proxy configured");
                }});
                
                // Handle proxy authentication
                function callbackFn(details) {{
                    return {{
                        authCredentials: {{
                            username: "{proxy_config['username']}",
                            password: "{proxy_config['password']}"
                        }}
                    }};
                }}
                
                chrome.webRequest.onAuthRequired.addListener(
                    callbackFn,
                    {{urls: ["<all_urls>"]}},
                    ["blocking"]
                );
                
                // Block WebRTC to prevent IP leaks
                chrome.webRequest.onBeforeRequest.addListener(
                    function(details) {{
                        // Block WebRTC-related requests that could leak IP
                        if (details.url.includes('stun:') || details.url.includes('turn:')) {{
                            return {{cancel: true}};
                        }}
                    }},
                    {{urls: ["<all_urls>"]}},
                    ["blocking"]
                );
                """
                
                # Write extension files
                with open(os.path.join(extension_dir, "manifest.json"), "w") as f:
                    json.dump(manifest, f)
                with open(os.path.join(extension_dir, "background.js"), "w") as f:
                    f.write(background_js)
                
                # Load extension in Chrome
                # Note: Don't use --disable-extensions when loading extensions
                options.add_argument(f'--load-extension={extension_dir}')
                options.add_argument('--disable-extensions-except=' + extension_dir)
                
                # Don't set proxy-server argument - let the extension handle it
                # The extension sets the proxy via chrome.proxy.settings
                
                Logger.info("Using undetected-chromedriver with proxy authentication extension")
                self.driver = uc.Chrome(options=options, **config.chrome_driver_kwargs())
            else:
                # Use undetected-chromedriver if no proxy (better anti-detection)
                self.driver = uc.Chrome(options=options, **config.chrome_driver_kwargs())
            
            # Apply enhanced anti-detection scripts (similar to undetected-chromedriver)
            # These scripts help bypass bot detection and prevent IP leaks
            stealth_script = """
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es', 'en-US', 'en']});
            Object.defineProperty(navigator, 'platform', {get: () => 'MacIntel'});
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ? 
                Promise.resolve({ state: Notification.permission }) : 
                originalQuery(parameters)
            );
            
            // Prevent WebRTC IP leaks
            if (window.RTCPeerConnection) {
                const originalRTCPeerConnection = window.RTCPeerConnection;
                window.RTCPeerConnection = function(...args) {
                    const pc = new originalRTCPeerConnection(...args);
                    const originalCreateDataChannel = pc.createDataChannel.bind(pc);
                    pc.createDataChannel = function() {
                        return null; // Disable data channels
                    };
                    return pc;
                };
            }
            
            // Override RTCPeerConnection methods to prevent IP leaks
            if (window.mozRTCPeerConnection) {
                delete window.mozRTCPeerConnection;
            }
            if (window.webkitRTCPeerConnection) {
                delete window.webkitRTCPeerConnection;
            }
            """
            
            # Try to use CDP (Chrome DevTools Protocol) for better stealth
            try:
                self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': stealth_script})
            except:
                # Fallback: execute script directly (less effective but works)
                try:
                    self.driver.execute_script(stealth_script)
                except:
                    # Basic fallback
                    self.driver.execute_script(
                        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                    )
            
            self.driver.set_page_load_timeout(config.PAGE_LOAD_TIMEOUT)
            
            # Verify proxy is working correctly (only if using proxy)
            if proxy_config:
                try:
                    Logger.info("Verifying proxy connection...")
                    # Navigate to IP check page
                    self.driver.get("https://ipv4.webshare.io/")
                    time.sleep(2)
                    actual_ip = self.driver.find_element(By.TAG_NAME, "body").text.strip()
                    Logger.info(f"Browser IP (via proxy): {actual_ip}")
                    
                    # Compare with expected IP from proxy manager
                    expected_ip = self.proxy_manager.get_current_ip()
                    if expected_ip:
                        if actual_ip == expected_ip:
                            Logger.success("✓ Proxy verification successful - IPs match")
                        else:
                            Logger.warning(
                                f"⚠ IP mismatch! Browser IP: {actual_ip}, "
                                f"Expected: {expected_ip} (may be due to rotation)"
                            )
                    else:
                        Logger.warning("Could not get expected IP from proxy manager")
                except Exception as e:
                    Logger.warning(f"Could not verify proxy IP in browser: {e}")
                    # Don't fail - continue anyway as proxy might still work
            
            return True
        except Exception as e:
            Logger.error(f"Error setting up driver: {e}")
            return False
    
    def _build_search_url(self, operation: str, page: int = 1) -> str:
        """
        Build Idealista search URL.
        
        URL format:
        - Page 1: https://www.idealista.com/{operation}-viviendas/valencia-valencia/
        - Page N: https://www.idealista.com/{operation}-viviendas/valencia-valencia/pagina-{N}.htm
        
        Args:
            operation: 'venta' or 'alquiler'
            page: Page number (default: 1)
        
        Returns:
            Full URL string
        """
        if operation == 'venta':
            base_path = f"{config.BASE_URL}/venta-viviendas/valencia-valencia"
        elif operation == 'alquiler':
            base_path = f"{config.BASE_URL}/alquiler-viviendas/valencia-valencia"
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        if page > 1:
            return f"{base_path}/pagina-{page}.htm"
        return f"{base_path}/"
    
    def _has_next_page(self) -> bool:
        """
        Check if there's a next page available.
        TODO: Update selectors based on actual Idealista pagination structure.
        User should inspect the page and provide the correct selectors.
        """
        try:
            # Try multiple possible selectors for next button
            selectors = [
                'a[title="Siguiente"]',
                'a.next',
                '.pagination-next',
                'a[aria-label="Siguiente"]',
                'a.pagination-next',
                '.pagination a:last-child'
            ]
            
            for selector in selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    # Check if button is disabled
                    classes = next_button.get_attribute('class') or ''
                    if 'disabled' in classes.lower():
                        return False
                    return next_button.is_enabled()
                except NoSuchElementException:
                    continue
            
            # Fallback: assume there's a next page if we can't determine
            # This is safer than stopping prematurely
            Logger.warning("Could not determine pagination status, assuming next page exists")
            return True
            
        except Exception as e:
            Logger.warning(f"Error checking next page: {e}")
            return True  # Default to True to continue scraping
    
    def _scrape_page(self, operation: str, page: int) -> List[Dict]:
        """
        Scrape a single page of results with retry logic.
        
        Args:
            operation: 'venta' or 'alquiler'
            page: Page number
        
        Returns:
            List of property dictionaries
        """
        url = self._build_search_url(operation, page)
        
        for attempt in range(config.MAX_RETRIES_PER_PAGE):
            try:
                Logger.info(f"Page {page} - Attempt {attempt + 1}/{config.MAX_RETRIES_PER_PAGE}")
                Logger.info(f"URL: {url}")
                
                # Navigate to page
                self.driver.get(url)
                
                # Wait for page to load and simulate human behavior
                time.sleep(random.uniform(2, 4))  # Initial page load wait
                
                # Simulate human-like scrolling behavior
                try:
                    # Scroll down slowly (like a human reading)
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
                    time.sleep(random.uniform(1, 2))
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                    time.sleep(random.uniform(1, 2))
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(random.uniform(1, 2))
                    # Scroll back up a bit
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.7);")
                    time.sleep(random.uniform(0.5, 1.5))
                except:
                    pass
                
                # Additional delay before scraping (appears more human-like)
                time.sleep(self._get_random_delay())
                
                # Wait for listings to load
                WebDriverWait(self.driver, config.PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR, 
                        'article.item, div.item, [data-id], .item-container'
                    ))
                )
                
                # Find property listings
                properties = []
                selectors = ['article.item', 'div.item', '[data-id]', '.item-container']
                
                for selector in selectors:
                    try:
                        properties = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if properties:
                            Logger.info(f"Found {len(properties)} properties using selector: {selector}")
                            break
                    except:
                        continue
                
                if not properties:
                    Logger.warning(f"No properties found on page {page}")
                    return []
                
                # Extract data from each property
                page_data = []
                for idx, prop in enumerate(properties, 1):
                    try:
                        data = PropertyExtractor.extract_property_data(
                            prop, 
                            operation, 
                            page, 
                            config.BASE_URL
                        )
                        if data['heading']:  # Only add if we got a heading
                            page_data.append(data)
                            Logger.info(f"  [{idx}/{len(properties)}] {data['heading'][:50]}...")
                    except Exception as e:
                        Logger.warning(f"  [{idx}/{len(properties)}] Error extracting property: {e}")
                        continue
                
                return page_data
                
            except TimeoutException:
                Logger.warning(f"Timeout loading page {page} (attempt {attempt + 1})")
                if attempt < config.MAX_RETRIES_PER_PAGE - 1:
                    delay = config.RETRY_DELAY_BASE * (2 ** attempt)
                    Logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    Logger.error(f"Failed to load page {page} after {config.MAX_RETRIES_PER_PAGE} attempts")
                    return []
                    
            except WebDriverException as e:
                Logger.error(f"WebDriver error on page {page}: {e}")
                # Try rotating proxy on driver errors
                if attempt < config.MAX_RETRIES_PER_PAGE - 1:
                    Logger.info("Attempting proxy rotation...")
                    # The proxy endpoint automatically rotates IPs, so we just reconnect
                    if self._rotate_proxy():
                        delay = config.RETRY_DELAY_BASE * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        return []
                else:
                    return []
                    
            except Exception as e:
                Logger.error(f"Unexpected error on page {page}: {e}")
                if attempt < config.MAX_RETRIES_PER_PAGE - 1:
                    delay = config.RETRY_DELAY_BASE * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return []
        
        return []
    
    def _rotate_proxy(self) -> bool:
        """
        Rotate proxy by reconnecting (rotating endpoint gives new IP automatically).
        
        Returns:
            True if successful, False otherwise
        """
        Logger.info("Rotating proxy (reconnecting to get new IP)...")
        
        # Get new IP from rotating endpoint
        new_ip = self.proxy_manager.get_current_ip()
        if new_ip:
            Logger.info(f"New IP obtained: {new_ip}")
        
        self.pages_scraped_with_current_proxy = 0
        
        # Reconnect driver to get new IP
        return self._setup_driver(use_proxy=True)
    
    def _should_rotate_proxy(self) -> bool:
        """Check if proxy should be rotated"""
        return (
            self.pages_scraped_with_current_proxy >= 
            config.ROTATE_PROXY_EVERY_N_PAGES
        )
    
    def scrape_operation(self, operation: str, start_page: int = 1):
        """
        Scrape all pages for a given operation.
        
        Args:
            operation: 'venta' or 'alquiler'
            start_page: Page number to start from (for resuming)
        """
        Logger.info(f"\n{'='*70}")
        Logger.info(f"Starting scrape for: {operation}")
        Logger.info(f"{'='*70}")
        
        data_manager = DataManager(operation)
        previous_page_urls: Optional[FrozenSet[str]] = None

        # Initialize driver with proxy
        if not self._setup_driver(use_proxy=True):
            Logger.error("Failed to initialize driver")
            return
        
        page = start_page
        total_properties = 0
        
        try:
            while True:
                # Check if proxy should be rotated
                if self._should_rotate_proxy():
                    Logger.info(f"Rotating proxy after {self.pages_scraped_with_current_proxy} pages")
                    if not self._rotate_proxy():
                        Logger.error("Failed to rotate proxy, stopping")
                        break
                
                # Scrape page
                page_data = self._scrape_page(operation, page)
                
                if not page_data:
                    Logger.warning(f"No data on page {page}, checking for next page...")
                    # Check if there's actually a next page
                    if not self._has_next_page():
                        Logger.info("No more pages available")
                        run_session.mark_operation_complete(operation)
                        break
                    page += 1
                    continue

                page_urls = frozenset(
                    (p.get("url") or "").strip()
                    for p in page_data
                    if (p.get("url") or "").strip()
                )
                # Need enough URLs to avoid false positives when extraction fails
                if (
                    previous_page_urls is not None
                    and len(page_urls) >= 8
                    and page_urls == previous_page_urls
                    and page > 1
                ):
                    Logger.warning(
                        f"Page {page} has the same listings as the previous page — "
                        "Idealista is repeating the last page; stopping this operation."
                    )
                    run_session.mark_operation_complete(operation)
                    break
                previous_page_urls = page_urls

                # Save data incrementally
                data_manager.save_properties(page_data)
                run_session.set_page_completed(operation, page)
                total_properties += len(page_data)
                
                Logger.success(
                    f"Page {page} complete: {len(page_data)} properties | "
                    f"Total: {total_properties} properties"
                )
                
                # Check for next page
                if not self._has_next_page():
                    Logger.info("Reached last page")
                    run_session.mark_operation_complete(operation)
                    break
                
                # Delay before next page
                delay = self._get_random_delay()
                Logger.info(f"Waiting {delay:.2f} seconds before next page...")
                time.sleep(delay)
                
                page += 1
                self.pages_scraped_with_current_proxy += 1
                
        except KeyboardInterrupt:
            Logger.warning("Scraping interrupted by user — session kept; rerun without IDEALISTA_FRESH to resume")
        except Exception as e:
            Logger.error(f"Error during scraping: {e}")
            import traceback

            traceback.print_exc()
            sys.stderr.flush()
        finally:
            if self.driver:
                self.driver.quit()
                Logger.info("Driver closed")
            
            # Clean up extension directory if it was created
            if self.extension_dir and os.path.exists(self.extension_dir):
                try:
                    import shutil
                    shutil.rmtree(self.extension_dir)
                    Logger.info("Cleaned up extension directory")
                except:
                    pass
            
            # Clean up user data directory if it was created
            if self.user_data_dir and os.path.exists(self.user_data_dir):
                try:
                    import shutil
                    shutil.rmtree(self.user_data_dir)
                    Logger.info("Cleaned up Chrome user data directory")
                except:
                    pass
            
            Logger.success(f"\nScraping complete for {operation}")
            Logger.success(f"Total properties scraped: {total_properties}")
    
    def scrape(self, operations: Optional[List[str]] = None, resume: bool = True):
        """
        Main scraping function - scrape all operations.

        Resume uses ``scraper_checkpoint.json`` (session): only continues after an
        interrupted run (``in_progress``). A finished operation always starts at page 1
        on the next run unless ``IDEALISTA_RESUME_FROM_CSV=1`` (legacy CSV max page).

        Set ``IDEALISTA_FRESH=1`` to clear the session and start every operation at page 1.

        Args:
            operations: List of operations to scrape (default: all from config)
            resume: If True, apply session (or CSV) resume rules; if False, always page 1
        """
        operations = operations or config.OPERATIONS

        if config.FRESH_RUN:
            run_session.clear_all()
            Logger.info("IDEALISTA_FRESH=1 — cleared run session; each operation starts at page 1")

        for operation in operations:
            start_page = 1
            data_manager = DataManager(operation)

            if resume:
                if config.FRESH_RUN:
                    start_page = 1
                elif config.RESUME_FROM_CSV:
                    last_page = data_manager.get_last_page()
                    if last_page > 0:
                        if not data_manager.is_page_complete(last_page, expected_count=30):
                            Logger.warning(
                                f"Page {last_page} appears incomplete (CSV mode). Re-scraping that page."
                            )
                            start_page = last_page
                        else:
                            Logger.info(f"Resuming {operation} from page {last_page + 1} (IDEALISTA_RESUME_FROM_CSV)")
                            start_page = last_page + 1
                else:
                    st = run_session.get_operation(operation)
                    if st["in_progress"] and st["last_completed_page"] > 0:
                        last_page = st["last_completed_page"]
                        if not data_manager.is_page_complete(last_page, expected_count=30):
                            Logger.warning(
                                f"Page {last_page} appears incomplete. "
                                f"Will re-scrape it to ensure all properties are captured."
                            )
                            start_page = last_page
                        else:
                            Logger.info(
                                f"Resuming {operation} from page {last_page + 1} (interrupted run)"
                            )
                            start_page = last_page + 1
                    else:
                        Logger.info(
                            f"Starting {operation} at page 1 (session: not in progress or fresh)"
                        )

            self.scrape_operation(operation, start_page)
            
            # Delay between operations
            if operation != operations[-1]:
                delay = self._get_random_delay() * 2
                Logger.info(f"Waiting {delay:.2f} seconds before next operation...")
                time.sleep(delay)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    Logger.info(
        f"Starting scraper — USE_PROXY={config.USE_PROXY}, HEADLESS_MODE={config.HEADLESS_MODE}, "
        f"FRESH={config.FRESH_RUN}, RESUME_FROM_CSV={config.RESUME_FROM_CSV} "
        f"(IDEALISTA_FRESH=1 clears session → page 1; IDEALISTA_USE_PROXY=0; IDEALISTA_CHROME_VERSION_MAIN=N)"
    )
    scraper = IdealistaScraper()

    try:
        scraper.scrape(resume=True)
    except Exception as e:
        Logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.stderr.flush()

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    main()
