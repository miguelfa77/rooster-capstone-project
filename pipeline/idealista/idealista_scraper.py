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
from typing import List, Dict, Optional
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
from fake_useragent import UserAgent

# Local imports
import config
from proxies import RotatingProxyManager, PROXY_ENDPOINT_CONFIG
from utils import Logger, PropertyExtractor, DataManager


class IdealistaScraper:
    """Main scraper class"""
    
    def __init__(self):
        """Initialize scraper with rotating proxy manager"""
        self.ua = UserAgent()
        self.driver = None
        self.proxy_manager = RotatingProxyManager(PROXY_ENDPOINT_CONFIG)
        self.pages_scraped_with_current_proxy = 0
    
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
        
        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f'user-agent={self.ua.random}')
        options.add_argument('--window-size=1920,1080')
        
        # Proxy configuration
        if use_proxy:
            proxy_str = self.proxy_manager.get_proxy_string()
            options.add_argument(f'--proxy-server={proxy_str}')
            
            # Get current IP for logging
            current_ip = self.proxy_manager.get_current_ip()
            if current_ip:
                Logger.info(f"Using proxy endpoint: {self.proxy_manager.config['address']}:{self.proxy_manager.config['port']}")
                Logger.info(f"Current IP: {current_ip}")
            else:
                Logger.warning("Could not determine current proxy IP")
        
        try:
            self.driver = uc.Chrome(options=options, version_main=None)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            self.driver.set_page_load_timeout(config.PAGE_LOAD_TIMEOUT)
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
                
                self.driver.get(url)
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
                        break
                    page += 1
                    continue
                
                # Save data incrementally
                data_manager.save_properties(page_data)
                total_properties += len(page_data)
                
                Logger.success(
                    f"Page {page} complete: {len(page_data)} properties | "
                    f"Total: {total_properties} properties"
                )
                
                # Check for next page
                if not self._has_next_page():
                    Logger.info("Reached last page")
                    break
                
                # Delay before next page
                delay = self._get_random_delay()
                Logger.info(f"Waiting {delay:.2f} seconds before next page...")
                time.sleep(delay)
                
                page += 1
                self.pages_scraped_with_current_proxy += 1
                
        except KeyboardInterrupt:
            Logger.warning("Scraping interrupted by user")
        except Exception as e:
            Logger.error(f"Error during scraping: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.driver:
                self.driver.quit()
                Logger.info("Driver closed")
            
            Logger.success(f"\nScraping complete for {operation}")
            Logger.success(f"Total properties scraped: {total_properties}")
    
    def scrape(self, operations: Optional[List[str]] = None, resume: bool = True):
        """
        Main scraping function - scrape all operations.
        
        Args:
            operations: List of operations to scrape (default: all from config)
            resume: If True, resume from last scraped page
        """
        operations = operations or config.OPERATIONS
        
        for operation in operations:
            start_page = 1
            
            if resume:
                data_manager = DataManager(operation)
                last_page = data_manager.get_last_page()
                if last_page > 0:
                    Logger.info(f"Resuming {operation} from page {last_page + 1}")
                    start_page = last_page + 1
            
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
    scraper = IdealistaScraper()
    
    try:
        scraper.scrape(resume=True)
    except Exception as e:
        Logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
