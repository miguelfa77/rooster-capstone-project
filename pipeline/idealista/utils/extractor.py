# utils/extractor.py
"""
Property data extraction utilities.
Extracts data from Idealista property listing cards.
"""

import re
from datetime import datetime
from typing import Dict, Tuple
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from .logger import Logger


class PropertyExtractor:
    """Extracts data from property listing cards"""
    
    @staticmethod
    def extract_heading(property_element, base_url: str) -> Tuple[str, str]:
        """
        Extract heading/title and URL from property card.
        
        Args:
            property_element: Selenium WebElement for property card
            base_url: Base URL for Idealista
        
        Returns:
            Tuple of (heading_text, url)
        """
        try:
            heading_elem = property_element.find_element(
                By.CSS_SELECTOR, 
                'a.item-link[role="heading"]'
            )
            heading = heading_elem.text.strip()
            href = heading_elem.get_attribute('href')
            
            # Ensure full URL
            if href and href.startswith('/'):
                url = f"{base_url}{href}"
            else:
                url = href or ""
            
            return heading, url
        except NoSuchElementException:
            return "", ""
        except Exception as e:
            Logger.error(f"Error extracting heading: {e}")
            return "", ""
    
    @staticmethod
    def extract_price_info(property_element) -> Dict[str, str]:
        """
        Extract price, currency, and period from property card.
        
        Args:
            property_element: Selenium WebElement for property card
        
        Returns:
            Dict with 'price', 'currency', 'period'
        """
        result = {'price': '', 'currency': '', 'period': ''}
        
        try:
            price_elem = property_element.find_element(
                By.CSS_SELECTOR, 
                'div.price-row span.item-price'
            )
            price_text = price_elem.text.strip()
            price_html = price_elem.get_attribute('innerHTML') or ""
            
            # Extract numeric price (remove dots used as thousand separators)
            price_match = re.search(r'(\d+(?:\.\d+)?)', price_text.replace('.', ''))
            if price_match:
                result['price'] = price_match.group(1)
            
            # Extract currency
            if '€' in price_text or '€' in price_html:
                result['currency'] = '€'
            
            # Extract period
            if '/mes' in price_text or '/mes' in price_html:
                result['period'] = 'mes'
            elif '/año' in price_text or '/año' in price_html:
                result['period'] = 'año'
                
        except NoSuchElementException:
            pass
        except Exception as e:
            Logger.error(f"Error extracting price: {e}")
        
        return result
    
    @staticmethod
    def extract_details(property_element) -> Dict[str, str]:
        """
        Extract property details (rooms, area, floor, time to center).
        
        Args:
            property_element: Selenium WebElement for property card
        
        Returns:
            Dict with 'rooms', 'area', 'floor', 'time_to_center'
        """
        result = {
            'rooms': '',
            'area': '',
            'floor': '',
            'time_to_center': ''
        }
        
        try:
            details_container = property_element.find_element(
                By.CSS_SELECTOR, 
                'div.item-detail-char'
            )
            detail_items = details_container.find_elements(
                By.CSS_SELECTOR, 
                'span.item-detail'
            )
            
            for item in detail_items:
                text = item.text.strip()
                
                if 'hab.' in text or 'dorm' in text:
                    result['rooms'] = text
                elif 'm²' in text:
                    result['area'] = text
                elif 'Planta' in text or 'planta' in text or 'ascensor' in text:
                    result['floor'] = text
                elif 'minutos' in text or 'minuto' in text:
                    result['time_to_center'] = text
                    
        except NoSuchElementException:
            pass
        except Exception as e:
            Logger.error(f"Error extracting details: {e}")
        
        return result
    
    @staticmethod
    def extract_description(property_element) -> str:
        """
        Extract property description.
        
        Args:
            property_element: Selenium WebElement for property card
        
        Returns:
            Description text or empty string
        """
        try:
            desc_elem = property_element.find_element(
                By.CSS_SELECTOR, 
                'div.item-description.description p.ellipsis'
            )
            return desc_elem.text.strip()
        except NoSuchElementException:
            return ""
        except Exception as e:
            Logger.error(f"Error extracting description: {e}")
            return ""
    
    @staticmethod
    def extract_property_data(property_element, operation: str, page: int, base_url: str) -> Dict:
        """
        Extract all data from a single property card.
        
        Args:
            property_element: Selenium WebElement for property card
            operation: 'venta' or 'alquiler'
            page: Page number
            base_url: Base URL for Idealista
        
        Returns:
            Dictionary with all property fields
        """
        heading, url = PropertyExtractor.extract_heading(property_element, base_url)
        price_info = PropertyExtractor.extract_price_info(property_element)
        details = PropertyExtractor.extract_details(property_element)
        description = PropertyExtractor.extract_description(property_element)
        
        return {
            'operation': operation,
            'heading': heading,
            'price': price_info['price'],
            'currency': price_info['currency'],
            'period': price_info['period'],
            'rooms': details['rooms'],
            'area': details['area'],
            'floor': details['floor'],
            'time_to_center': details['time_to_center'],
            'description': description,
            'url': url,
            'page': page,
            'scraped_at': datetime.now().isoformat(),
        }

