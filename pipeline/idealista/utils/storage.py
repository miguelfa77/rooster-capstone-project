# utils/storage.py
"""
Data storage and checkpoint management utilities.
Handles CSV file operations and progress tracking.
"""

import csv
from pathlib import Path
from typing import List, Dict
from config import OUTPUT_DIR, CSV_SEPARATOR
from utils.logger import Logger


class DataManager:
    """Handles data storage and checkpoint management"""
    
    def __init__(self, operation: str):
        """
        Initialize data manager for an operation.
        
        Args:
            operation: 'venta' or 'alquiler'
        """
        self.operation = operation
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # CSV file for this operation
        self.csv_file = self.output_dir / f"idealista_{operation}.csv"
        self.fieldnames = [
            'operation', 'heading', 'price', 'currency', 'period',
            'rooms', 'area', 'floor', 'time_to_center', 'description',
            'url', 'page', 'scraped_at'
        ]
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=self.fieldnames, 
                    delimiter=CSV_SEPARATOR
                )
                writer.writeheader()
            Logger.info(f"Created CSV file: {self.csv_file}")
    
    def save_properties(self, properties: List[Dict]):
        """
        Append properties to CSV file incrementally.
        
        Args:
            properties: List of property dictionaries
        """
        if not properties:
            return
        
        try:
            with open(self.csv_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=self.fieldnames, 
                    delimiter=CSV_SEPARATOR
                )
                
                for prop in properties:
                    # Ensure all fields are present
                    row = {field: prop.get(field, '') for field in self.fieldnames}
                    writer.writerow(row)
            
            Logger.success(f"Saved {len(properties)} properties to {self.csv_file}")
        except Exception as e:
            Logger.error(f"Error saving properties: {e}")
    
    def get_last_page(self) -> int:
        """
        Get last scraped page number from CSV file.
        Useful for resuming interrupted scraping sessions.
        
        Returns:
            Last page number found in CSV, or 0 if file doesn't exist
        """
        if not self.csv_file.exists():
            return 0
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
                max_page = 0
                for row in reader:
                    try:
                        page = int(row.get('page', 0))
                        max_page = max(max_page, page)
                    except ValueError:
                        continue
                return max_page
        except Exception as e:
            Logger.warning(f"Error reading last page: {e}")
            return 0

