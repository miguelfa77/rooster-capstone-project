# utils/storage.py
"""
Data storage and checkpoint management utilities.
Handles CSV file operations and progress tracking.
"""

import csv
from pathlib import Path
from typing import List, Dict
from ..config import OUTPUT_DIR, CSV_SEPARATOR
from .logger import Logger


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
        Append each observation to the CSV (time series: same url may appear many times
        with different scraped_at). Only skips exact duplicates within this batch
        (same url + same scraped_at).
        """
        if not properties:
            return

        seen_keys = set()
        to_write: List[Dict] = []
        skipped_batch_dupes = 0

        for prop in properties:
            url = (prop.get("url") or "").strip()
            if not url:
                continue
            scraped_at = (prop.get("scraped_at") or "").strip()
            key = (url, scraped_at)
            if key in seen_keys:
                skipped_batch_dupes += 1
                continue
            seen_keys.add(key)
            to_write.append(prop)

        if not to_write:
            if skipped_batch_dupes:
                Logger.info(f"Skipped {skipped_batch_dupes} duplicate (url, scraped_at) in batch")
            return

        try:
            with open(self.csv_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.fieldnames,
                    delimiter=CSV_SEPARATOR,
                )
                for prop in to_write:
                    row = {field: prop.get(field, "") for field in self.fieldnames}
                    writer.writerow(row)

            msg = f"Appended {len(to_write)} row(s) to {self.csv_file}"
            if skipped_batch_dupes:
                msg += f" (skipped {skipped_batch_dupes} duplicate key in batch)"
            Logger.success(msg)
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
    
    def get_page_count(self, page: int) -> int:
        """
        Get number of properties saved for a specific page.
        Useful for detecting incomplete pages.
        
        Args:
            page: Page number to check
        
        Returns:
            Number of properties found for that page
        """
        if not self.csv_file.exists():
            return 0
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
                count = 0
                for row in reader:
                    try:
                        if int(row.get('page', 0)) == page:
                            count += 1
                    except ValueError:
                        continue
                return count
        except Exception as e:
            Logger.warning(f"Error reading page count: {e}")
            return 0
    
    def is_page_complete(self, page: int, expected_count: int = 30) -> bool:
        """
        Check if a page appears to be complete.
        A page is considered incomplete if it has fewer than expected_count properties.
        
        Args:
            page: Page number to check
            expected_count: Expected number of properties per page (default: 30)
        
        Returns:
            True if page appears complete, False otherwise
        """
        actual_count = self.get_page_count(page)
        if actual_count == 0:
            return False  # Page not scraped at all
        if actual_count < expected_count * 0.8:  # Less than 80% of expected
            Logger.warning(
                f"Page {page} appears incomplete: {actual_count} properties "
                f"(expected ~{expected_count})"
            )
            return False
        return True

