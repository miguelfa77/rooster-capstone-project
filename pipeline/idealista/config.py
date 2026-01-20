# config.py
"""
Configuration settings for Idealista scraper.
Centralized configuration for easy modification.
"""

from pathlib import Path

# Base URLs
BASE_URL = "https://www.idealista.com"
OPERATIONS = ['alquiler', 'venta']  # Order matters - alquiler first

# Scraping behavior
DELAY_MIN_SECONDS = 4
DELAY_MAX_SECONDS = 8
PAGE_LOAD_TIMEOUT = 20
MAX_RETRIES_PER_PAGE = 3
RETRY_DELAY_BASE = 5  # Base seconds for exponential backoff

# Proxy settings
ROTATE_PROXY_EVERY_N_PAGES = 5  # Rotate proxy every N pages
PROXY_TEST_TIMEOUT = 10

# Data storage
OUTPUT_DIR = Path("data")
CSV_SEPARATOR = "|"  # Pipe separator instead of comma
CHECKPOINT_FILE = "scraper_checkpoint.json"

# Logging
LOG_TIMESTAMPS = True

