# config.py
"""
Configuration settings for Idealista scraper.
Centralized configuration for easy modification.
"""

import os
from pathlib import Path

# Set false to debug without proxy (e.g. stuck on proxy IP check): IDEALISTA_USE_PROXY=0
USE_PROXY = os.environ.get("IDEALISTA_USE_PROXY", "true").lower() in ("1", "true", "yes")

# Start every operation at page 1 and clear session file (no resume from interrupted run).
FRESH_RUN = os.environ.get("IDEALISTA_FRESH", "false").lower() in ("1", "true", "yes")

# Old behavior: resume from max `page` column in CSV (ignores session file for start page).
RESUME_FROM_CSV = os.environ.get("IDEALISTA_RESUME_FROM_CSV", "false").lower() in (
    "1",
    "true",
    "yes",
)


def chrome_driver_kwargs() -> dict:
    """
    undetected_chromedriver options. By default omit version_main (auto-match installed Chrome).
    If Chrome/driver mismatch hangs, set e.g. IDEALISTA_CHROME_VERSION_MAIN=131
    """
    v = os.environ.get("IDEALISTA_CHROME_VERSION_MAIN", "").strip()
    if v.isdigit():
        return {"version_main": int(v)}
    return {}

# Base URLs
BASE_URL = "https://www.idealista.com"
OPERATIONS = ['alquiler', 'venta']  # Order matters - alquiler first

# Scraping behavior
DELAY_MIN_SECONDS = 8  # Increased delays to appear more human-like
DELAY_MAX_SECONDS = 15  # Longer random delays
PAGE_LOAD_TIMEOUT = 30  # Increased timeout
MAX_RETRIES_PER_PAGE = 3
RETRY_DELAY_BASE = 10  # Longer base delay for retries

# Proxy settings
ROTATE_PROXY_EVERY_N_PAGES = 3  # Rotate proxy more frequently to avoid detection
PROXY_TEST_TIMEOUT = 10

# Browser settings (for Selenium scraper)
HEADLESS_MODE = False  # Set to True to run browser in background (may be more detectable)

# Data storage
OUTPUT_DIR = Path("data")
CSV_SEPARATOR = "|"  # Pipe separator instead of comma
CHECKPOINT_FILE = "scraper_checkpoint.json"

# Logging
LOG_TIMESTAMPS = True

