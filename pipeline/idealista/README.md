# Idealista Scraper

A robust web scraper for extracting real estate listings from Idealista.com (Valencia, Spain).

## Project Structure

```
pipeline/idealista/
├── idealista_scraper.py    # Main scraper entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── proxies/
│   ├── __init__.py
│   └── proxies.py         # Rotating proxy endpoint management
└── utils/
    ├── __init__.py
    ├── logger.py          # Logging utilities
    ├── extractor.py       # Property data extraction
    └── storage.py         # CSV storage and checkpoint management
```

## Features

- **Rotating Proxy Support**: Uses a rotating proxy endpoint that automatically provides new IPs
- **Incremental Saves**: Saves data after each page to prevent data loss
- **Resume Capability**: Can resume from last scraped page if interrupted
- **Robust Error Handling**: Retries with exponential backoff
- **Progress Tracking**: Detailed logging with timestamps
- **Pipe-Separated CSV**: Uses `|` separator to avoid comma conflicts

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure proxy endpoint** in `proxies/proxies.py`:
   ```python
   PROXY_ENDPOINT_CONFIG = {
       'name': 'your_proxy_name',
       'username': 'your_username',
       'password': 'your_password',
       'address': 'your_proxy_endpoint',
       'port': 'your_port'
   }
   ```

3. **Adjust configuration** in `config.py` if needed (delays, timeouts, etc.)

## Usage

Run the scraper:
```bash
python idealista_scraper.py
```

The scraper will:
1. Start with 'alquiler' (rental) listings
2. Then scrape 'venta' (sale) listings
3. Save data incrementally to `data/idealista_alquiler.csv` and `data/idealista_venta.csv`
4. Resume from last page if interrupted

## Output Format

CSV files with pipe (`|`) separator containing:
- `operation`: 'venta' or 'alquiler'
- `heading`: Property title
- `price`: Price value
- `currency`: Currency symbol (€)
- `period`: 'mes' or 'año' for rentals
- `rooms`: Number of rooms
- `area`: Area in m²
- `floor`: Floor information
- `time_to_center`: Time to city center
- `description`: Property description
- `url`: Full URL to listing
- `page`: Page number where found
- `scraped_at`: ISO timestamp

## Configuration

Key settings in `config.py`:
- `DELAY_MIN_SECONDS` / `DELAY_MAX_SECONDS`: Random delay between pages
- `ROTATE_PROXY_EVERY_N_PAGES`: How often to rotate proxy
- `MAX_RETRIES_PER_PAGE`: Retry attempts for failed pages
- `OUTPUT_DIR`: Directory for output files

## Notes

- The scraper uses `undetected-chromedriver` to avoid detection
- Proxy endpoint automatically rotates IPs on each connection
- Pagination detection may need adjustment based on Idealista's HTML structure
- Always respect Idealista's Terms of Service and robots.txt

