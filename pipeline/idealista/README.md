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
    ├── storage.py         # CSV storage
    └── run_session.py     # Run session (resume after crash vs fresh start)
```

## Features

- **Rotating Proxy Support**: Uses a rotating proxy endpoint that automatically provides new IPs
- **Incremental Saves**: Saves data after each page to prevent data loss
- **Resume capability**: After a crash, continues from the last saved page (see `scraper_checkpoint.json`). A **finished** run always starts again at page 1 unless you use CSV-based resume.
- **Last-page loop guard**: Stops if the next URL returns the **same listings** as the previous page (Idealista often repeats the final page while pagination still says “next”).
- **Robust Error Handling**: Retries with exponential backoff
- **Progress Tracking**: Detailed logging with timestamps
- **Pipe-Separated CSV**: Uses `|` separator to avoid comma conflicts
- **Time series in the DB**: The CSV **appends** one row per listing per observation (`url` + `scraped_at`). Load with `load_idealista_raw` → `load_listings`; history for charts lives in **`core.listing_snapshots`** (and `core.listings` is current state).

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

**Recommended** (repo root, resolves imports correctly):

```bash
cd /path/to/rooster-capstone-project
source ./bin/activate   # or your venv
python -m pipeline.idealista.idealista_scraper
```

Do **not** add a `.py` suffix to `-m`. You can also run `python pipeline/idealista/idealista_scraper.py` from the repo root (it adds the repo to `sys.path`). The script `chdir`s to `pipeline/idealista` on startup so `data/` and checkpoints stay next to the scraper.

For redirected logs / `nohup`, use unbuffered output:

```bash
python -u -m pipeline.idealista.idealista_scraper
```

**Alternative 1: Use the requests-html scraper (renders JavaScript without Selenium):**
```bash
python idealista_scraper_html.py
```

**Alternative 2: Use the simple requests-based scraper (fastest, but gets blocked by Idealista):**
```bash
python idealista_scraper_requests.py
```

The scraper will:
1. Start with 'alquiler' (rental) listings
2. Then scrape 'venta' (sale) listings
3. Save data incrementally to `data/idealista_alquiler.csv` and `data/idealista_venta.csv`
4. Track progress in `scraper_checkpoint.json` and resume only if that run was **interrupted**

## Resume session & environment

| Variable | Effect |
|----------|--------|
| *(default)* | Each operation starts at **page 1** if the last run **completed** normally. If the process died mid-run (`in_progress` in the session file), it **resumes** after the last fully saved page. |
| `IDEALISTA_FRESH=1` | Deletes `scraper_checkpoint.json` and starts **every** operation at page 1 (intentional full pass). Existing CSV rows are still deduplicated by URL. |
| `IDEALISTA_RESUME_FROM_CSV=1` | Legacy: start page = max `page` column in the CSV + 1 (old behavior). |

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

## Scraper Options

1. **`idealista_scraper.py`** (Recommended)
   - Uses Selenium with `undetected-chromedriver`
   - Most reliable for Idealista's DataDome bot protection
   - Uses real Chrome browser (best TLS fingerprint matching)
   - Handles JavaScript rendering automatically
   - Requires ChromeDriver (auto-downloaded by undetected-chromedriver)
   - Best success rate for protected sites

2. **`idealista_scraper_html.py`**
   - Uses `requests-html` library
   - Renders JavaScript via PyQt5/PyQtWebKit (no ChromeDriver needed)
   - Lighter than Selenium but may still get blocked
   - Good alternative if Selenium doesn't work

3. **`idealista_scraper_requests.py`**
   - Uses `requests` + `BeautifulSoup`
   - Fastest option, but gets blocked by Idealista (403 errors)
   - No JavaScript rendering capability
   - Useful for testing or sites without bot protection

## Configuration

Key settings in `config.py`:
- `DELAY_MIN_SECONDS` / `DELAY_MAX_SECONDS`: Random delay between pages
- `ROTATE_PROXY_EVERY_N_PAGES`: How often to rotate proxy
- `MAX_RETRIES_PER_PAGE`: Retry attempts for failed pages
- `HEADLESS_MODE`: Run browser in background (default: False - visible browser is less detectable)
- `OUTPUT_DIR`: Directory for output files

## Troubleshooting

**Process runs but log file stays empty / no console output**  
Python buffers `stdout` when it is not a terminal (e.g. `nohup`, `>> file`). `Logger` uses `flush=True`, and `idealista_scraper.py` sets `PYTHONUNBUFFERED` and line-buffering when run as `__main__`. If logs still lag, use:

```bash
python -u idealista_scraper.py
```

**Process runs but nothing is scraped / Chrome seems stuck**  
1. **Chrome vs chromedriver** — A wrong pinned major version can hang. This project no longer hard-codes `version_main`; uc auto-matches Chrome. If it still hangs, set your installed major version, e.g. `export IDEALISTA_CHROME_VERSION_MAIN=131` (check *Chrome → About Google Chrome*).  
2. **Proxy** — The first browser step can block on the proxy or `ipv4.webshare.io`. Test without proxy: `export IDEALISTA_USE_PROXY=0` (Idealista may block your home IP; use only to confirm the browser starts).  
3. **Watch the browser window** — With `HEADLESS_MODE = False`, you should see navigation; if the window never appears, Chrome/driver setup failed (check terminal for errors after the flush fix).

## Notes

- **ChromeDriver**: `undetected-chromedriver` will automatically download the correct ChromeDriver version on first run
- **Proxy**: Proxy endpoint automatically rotates IPs on each connection
- **Browser Window**: By default, the scraper runs with a visible browser window (better for anti-detection). Set `HEADLESS_MODE = True` in `config.py` to run in background
- **Pagination**: Pagination detection may need adjustment based on Idealista's HTML structure
- **Bot Protection**: Idealista uses DataDome protection - the Selenium scraper with `undetected-chromedriver` is designed to bypass this
- **Ethics**: Always respect Idealista's Terms of Service and robots.txt

