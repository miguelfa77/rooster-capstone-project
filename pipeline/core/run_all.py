#!/usr/bin/env python3
"""
Run all core loaders. Execute from repo root:
  python -m pipeline.core.run_all
"""

from . import enrich_listings
from . import load_listings
from . import load_neighborhoods
from . import load_parcels
from . import load_streets


def main() -> None:
    print("=== load_listings ===")
    load_listings.main()
    print("\n=== load_streets ===")
    load_streets.main()
    print("\n=== load_neighborhoods ===")
    load_neighborhoods.main()
    print("\n=== load_parcels ===")
    load_parcels.main()
    print("\n=== enrich_listings ===")
    enrich_listings.main()
    print("\nDone.")


if __name__ == "__main__":
    main()
