#!/usr/bin/env python3
"""
Run all raw loaders in sequence. Execute from repo root:
  python -m pipeline.raw.run_all
"""

from . import load_barrios_raw
from . import load_catastro_parcels_raw
from . import load_catastro_vias_raw
from . import load_idealista_raw


def main() -> None:
    print("=== load_idealista_raw ===")
    load_idealista_raw.main()
    print("\n=== load_catastro_vias_raw ===")
    load_catastro_vias_raw.main()
    print("\n=== load_barrios_raw ===")
    load_barrios_raw.main()
    print("\n=== load_catastro_parcels_raw ===")
    load_catastro_parcels_raw.main()
    print("\nDone.")


if __name__ == "__main__":
    main()
