# Real Estate AI Agent – Plan

## Goal
Build an AI chatbot that answers real-estate questions using your **Idealista** listings and (later) **catastro** (cadastre) data. The agent fetches relevant data when queried and uses it as context to answer.

## Data Sources

| Source | Format | Content |
|--------|--------|--------|
| **Idealista** | `idealista_alquiler.csv`, `idealista_venta.csv` (pipe-separated) | Listings: operation, heading, price, rooms, area, floor, neighborhood, description, url, etc. |
| **Catastro** | `46_UA_23012026_SHP.zip` (single zip), `46_U_23012026_CAT` (dir) | Spanish cadastre: province 46 (València). SHP zip contains per-municipality layers (Valencia city = 46900uA 46005 VALENCIA/PARCELA). CAT = text data. |

## Architecture

1. **Data layer**
   - **Idealista**: Load both CSVs into a **SQLite** database. Simple, portable, no server, good for filtering (price, area, rooms, neighborhood, operation).
   - **Optional RAG**: Embed `heading` + `description` and use a vector store (e.g. Chroma or SQLite with sqlite-vss) for semantic search (“piso tranquilo cerca del Turia”).
   - **Catastro**: Load Valencia city parcels from `46_UA_23012026_SHP.zip` (PARCELA layer) into SQLite with REFCAT, via, numero, area, and geometry as GeoJSON for visualization. Listing–parcel links in `listing_parcel` table enrich listings with cadastral data.

2. **Agent (LangChain)**
   - **Tools**:  
     - `query_listings`: filter by operation (alquiler/venta), max price, min/max area, rooms, neighborhood/area text, limit.  
     - `get_listing_by_id` or by URL: return one listing’s full details.  
     - (Optional) `search_listings_semantic`: natural-language search over descriptions.  
     - `get_parcel_data(refcat)`, `get_parcel_geojson(refcat)` (map), `search_parcels_by_address`; `get_listing_with_catastro`, `link_listing_to_parcel`.
   - **Agent**: Use a **ReAct**-style agent (or LCEL with tool-calling) so the LLM decides when to call which tool and then answers from tool results.
   - **Prompt**: System prompt stating the agent is a Valencia real-estate assistant with access to Idealista listings (and later catastro), and that it must base answers on retrieved data.

3. **Interface**
   - Simple **CLI** or **script**: user input → agent run → print response (and optionally sources).

## Implementation Order

1. **Phase 1 – Idealista + Agent**
   - [x] Plan (this document)
   - [ ] Create `agent/` package and dependencies (langchain, langchain-openai or langchain-anthropic, sqlite3).
   - [ ] Script to load `idealista_alquiler.csv` and `idealista_venta.csv` into SQLite (schema matches CSV columns).
   - [ ] LangChain tools that query SQLite (query_listings, get_listing_details).
   - [ ] LangChain agent (ReAct/tool-calling) + system prompt.
   - [ ] Simple chat loop (CLI) that runs the agent and prints answers.
   - [ ] README with setup and usage.

2. **Phase 2 – Catastro** ✅
   - [x] Use `46_UA_23012026_SHP.zip` (single zip); extract Valencia PARCELA shapefile.
   - [x] Load parcels into SQLite (`db/catastro.db`) with REFCAT, via, numero, area_catastro, geojson.
   - [x] Tools: `get_parcel_data`, `get_parcel_geojson` (visualization), `search_parcels_by_address`.
   - [x] Listing enrichment: `listing_parcel` table + `get_listing_with_catastro`, `link_listing_to_parcel`.
   - [ ] Optional: geocode Idealista listings and match to parcel (point-in-polygon) to auto-fill links.

3. **Listing enrichment** ✅
   - [x] Catastro vías loaded into `streets` table (via_code ↔ street name) in catastro.db.
   - [x] `parse_listing.py`: parse heading → street_name, neighborhood; resolve via_code from streets table.
   - [x] `enrich_listings.py`: backfill listings with street_name, neighborhood, via_code, price_int, area_sqm, rooms_int.
   - [x] `query_listings` accepts street_name_contains and via_code for street-based search.

4. **Optional**
   - [ ] Semantic search over listings (embeddings + vector store).
   - [ ] Web UI (e.g. Streamlit/Gradio) instead of or in addition to CLI.

## Database Choice (summary)

- **SQLite** for Idealista (and catastro tabular): no server, single file, easy to version and ship. Good for filters and “show me N listings that match X”.
- **Vector store** (Chroma / FAISS / sqlite-vss): only if we add semantic search over descriptions.
- **Catastro spatial**: SpatiaLite or GeoPackage if we need “listings near this parcel” or “buildings in this block”.

## Files to Create (Phase 1)

```
agent/
├── PLAN.md                 # This file
├── README.md               # Setup + usage
├── requirements.txt       # langchain, langchain-openai, etc.
├── load_idealista_db.py   # Load CSVs → SQLite
├── db/
│   └── idealista.db       # Generated SQLite DB (or under data/)
├── tools/
│   └── listings.py        # LangChain tools: query_listings, get_listing
└── chat.py                # CLI chat loop (if present)
```

Next step: implement Phase 1 (loader, tools, agent, chat).
