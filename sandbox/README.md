# Rooster Sandbox Service

Minimal FastAPI sandbox that executes visualization Python code against tabular rows.

## API

`POST /execute`

```json
{
  "code": "import plotly.express as px\nfig = px.scatter(df, x='x', y='y')\nprint(fig.to_json())",
  "data": [{"x": 1, "y": 2}]
}
```

Response:

```json
{
  "success": true,
  "output_type": "plotly_json",
  "output": "...",
  "error": null
}
```

## Runtime notes

- Timeout enforced at 15 seconds by a subprocess harness.
- Choropleth geometry is supplied per row in the request payload: column **`geom`** holds GeoJSON geometry dicts when the executor sets `include_geometry=true`. The sandbox does not read static GeoJSON files from disk.
- This service does not use a database; do not set **`DATABASE_URL`** (or other DB credentials) on it.
- For Railway deployment, configure:
  - Memory limit: 512MB
  - Network: private networking only
  - Root filesystem read-only where possible, with `/tmp` writable
