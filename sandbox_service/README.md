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
- GeoJSON available at `/data/valencia_barrios.geojson`.
- For Railway deployment, configure:
  - Memory limit: 512MB
  - Network: private networking only
  - Root filesystem read-only where possible, with `/tmp` writable
