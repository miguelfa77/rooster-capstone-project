"""Write api/openapi.json, including the SSE event and primitive schemas.

SSE payloads are not request/response bodies, so FastAPI does not list them;
they are added under components.schemas for frontend type generation.
PrimitiveEvent.primitive is declared as PrimitivePayload (a dict at runtime).

    python -m api.export_openapi
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from agent.synthesizer import PrimitiveBlock
from agent.turn import TurnEvent
from api.main import app
from api.models import PrimitivePayload

OUT_PATH = Path(__file__).parent / "openapi.json"
REF = "#/components/schemas/{model}"


def _add(schemas: dict, name: str, adapter: TypeAdapter) -> None:
    schema = adapter.json_schema(ref_template=REF)
    schemas.update(schema.pop("$defs", {}))
    schemas[name] = schema


def build_openapi() -> dict:
    spec = app.openapi()
    schemas = spec.setdefault("components", {}).setdefault("schemas", {})
    _add(schemas, "TurnEvent", TypeAdapter(TurnEvent))
    _add(schemas, "PrimitiveBlock", TypeAdapter(PrimitiveBlock))
    _add(schemas, "PrimitivePayload", TypeAdapter(PrimitivePayload))
    schemas["PrimitiveEvent"]["properties"]["primitive"] = {"$ref": REF.format(model="PrimitivePayload")}
    return spec


if __name__ == "__main__":
    OUT_PATH.write_text(json.dumps(build_openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
