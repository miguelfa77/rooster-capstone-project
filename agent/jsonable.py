"""Convert database, pandas, numpy, geometry and pydantic values into plain JSON values."""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

_PANDAS_NULL_TYPES = ("NaTType", "NAType")  # pd.NaT, pd.NA — checked by name to avoid importing pandas


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if type(obj).__name__ in _PANDAS_NULL_TYPES:  # before datetime: NaT is a datetime subclass
        return None
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, Decimal):
        return to_jsonable(float(obj))
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if hasattr(obj, "__geo_interface__"):
        return to_jsonable(obj.__geo_interface__)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return to_jsonable(obj.tolist())
    return str(obj)
