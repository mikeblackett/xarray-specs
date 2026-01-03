from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import attrs as at
import numpy as np

__all__ = [
    'DataArraySchema',
    'VariableSchema',
]

type Attrs = Mapping[str, Any] | None
type Coords = Mapping[str, Any] | None
type Dims = Sequence[str] | None
type DType = np.dtype | None
type DTypes = Mapping[str, np.dtype] | None
type Encoding = Mapping[str, Any] | None
type Shape = Sequence[int] | None
type Size = int | None
type Name = str | None


@at.frozen(kw_only=True)
class Schema: ...


@at.frozen(kw_only=True)
class VariableSchema(Schema):
    attrs: Attrs
    dims: Dims
    dtype: DType
    encoding: Encoding
    shape: Shape
    size: Size


@at.frozen(kw_only=True)
class DataArraySchema(VariableSchema):
    coords: Coords
    name: Name
