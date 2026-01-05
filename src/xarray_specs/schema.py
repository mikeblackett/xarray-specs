from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import attrs as at
import numpy as np

__all__ = [
    'DataArraySchema',
    'DatasetSchema',
    'VariableSchema',
]

# NOTE: We don't allow `Mapping` with `Hashable` keys, because they are intended to
# be passed `TypedDict`s which require `str` keys. This should be mentioned in the docs.
type Attrs = Mapping[str, Any] | None
type Coords = Mapping[str, Any] | None
type DataVars = Mapping[str, Any] | None
type Dims = Sequence[Hashable] | None
type DType = np.dtype | None
type Encoding = Mapping[str, Any] | None
type Name = Hashable | None
type Shape = Sequence[int] | None
type Size = int | None


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


@at.frozen(kw_only=True)
class DatasetSchema(Schema):
    attrs: Attrs
    coords: Coords
    data_vars: DataVars
    dims: Dims
    encoding: Encoding
