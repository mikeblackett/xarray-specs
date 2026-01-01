from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Self, TypeVar

import attrs as at
import numpy as np
import xarray as xr

from xarray_specs.converter import converter

TXarray = TypeVar('TXarray', xr.DataArray, xr.Dataset)


@at.frozen(kw_only=True)
class Variable:
    attrs: dict[str, Any] = at.field(factory=dict)
    dims: tuple[str, ...] = at.field(factory=tuple)
    dtype: np.dtype | None = None
    name: str | None = None
    shape: tuple[int, ...] = at.field(factory=tuple)
    size: int | None = None


@at.frozen(kw_only=True)
class DataArraySpec:
    attrs: dict[str, Any] = at.field(factory=dict)
    coords: dict[str, Variable] = at.field(factory=dict)
    dims: tuple[str, ...] = at.field(factory=tuple)
    dtype: np.dtype | None = None
    encoding: dict[str, Any] = at.field(factory=dict)
    name: str | None = None
    shape: tuple[int, ...] = at.field(factory=tuple)
    size: int | None = None


@at.frozen(kw_only=True)
class DatasetSpec:
    attrs: dict[str, Any] = at.field(factory=dict)
    coords: dict[str, Variable] = at.field(factory=dict)
    data_vars: dict[str, Variable] = at.field(factory=dict)
    dtypes: Mapping[str, np.dtype] = at.field(factory=dict)
    encoding: dict[str, Any] = at.field(factory=dict)
    dims: dict[str, int] = at.field(factory=dict)


@at.frozen(kw_only=True)
class Spec(Generic[TXarray]):
    @classmethod
    def structure(cls, obj: Mapping) -> Self:
        return converter.structure(obj, cls)

    @classmethod
    def unstructure(cls, obj: TXarray) -> dict:
        unstructure_as = (
            DataArraySpec if isinstance(obj, xr.DataArray) else xr.Dataset
        )
        return converter.unstructure(obj, unstructure_as)

    @classmethod
    def validate(cls, obj: TXarray) -> None:
        cls.structure(cls.unstructure(obj))
        return None
