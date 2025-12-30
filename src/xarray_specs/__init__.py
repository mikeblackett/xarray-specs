from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Generic, Self, TypeVar, get_args, get_origin

import attrs as at
import cattrs as cat
import numpy as np
import xarray as xr

TXarray = TypeVar("TXarray", xr.DataArray, xr.Dataset)


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
        unstructure_as = DataArraySpec if isinstance(obj, xr.DataArray) else xr.Dataset
        return converter.unstructure(obj, unstructure_as)

    @classmethod
    def validate(cls, obj: TXarray) -> None:
        cls.structure(cls.unstructure(obj))
        return None


def is_annotated_dtype(typ):
    return get_origin(typ) is Annotated and get_args(typ)[0] is np.dtype


def structure_annotated_dtype(value, typ):
    _, *meta = get_args(typ)
    actual = np.dtype(value)
    for constraint in meta:
        expected = np.dtype(constraint)
        if actual == expected:
            return actual
    raise ValueError(f"{actual} is not one of {[np.dtype(c) for c in meta]}.")


converter = cat.Converter()
converter.register_unstructure_hook(np.dtype, lambda v: str(np.dtype(v)))
converter.register_structure_hook_func(is_annotated_dtype, structure_annotated_dtype)
converter.register_structure_hook(np.dtype, lambda v, _: np.dtype(v))
converter.register_unstructure_hook_factory(
    at.has,
    lambda cls: cat.gen.make_dict_unstructure_fn(
        cls, converter, _cattrs_omit_if_default=False, _cattrs_use_alias=True
    ),
)
converter.register_structure_hook_factory(
    at.has,
    lambda cls: cat.gen.make_dict_structure_fn(
        cls,
        converter,
        _cattrs_forbid_extra_keys=False,
        _cattrs_prefer_attrib_converters=True,
        _cattrs_use_alias=True,
    ),
)
