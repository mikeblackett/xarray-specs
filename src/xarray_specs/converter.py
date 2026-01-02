from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args, get_origin, is_typeddict

import attrs as at
import cattrs as cat
import numpy as np
import xarray as xr

from xarray_specs.schema import VariableSchema

__all__ = ['make_converter']


def make_converter(converter: cat.Converter) -> cat.Converter:
    for typ in (int, float, str, bytes):
        converter.register_structure_hook(typ, _validate_type)

    @converter.register_unstructure_hook
    def unstructure_scalar(value: np.generic) -> str | int | float | complex:
        return value.item()

    @converter.register_structure_hook
    def structure_scalar(value: Any, typ: type[np.generic]) -> np.generic:
        return typ(value)

    @converter.register_unstructure_hook
    def unstructure_dtype(value: np.dtype) -> str:
        return value.name

    @converter.register_structure_hook
    def structure_dtype(value: Any, typ: type[np.dtype]) -> np.dtype:
        return typ(value)

    @converter.register_unstructure_hook_factory(_is_generic_dtype)
    def unstructure_dtype_factory(
        cls: Any, converter: cat.Converter
    ) -> Callable:
        handler = converter.get_unstructure_hook(np.dtype)

        def unstructure_generic_dtype(value: np.dtype) -> str:
            return handler(value)

        return unstructure_generic_dtype

    @converter.register_structure_hook_factory(_is_generic_dtype)
    def structure_dtype_factory(
        cls: Any, converter: cat.Converter
    ) -> Callable:
        scalar = get_args(cls)[0]
        handler = converter.get_structure_hook(np.dtype)

        def structure_generic_dtype(
            value: Any, typ: type[np.dtype]
        ) -> np.dtype:
            dtype = handler(value, typ)
            if dtype.type != scalar:
                raise TypeError(
                    f'expected {np.dtype(scalar)!r}, got {dtype!r}.'
                )
            return dtype

        return structure_generic_dtype

    @converter.register_unstructure_hook
    def unstructure_data_array(value: xr.DataArray) -> dict:
        # Here we unstructure to VariableSchema, not DataArraySchema.
        # This hook is only for coordinates/data variables which should not
        # have `name` or `coords` fields`.
        return converter.unstructure(value, VariableSchema)

    @converter.register_unstructure_hook_factory(is_typeddict)
    def unstructure_coordinates_factory(cls: Any, converter: cat.Converter):
        def unstructure_coordinates(obj):
            result = {}
            for key, schema in cls.__annotations__.items():
                if key not in obj:
                    raise KeyError(f'unknown item {key}')
                result[key] = converter.unstructure(obj[key], schema)
            return result

        return unstructure_coordinates

    converter.register_unstructure_hook_factory(
        at.has,
        lambda cls: cat.gen.make_dict_unstructure_fn(
            cls, converter, _cattrs_use_alias=True
        ),
    )
    converter.register_structure_hook_factory(
        at.has,
        lambda cls: cat.gen.make_dict_structure_fn(
            cls,
            converter,
            _cattrs_forbid_extra_keys=False,
            # Allow user specified converters to take precedence
            _cattrs_prefer_attrib_converters=True,
            _cattrs_use_alias=True,
        ),
    )

    return converter


def _is_generic_dtype(typ: type) -> bool:
    return get_origin(typ) is np.dtype


def _validate_type[T: type](value: Any, typ: T) -> T:
    if not isinstance(value, typ):
        raise TypeError(f'{value!r} not an instance of {typ}')
    return value
