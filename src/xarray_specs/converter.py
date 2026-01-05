from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args, get_origin, is_typeddict

import attrs as at
import cattrs as cat
import numpy as np
import xarray as xr
import xarray.core.utils as xr_utils

from xarray_specs.schema import DatasetSchema, Schema, VariableSchema

__all__ = ['make_converter']


class XarrayConverter(cat.Converter):
    def validate(
        self, obj: Any, unstructure_as: type[Schema], **kwargs: Any
    ) -> None:
        raw = self.unstructure(obj, unstructure_as)
        self.structure(raw, unstructure_as)


def make_converter(*args, **kwargs) -> XarrayConverter:
    converter = XarrayConverter(*args, **kwargs)
    configure_converter(converter)
    return converter


def configure_converter(converter: cat.Converter) -> cat.Converter:
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
        # TODO: Support structured dtypes
        return value.str

    @converter.register_structure_hook
    def structure_dtype(value: Any, typ: type[np.dtype]) -> np.dtype:
        # TODO: Add custom exception message
        return typ(value)

    @converter.register_unstructure_hook_factory(_is_generic_dtype)
    def unstructure_dtype_factory(
        cls: Any, converter: cat.Converter
    ) -> Callable:
        handler = converter.get_unstructure_hook(np.dtype)

        def unstructure_generic_dtype(value: np.dtype) -> str:
            return handler(value)

        return unstructure_generic_dtype

    def structure_dataset_dims(
        value: xr_utils.FrozenMappingWarningOnValuesAccess, typ: type
    ) -> list[str]:
        return list(value.keys())

    converter.register_structure_hook_func(
        lambda cls: issubclass(
            cls, xr_utils.FrozenMappingWarningOnValuesAccess
        ),
        structure_dataset_dims,
    )

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
    def unstructure_numpy_array(value: np.ndarray) -> list:
        return value.tolist()

    @converter.register_unstructure_hook
    def unstructure_data_array(value: xr.DataArray) -> dict:
        # Unstructure to VariableSchema, not DataArraySchema.
        # This hook is only for nested data arrays (i.e. coordinates/data variables)
        # which should not have fields `name` (it's the mapping key) or `coords`
        # (we don't want to recurse into the coords of coords...)`.
        return converter.unstructure(value, VariableSchema)

    @converter.register_unstructure_hook
    def unstructure_dataset(value: xr.Dataset) -> dict:
        return converter.unstructure(value, DatasetSchema)

    @converter.register_unstructure_hook_factory(is_typeddict)
    def unstructure_coordinates_factory(cls: Any, converter: cat.Converter):
        def unstructure_coordinates(obj):
            result = {}
            for key, schema in cls.__annotations__.items():
                if key in obj:
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
