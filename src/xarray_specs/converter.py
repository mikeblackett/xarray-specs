from __future__ import annotations

from typing import Any, get_args, get_origin

import attrs as at
import cattrs as cat
import numpy as np


def _is_generic_dtype(typ: type) -> bool:
    return get_origin(typ) is np.dtype


def _structure_generic_dtype(value: Any, typ: np.dtype) -> np.dtype:
    scalar, *_ = get_args(typ)
    dtype = np.dtype(value)
    if dtype != scalar:
        raise TypeError(f'expected {np.dtype(scalar)!r}, got {dtype!r}.')
    return dtype


def _unstructure_dtype(value: np.dtype) -> str:
    return value.name


def _validate_type[T: type](value: Any, type_of: T) -> T:
    if not isinstance(value, type_of):
        raise TypeError(f'{value!r} not an instance of {type_of}')
    return value


converter = cat.Converter()
for typ in (int, float, str, bytes):
    converter.register_structure_hook(typ, _validate_type)
converter.register_structure_hook(np.dtype, lambda d, _: np.dtype(d))
converter.register_structure_hook_func(
    _is_generic_dtype, _structure_generic_dtype
)
converter.register_unstructure_hook_func(_is_generic_dtype, _unstructure_dtype)
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
