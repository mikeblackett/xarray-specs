from __future__ import annotations

from typing import Annotated, get_args, get_origin

import attrs as at
import cattrs as cat
import numpy as np


def is_annotated_dtype(typ):
    return get_origin(typ) is Annotated and get_args(typ)[0] is np.dtype


def structure_annotated_dtype(value, typ):
    _, *meta = get_args(typ)
    actual = np.dtype(value)
    for constraint in meta:
        expected = np.dtype(constraint)
        if actual == expected:
            return actual
    raise ValueError(f'{actual} is not one of {[np.dtype(c) for c in meta]}.')


converter = cat.Converter()
converter.register_unstructure_hook(np.dtype, lambda v: str(np.dtype(v)))
converter.register_structure_hook_func(
    is_annotated_dtype, structure_annotated_dtype
)
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
