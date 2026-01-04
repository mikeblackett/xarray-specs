import re

import hypothesis as hp
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest as pt

from xarray_specs import make_converter


@hp.given(dtype=npst.scalar_dtypes(), data=st.data())
def test_structure_numpy_dtype(dtype: np.dtype, data: st.DataObject) -> None:
    """Should structure a numpy dtype from a Python string."""
    converter = make_converter()
    value = data.draw(st.sampled_from([dtype.str, dtype.name, str(dtype)]))
    assert np.dtype(value) == converter.structure(value, np.dtype[dtype.type])
    assert np.dtype(value) == converter.structure(value, np.dtype)


@hp.given(value=st.one_of(st.text(min_size=4), st.integers(), st.floats()))
def test_structure_invalid_numpy_dtype_failure(value: object) -> None:
    """Should fail to structure an invalid dtype."""
    converter = make_converter()
    with pt.raises(Exception):
        # TODO: Add exception message matching
        converter.structure(value, np.dtype)


@hp.given(data=st.data())
def test_structure_incompatible_numpy_dtype_failure(
    data: st.DataObject,
) -> None:
    """Should fail to structure an incompatible dtype."""
    converter = make_converter()
    expected_scalar = data.draw(npst.scalar_dtypes()).type
    dtype = data.draw(npst.scalar_dtypes())
    hp.assume(dtype.type != expected_scalar)
    value = dtype.str
    with pt.raises(
        Exception,
        match=re.escape(
            f'expected {np.dtype(expected_scalar)!r}, got {dtype!r}.'
        ),
    ):
        converter.structure(value, np.dtype[expected_scalar])


@hp.given(value=npst.scalar_dtypes())
def test_unstructure_numpy_dtype(value: np.dtype) -> None:
    """Should unstructure a numpy dtype to a Python string."""
    converter = make_converter()
    assert value.str == converter.unstructure(value, np.dtype[value.type])
    assert value.str == converter.unstructure(value, np.dtype)
