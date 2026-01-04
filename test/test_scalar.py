import hypothesis as hp
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

from xarray_specs import make_converter


@st.composite
def scalars(draw: st.DrawFn) -> np.generic:
    """Generate numpy scalar types"""
    dtype = draw(npst.scalar_dtypes())
    scalar = draw(npst.from_dtype(dtype, allow_nan=False))
    return scalar


@hp.given(value=scalars())
def test_structure_numpy_scalar(self, value: np.generic) -> None:
    """Should structure a numpy scalar from a Python primitive."""
    # TODO: This test currently fails for datetime types.
    #   Need to work out how to handle datetime date units.
    converter = make_converter()
    assert value == converter.structure(value.item(), type(value))


# TODO: With the current implementation, it is not possible to validate the conversion of a primitive to a specific scalar. The primitive has no notion of type and many primitives
# can be converted to multiple types. for example integers to floats and vice versa, booleans to timedeltas etc.


@hp.given(value=scalars())
def test_unstructure_numpy_scalar(value: np.generic) -> None:
    """Should unstructure a numpy scalar to a Python primitive."""
    converter = make_converter()
    assert value.item() == converter.unstructure(value)
