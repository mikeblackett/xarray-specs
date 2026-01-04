import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt

from xarray_specs.converter import make_converter


@st.composite
def primitives(draw: st.DrawFn):
    return draw(
        st.one_of(st.text(), st.integers(), st.floats(), st.booleans())
    )


@hp.given(value=primitives())
def test_no_primitive_value_coercion(value: int | float | str | bool) -> None:
    """Should not coerce primitive values."""
    converter = make_converter()
    result = converter.structure(value, type(value))
    assert type(result) is type(value)


@hp.given(data=st.data())
def test_invalid_primitive_structure_fails(data: st.DataObject) -> None:
    """Should raise an exception for incompatible primitive values."""
    converter = make_converter()
    value = data.draw(primitives())
    typ = data.draw(st.sampled_from([int, float, str, bool]))
    hp.assume(type(value) is not typ)
    hp.assume(not (type(value) is bool and typ is int))
    with pt.raises(TypeError):
        converter.structure(value, typ)
