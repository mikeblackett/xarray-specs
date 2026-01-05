from collections.abc import Hashable

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


@hp.given(value=primitives())
def test_hashable_allows_primitive_value_passthrough(
    value: int | float | str | bool,
) -> None:
    """Structuring a primitive value to a Hashable should pass through the value."""
    converter = make_converter()
    result = converter.structure(value, Hashable)
    assert type(result) is type(value)


def test_invalid_hashable_fails() -> None:
    """Structuring a non-hashable value to Hashable should raise an exception."""
    converter = make_converter()
    with pt.raises(TypeError, match='expected a hashable'):
        converter.structure({}, Hashable)
