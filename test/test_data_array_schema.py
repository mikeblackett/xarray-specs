from typing import Literal, TypedDict

import attrs as at
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_specs import DataArraySchema, make_converter


@st.composite
def data_arrays(draw: st.DrawFn):
    sizes = draw(xrst.dimension_sizes())
    data = draw(xrst.variables(dims=st.just(sizes)))
    coords = {
        key: draw(xrst.variables(dims=st.just({key: value})))
        for key, value in sizes.items()
    }
    name = draw(xrst.names())
    return xr.DataArray(data=data, coords=coords, name=name)


@hp.given(da=data_arrays())
def test_data_array_structure(da: xr.DataArray) -> None:
    """Should be able to structure a data array schema."""
    converter = make_converter()
    raw = converter.unstructure(da, DataArraySchema)
    converter.structure(raw, DataArraySchema)


@hp.given(da=data_arrays())
def test_data_array_unstructure(da: xr.DataArray) -> None:
    """Should be able to un-structure a data array."""
    converter = make_converter()
    converter.unstructure(da, DataArraySchema)


def test_schema_composition() -> None:
    # TODO: Replace this with parameterized tests
    converter = make_converter()

    @at.frozen
    class Time(DataArraySchema):
        dims: tuple[str]
        dtype: np.dtype[np.datetime64]

    class Coords(TypedDict):
        time: Time

    class Attrs(TypedDict):
        units: Literal['degK']
        precision: np.int32

    @at.frozen
    class Temperature(DataArraySchema):
        dtype: np.dtype[np.float64]
        dims: tuple[str]
        name: str = at.field(validator=at.validators.matches_re(r'^temp.+$'))
        attrs: Attrs

    da = xr.DataArray(
        data=np.random.rand(10),
        dims=['time'],
        name='temperature',
        coords={
            'time': pd.date_range('2023-01-01', periods=10),
        },
        attrs={
            'units': 'degK',
            'precision': np.int32(2),
        },
    )

    converter.validate(da, Temperature)
