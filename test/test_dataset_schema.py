from typing import Literal, TypedDict

import attrs as at
import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_specs import DataArraySchema, DatasetSchema, make_converter


@st.composite
def datasets(draw: st.DrawFn):
    sizes = draw(xrst.dimension_sizes(min_dims=1))
    coords = {
        name: draw(xrst.variables(dims=st.just({name: size})))
        for name, size in sizes.items()
    }
    variables = draw(xrst.dimension_names(min_dims=1))
    hp.assume(set(coords.keys()) & set(variables) == set())
    data_vars = {
        name: draw(xrst.variables(dims=st.just(sizes))) for name in variables
    }
    attrs = draw(xrst.attrs())
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


@hp.given(da=datasets())
def test_data_array_structure(da: xr.DataArray) -> None:
    """Should be able to structure a data array schema."""
    converter = make_converter()
    raw = converter.unstructure(da, DatasetSchema)
    converter.structure(raw, DatasetSchema)


@hp.given(da=datasets())
def test_data_array_unstructure(da: xr.DataArray) -> None:
    """Should be able to un-structure a data array."""
    converter = make_converter()
    converter.unstructure(da, DatasetSchema)


def test_schema_composition() -> None:
    # TODO: Replace this with parameterized tests
    converter = make_converter()

    class TemperatureAttrs(TypedDict):
        units: Literal['degK']

    @at.frozen
    class Time(DataArraySchema):
        dims: tuple[str]
        dtype: np.dtype[np.datetime64]

    @at.frozen
    class Temperature(DataArraySchema):
        dtype: np.dtype[np.float64]
        dims: tuple[str]
        name: str = at.field(validator=at.validators.matches_re(r'^temp.+$'))
        attrs: TemperatureAttrs

    class DataVars(TypedDict):
        temperature: Temperature

    class Coords(TypedDict):
        time: Time

    class Attrs(TypedDict):
        source: str

    @at.frozen
    class Climate(DatasetSchema):
        data_vars: DataVars
        coords: Coords
        attrs: Attrs

    ds = xr.Dataset(
        data_vars={
            'temperature': ('time', np.random.rand(10), {'units': 'degK'}),
        },
        coords={
            'time': pd.date_range('2023-01-01', periods=10),
        },
        attrs={
            'source': 'xyz',
        },
    )

    converter.validate(ds, Climate)
