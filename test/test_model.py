import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt  # noqa: F401
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_specs import DataArraySchema, make_converter

converter = make_converter()

# TODO: Need tests!


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
def test_data_array_unstructure(da: xr.DataArray) -> None:
    """Should be able to un-structure a data array to a schema without error"""
    converter.unstructure(da, DataArraySchema)


@hp.given(da=data_arrays())
def test_data_array_structure(da: xr.DataArray) -> None:
    """Should be able to structure a data array schema without error"""
    raw = converter.unstructure(da, DataArraySchema)
    print(raw)
    converter.structure(raw, DataArraySchema)
