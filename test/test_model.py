import hypothesis as hp
import hypothesis.strategies as st
import pytest as pt  # noqa: F401
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_specs import DataArraySpec, converter


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
    """Should be able to un-structure a data array to a spec without error"""
    converter.unstructure(da, DataArraySpec)
