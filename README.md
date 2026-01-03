# xarray-specs

A tiny validation library for xarray.

> [!WARNING]  
> This package is in an early stage of development.
> Frequent and breaking changes are expected.

## Overview

Create specifications of [xarray](https://docs.xarray.dev/en/stable/index.html) objects using [attrs](https://www.attrs.org/en/stable/index.html) classes and type hints. Validate xarray objects against your specs using [cattrs](https://catt.rs/en/stable/index.html).

- covariant field refinement
- type narrowing

## Quick start

```python
from typing import Literal, TypedDict

import attrs as at
import numpy as np
import pandas as pd
import xarray as xr

from xarray_specs import DataArraySchema, make_converter

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


@at.frozen
class Time(DataArraySchema):
    dtype: np.dtype[np.datetime64]


# Use `TypedDict` to model `DataArray.coords` and `Dataset.data_vars` etc
class Coords(TypedDict):
    time: Time


# Use `TypedDict` to model `DataArray.attrs` and `Dataset.attrs`
class Attrs(TypedDict):
    units: Literal['degK']
    precision: np.int32


@at.frozen
class Temperature(DataArraySchema):
    dtype: np.dtype[np.float64]
    dims: tuple[str]
    # Use `attrs` validators for constraints you can't express with the type system
    name: str = at.field(validator=at.validators.matches_re(r'^temp.+$'))
    attrs: Attrs


validator = make_converter()
validator.validate(da, Temperature)
```

## Installation

`xarray-specs` is still in development.

You can install it from source:

```bash
pip install git+https://github.com/mikeblackett/xarray-specs
```

or clone it for development:

```bash
git clone https://github.com/mikeblackett/xarray-specs
uv sync --dev
```
