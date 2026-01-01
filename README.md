# xarray-specs

A tiny validation library for xarray.

> [!WARNING]  
> This package is in an early stage of development.
> Frequent and breaking changes are expected.

## Overview

Create specifications of [xarray](https://docs.xarray.dev/en/stable/index.html) objects using [attrs](https://www.attrs.org/en/stable/index.html) classes and type hints. Validate xarray objects against your specs using [cattrs](https://catt.rs/en/stable/index.html).

## Quick start

```python
from typing import Literal

import attrs as at
import numpy as np
import xarray as xr

from xarray_specs import DataArraySpec

da = xr.DataArray(
    data=[1, 2, 3],
    dims=['x'],
    name='yipeee',
)


@at.frozen
class MySpec(DataArraySpec):
    dtype: np.dtype[np.int64]
    dims: tuple[Literal['x']]
    # Use `attrs` validators for constraints you can't express with the type system
    name: str = at.field(validator=at.validators.matches_re(r'^yipee+$'))


MySpec.validate(da)
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
