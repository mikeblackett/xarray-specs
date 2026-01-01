from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import attrs as at
import numpy as np
import xarray as xr

from xarray_specs.converter import converter

type Attrs = Mapping[str, Any]
type Dims = Sequence[str]
type Coords = Mapping[str, DataArraySpec]
type DataVars = Mapping[str, DataArraySpec]
type DTypes = Mapping[str, np.generic]
type Encoding = Mapping[str, Any]
type Shape = Sequence[int]
type Sizes = Mapping[str, int]


@at.frozen(kw_only=True)
class Spec[T: (xr.DataArray, xr.Dataset)]:
    """A base class for xarray specifications."""

    @classmethod
    def structure(cls, obj: Mapping) -> Self:
        """Return an new instance of this class from a mapping of attributes to simple Python types.

        Parameters
        ----------
        obj : Mapping
            A mapping of simple Python types.

        Returns
        -------
        Self
           An new instance of this class.
        """
        return converter.structure(obj, cls)

    @classmethod
    def unstructure(cls, obj: T) -> dict:
        """Convert an object to a dictionary of attributes to simple Python types.

        Parameters
        ----------
        obj : T
           The object to convert.

        Returns
        -------
        dict
            A dictionary of attributes to simple Python types.
        """
        return converter.unstructure(obj, cls)

    @classmethod
    def validate(cls, obj: T) -> None:
        """Validate the structure of an xarray object against this specification.

        Parameters
        ----------
        obj : T
           The xarray object to validate.

        Returns
        -------
        None

        Raises
        ------
        ExceptionGroup
           If the validation fails.
        """
        cls.structure(cls.unstructure(obj))
        return None


@at.frozen(kw_only=True)
class DataArraySpec(Spec[xr.DataArray]):
    """Base specification for xarray `DataArray` objects.

    Parameters
    ----------
    attrs : Attrs | None, optional
        Mapping of expected attributes.
    dims : Dims | None, optional
        Sequence of expected dimensions.
    dtype : DType | None, optional
        Expected data type.
    coords : Coords | None, optional
        Mapping of expected coordinates.
    encoding : Encoding | None, optional
        Mapping of expected encoding settings.
    shape : Shape | None, optional
        Expected array shape.
    size: int | None, optional
       Expected array size.
    name : str | None, optional
       Expected name of the array.

    Attributes
    ----------
    attrs : Attrs | None, optional
    dims : Dims | None, optional
    dtype : DType | None, optional
    coords : Coords | None, optional
    encoding : Encoding | None, optional
    shape : Shape | None, optional
    size: int | None, optional
    name : str | None, optional
    """

    attrs: Attrs | None
    dims: Dims | None
    dtype: np.generic | None
    coords: Coords | None
    encoding: Encoding | None
    shape: Shape | None
    size: int | None
    name: str | None


@at.frozen(kw_only=True)
class DatasetSpec(Spec[xr.Dataset]):
    """Base specification for xarray `Dataset` objects."""

    attrs: Attrs | None
    coords: Coords | None
    data_vars: DataVars | None
    dtypes: DTypes | None
    encoding: Encoding | None
    dims: Sizes | None
