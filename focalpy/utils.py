"""Utilities."""

import os
from collections.abc import Mapping

import numpy as np

########################################################################################
# type annotations
# type hint for path-like objects
PathType = str | os.PathLike
# type hint for keyword arguments
KwargsType = Mapping | None
# type hint for fillna arguments
FillnaType = float | Mapping | bool | None


########################################################################################
# geo utils
def as_coord_arr(site_gser):
    """Convert a GeoSeries of point geometries to a coordinate array.

    Parameters
    ----------
    site_gser : geopandas.GeoSeries or array-like of shape (n, 2)
        Site locations. If a :class:`~geopandas.GeoSeries`, x/y
        coordinates are extracted from the point geometries.

    Returns
    -------
    numpy.ndarray of shape (n, 2)
    """
    if hasattr(site_gser, "x") and hasattr(site_gser, "y"):
        return np.column_stack([site_gser.x, site_gser.y])
    return np.asarray(site_gser)
