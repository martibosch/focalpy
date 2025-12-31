"""Plotting utils."""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy import typing as npt

from focalpy import utils


def plot_raster_and_gdf(
    raster_da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    column: str | npt.ArrayLike | pd.Series | pd.Index,
    *,
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    alpha: float = 0.7,
    edgecolor: str | tuple[float] = "k",
    da_plot_kwargs: utils.KwargsType = None,
    gdf_plot_kwargs: utils.KwargsType = None,
    add_basemap_kwargs: utils.KwargsType = None,
):
    """Plot a raster data array and a geo-data frame with the same color scale."""
    if ax is None:
        _, ax = plt.subplots()
    if isinstance(column, str):
        column_data = gdf[column]
    else:
        column_data = column
    if vmin is None:
        vmin = min(raster_da.min().item(), column_data.min())
    if vmax is None:
        vmax = max(raster_da.max().item(), column_data.max())
    if da_plot_kwargs is None:
        da_plot_kwargs = {}
    if gdf_plot_kwargs is None:
        gdf_plot_kwargs = {}
    if add_basemap_kwargs is None:
        add_basemap_kwargs = {}
    raster_da.plot(
        ax=ax, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, **da_plot_kwargs
    )
    gdf.plot(
        column_data,
        ax=ax,
        cmap=cmap,
        edgecolor=edgecolor,
        vmin=vmin,
        vmax=vmax,
        **gdf_plot_kwargs,
    )
    cx.add_basemap(ax, crs=gdf.crs, **add_basemap_kwargs)
    return ax
