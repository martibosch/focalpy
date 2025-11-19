"""Terrain attributes."""

from collections.abc import Callable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xdem
from rasterio import mask

from focalpy import core, utils

__all__ = ["compute_terrain_attributes"]


def compute_terrain_attributes(
    # dem: np.ndarray | rio.io.MemoryFile | utils.PathType,
    dem_filepath: utils.PathType,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    terrain_attributes: str | Sequence[str],
    *,
    stats: str | Callable | Sequence[str] | Sequence[Callable] | None = None,
    **terrain_attributes_kwargs: utils.KwargsType,
) -> pd.DataFrame:
    """
    Compute mutli-scale terrain attributes from `xdem`.

    Parameters
    ----------
    dem_filepath : path-like
        Path to a raster file with the DEM data.
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The buffer distances to compute features, in the same units as the landscape
        raster CRS.
    terrain_attributes : list-like of str
       The terrain attributes to compute. Can be any terrain attribute available in
        `xdem.DEM.get_terrain_attribute`.
    stats : str, callable, list-like of str or list-like of callable, optional
        The statistics to compute on the terrain attributes. Can be any statistic
        available in `geoutils.Raster.get_stats` or a custom callable function. If
        `None`, all available statistics from `geoutils` are computed.
    terrain_attributes_kwargs : mapping, optional
        Additional keyword arguments to pass to `xdem.DEM.get_terrain_attribute`.

    Returns
    -------
    features_df : pandas.DataFrame
        The computed terrain attributes for each site (first-level index) and buffer
        distance (second-level index).
    """
    # ensure that dem is an xdem.DEM
    # if isinstance(dem, np.ndarray):
    #     from_array_kwargs = {}
    #     crs = None
    #     dem = xdem.DEM.from_array(dem, affine, crs, **from_array_kwargs)
    # dem = xdem.DEM(dem)

    # return core.compute_raster_features(
    #     dem.data,
    #     sites,
    #     buffer_dists,
    #     affine=dem.transform,
    #     stats=[],
    #     add_stats={
    #         terrain_attribute: getattr(xdem, terrain_attribute)
    #         for terrain_attribute in terrain_attributes
    #     },
    # )

    # if we only have one terrain attribute, put it as a list
    if not pd.api.types.is_list_like(terrain_attributes):
        terrain_attributes = [terrain_attributes]

    if stats is not None:
        # if we only have one stat, put it in as a list
        if not pd.api.types.is_list_like(stats):
            stats = [stats]

    # process results differently if we compute one or multiple terrain attributes
    if len(terrain_attributes) == 1:
        # for one terrain attribute only, just add its name prefix to the dictionary key
        def _get_stats(result, terrain_attibutes):
            return {
                f"{terrain_attibutes[0]}_{stat}": value
                for stat, value in result.get_stats(stats_name=stats).items()
            }
    else:
        # for multiple terrain attributes, flatten the resulting nested dictionary
        def _get_stats(result, terrain_attributes):
            return {
                f"{terrain_attribute}_{stat}": value
                for terrain_attribute, _result in zip(terrain_attributes, result)
                for stat, value in _result.get_stats(stats_name=stats).items()
            }

    with rio.open(dem_filepath) as src:

        def _compute_terrain_attributes(buffers):
            return pd.DataFrame(
                [
                    _get_stats(
                        xdem.DEM.from_array(
                            np.where(
                                buffer_dem_arr != src.nodata, buffer_dem_arr, np.nan
                            ),
                            buffer_transform,
                            src.crs,
                            nodata=src.nodata,
                        ).get_terrain_attribute(
                            terrain_attributes, **terrain_attributes_kwargs
                        ),
                        terrain_attributes,
                    )
                    for buffer_dem_arr, buffer_transform in [
                        mask.mask(src, [buffer_geom], crop=True)
                        for buffer_geom in buffers
                    ]
                ],
                index=buffers.index,
            )

        # ACHTUNG: using `.rename(columns=str.lower)` because when `stats` is None, xdem
        # returns all stats capitalized
        return core._compute_features(
            dem_filepath, sites, buffer_dists, _compute_terrain_attributes
        ).rename(columns=str.lower)
