"""Terrain attributes."""

from collections.abc import Callable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xdem
from rasterio import mask
from tqdm.auto import tqdm

from focalpy import core, settings, utils

__all__ = ["compute_terrain_attributes", "compute_topographic_position_index"]


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
    **terrain_attributes_kwargs : mapping, optional
        Keyword arguments to pass to `xdem.DEM.get_terrain_attribute`.

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
            buffer_attributes = []
            for buffer_dem_arr, buffer_transform in tqdm(
                [mask.mask(src, [buffer_geom], crop=True) for buffer_geom in buffers],
                total=len(buffers),
            ):
                buffer_attributes.append(
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
                )

            return pd.DataFrame(
                buffer_attributes,
                index=buffers.index,
            )

        # ACHTUNG: using `.rename(columns=str.lower)` because when `stats` is None, xdem
        # returns all stats capitalized
        return core._compute_features(
            dem_filepath, sites, buffer_dists, _compute_terrain_attributes
        ).rename(columns=str.lower)


def compute_topographic_position_index(
    dem_filepath: utils.PathType,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    *,
    fillna: utils.FillnaType = None,
    **zonal_stats_kwargs: utils.KwargsType,
) -> pd.DataFrame:
    """
    Compute multi-scale topographic position index (TPI).

    Following Weiss (2001), the TPI at a given scale is the difference between the
    elevation at the site and the mean elevation over a circular neighbourhood of
    radius `buffer_dists` centered on it. Positive values denote local convexities
    (crests, ridges, spurs) and negative values local concavities (valleys, hollows),
    independently of the absolute elevation.

    ACHTUNG: this is the scale-aware TPI, i.e., the buffer distance sets the
    neighbourhood that defines the attribute. This is NOT what
    `compute_terrain_attributes` returns for the "topographic_position_index" attribute:
    there, the per-cell TPI is computed by `xdem` over a `window_size` cell window
    (3 by default) and the buffer only sets the extent over which such per-cell values
    are then aggregated. Since the per-cell TPI is a discrete Laplacian, it nearly
    cancels out when averaged over a large buffer, yielding values orders of magnitude
    below the vertical accuracy of the DEM.

    Parameters
    ----------
    dem_filepath : path-like
        Path to a raster file with the DEM data.
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The neighbourhood radii to compute the TPI, in the same units as the DEM CRS.
    fillna : numeric, mapping, bool, optional
        Value to use to fill NaN values in the resulting features DataFrame, passed to
        `pandas.DataFrame.fillna`. If `False`, no filling is performed. If `None`, the
        default value set in `settings.RASTER_FEATURES_FILLNA` is used.
    **zonal_stats_kwargs : mapping, optional
        Keyword arguments to pass to `rasterstats.zonal_stats` when computing the
        neighbourhood mean elevation.

    Returns
    -------
    tpi_df : pandas.DataFrame
        The computed TPI for each site (first-level index) and neighbourhood radius
        (second-level index).
    """
    # ensure that we have a geo-series
    if isinstance(sites, gpd.GeoDataFrame):
        sites = sites.geometry

    # elevation at the site: a single value per site, broadcast over the radii below.
    # ACHTUNG: do not fill NaN here, otherwise a no-data site would silently become a
    # 0 m elevation and produce a spurious (large, negative) TPI
    site_elevation_ser = core.compute_raster_value_at_site(
        dem_filepath, sites, feature_names="elevation", fillna=False
    )["elevation"]

    # mean elevation over the circular neighbourhood of each radius
    neighborhood_elevation_ser = core.compute_raster_features(
        dem_filepath,
        sites,
        buffer_dists,
        stats="mean",
        fillna=False,
        **zonal_stats_kwargs,
    )["mean"]

    tpi_df = (
        neighborhood_elevation_ser.rsub(
            site_elevation_ser.reindex(
                neighborhood_elevation_ser.index, level=sites.index.name or 0
            )
        )
        .rename("topographic_position_index")
        .to_frame()
    )

    if fillna is None:
        fillna = settings.RASTER_FEATURES_FILLNA
    if fillna == 0 or fillna:
        tpi_df = tpi_df.fillna(fillna)

    return tpi_df
