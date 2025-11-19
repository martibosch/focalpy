"""Urban form."""

from collections.abc import Sequence

import geopandas as gpd
import momepy
import pandas as pd

from focalpy import core, utils


def compute_urban_morphometrics(
    buildings_gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    momepy_metrics: str | Sequence[str],
    *gb_reduce_args: Sequence,
    momepy_metrics_args_dict: utils.KwargsType = None,
    momepy_metrics_kwargs_dict: utils.KwargsType | None = None,
    gb_reduce_method: str = "agg",
    **gb_reduce_kwargs: utils.KwargsType,
) -> pd.DataFrame:
    """
    Compute multi-scale urban morphometrics from `momepy`.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        Building footprints (polygon geometries).
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The buffer distances to compute features, in the same units as the landscape
        raster CRS.
    momepy_metrics : str or list-like of str
        The `momepy` metrics to compute. Can be any momepy measure that takes the
        building geometries as the first argument (see the `momepy` API reference for
        more details on the suitable metrics).
    gb_reduce_args : list-like, optional
        Additional positional arguments to pass to the group-by reduce-like method.
    momepy_metrics_args_dict, momepy_metrics_kwargs_dict : dict, optional
        Dictionaries mapping each metric function name to the respective additional
        positional and keyword arguments.
    gb_reduce_method : str, default "agg"
        The group-by reduce-like method to apply to the data. This can be any method
        available on the `pandas.core.groupby.DataFrameGroupBy` object, e.g.,
        "sum", "mean", "median", "min", "max", or "agg".
    gb_reduce_kwargs : mapping, optional
        Additional keyword arguments to pass to the group-by reduce-like method.

    Returns
    -------
    metrics_df : pandas.DataFrame
        The computed urban morphometrics for each site (first-level index) and buffer
        distance (second-level index).
    """
    # if we only have one momepy metric, put it as a list
    if not pd.api.types.is_list_like(momepy_metrics):
        momepy_metrics = [momepy_metrics]

    # # if we don't have args or kwargs, create empty lists of the same length as
    # # `momepy_metrics`
    # if momepy_metrics_args is None:
    #     momepy_metrics_args = [()] * len(momepy_metrics)
    # if momepy_metrics_kwargs is None:
    #     momepy_metrics_kwargs = [{}] * len(momepy_metrics)

    # def _compute_urban_morphometrics(buffers):
    #     return pd.concat(
    #         [
    #             getattr(momepy, metric)(
    #                 buildings_gdf, *momepy_metric_args, **momepy_metric_kwargs
    #             )
    #             for metric, momepy_metric_args, momepy_metric_kwargs in zip(
    #                 momepy_metrics, momepy_metrics_args, momepy_metrics_kwargs
    #             )
    #         ],
    #         axis="columns",
    #     )

    # if we don't have dictionaries of args or kwargs, create empty ones
    if momepy_metrics_args_dict is None:
        momepy_metrics_args_dict = {}
    if momepy_metrics_kwargs_dict is None:
        momepy_metrics_kwargs_dict = {}

    return core.compute_vector_features(
        gpd.GeoDataFrame(
            pd.concat(
                [
                    getattr(momepy, metric)(
                        buildings_gdf,
                        *momepy_metrics_args_dict.get(metric, []),
                        **momepy_metrics_kwargs_dict.get(metric, {}),
                    ).rename(metric)
                    for metric in momepy_metrics
                ],
                axis="columns",
            ),
            geometry=buildings_gdf.geometry,
        ),
        sites,
        buffer_dists,
        *gb_reduce_args,
        **gb_reduce_kwargs,
    )
