"""Core."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import pyregeon
import rasterio as rio
import rasterstats
from sklearn.base import BaseEstimator

import focalpy
from focalpy import settings, utils

__all__ = ["compute_vector_features", "compute_raster_features", "FocalAnalysis"]


# compute methods
def _compute_features(
    data,
    sites: gpd.GeoSeries | gpd.GeoDataFrame,
    buffer_dists: float | Sequence[float],
    buffers_to_features: Callable,
    *buffers_to_features_args: Sequence,
    **buffers_to_features_kwargs: utils.KwargsType,
) -> pd.DataFrame:
    # if we only have one buffer distance, put it as a list
    if not pd.api.types.is_list_like(buffer_dists):
        buffer_dists = [buffer_dists]

    feature_dfs = []
    for buffer_dist in buffer_dists:
        feature_dfs.append(
            buffers_to_features(
                sites.buffer(buffer_dist),
                *buffers_to_features_args,
                **buffers_to_features_kwargs,
            ).assign(**{"buffer_dist": buffer_dist})
        )
    return (
        pd.concat(
            feature_dfs,
            axis="rows",
        )
        .set_index("buffer_dist", append=True)
        .sort_index()
    )


## vector
def compute_vector_features(
    data: gpd.GeoDataFrame | utils.PathType,
    sites: gpd.GeoSeries | gpd.GeoDataFrame,
    buffer_dists: float | Sequence[float],
    *gb_reduce_args: Sequence,
    gb_reduce_method: str = "agg",
    fillna: utils.FillnaType = None,
    **gb_reduce_kwargs: utils.KwargsType,
) -> pd.DataFrame:
    """Compute multi-scale vector aggregation features.

    Parameters
    ----------
    data : geopandas.GeoDataFrame or path-like
        Vector data to compute features.
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The buffer distances to compute features, in the same units as the vector data
        CRS.
    gb_reduce_args : list-like, optional
        Additional positional arguments to pass to the group-by reduce-like method.
    gb_reduce_method : str, default "agg"
        The group-by reduce-like method to apply to the data. This can be any method
        available on the `pandas.core.groupby.DataFrameGroupBy` object, e.g.,
        "sum", "mean", "median", "min", "max", or "agg".
    fillna : numeric, mapping, bool, optional
        Value to use to fill NaN values in the resulting features DataFrame, passed to
        `pandas.DataFrame.fillna`. If `False`, no filling is performed. If `None`, the
        default value set in `settings.VECTOR_FEATURES_FILLNA` is used.
    **gb_reduce_kwargs : mapping, optional
        Keyword arguments to pass to the group-by reduce-like method.

    Returns
    -------
    features_df : pandas.DataFrame
        The computed features for each site (first-level index) and buffer distance
        (second-level index).
    """
    # TODO: do we really need to accept file-like `data` or should we support geo-data
    # frames only?
    if isinstance(data, utils.PathType):
        gdf = gpd.read_file(data)
    else:
        gdf = data

    # ensure that we have an index name (for proper groupby/reset_index operations)
    site_index_name = sites.index.name
    if site_index_name is None:
        site_index_name = "site_id"
        sites = sites.rename_axis(site_index_name)
    # get only the geo series (no data frame columns)
    if isinstance(sites, gpd.GeoDataFrame):
        sites = sites.geometry

    def _gb_reduce(buffers):
        return (
            getattr(
                buffers.to_frame(name="geometry")
                .sjoin(gdf)
                # remove right index resulting column in the sjoin data frame
                # see https://github.com/geopandas/geopandas/issues/498
                .drop(columns=["geometry", gdf.index.name], errors="ignore")
                .reset_index(sites.index.name)
                .groupby(by=sites.index.name),
                gb_reduce_method,
            )(*gb_reduce_args, **gb_reduce_kwargs)
            # ACHTUNG: use `reindex` to ensure that all sites with no overlapping
            # geometries are included too (which will have NaN values that we can
            # subsequently manage with `fillna`)
            .reindex(sites.index)
        )

    if gb_reduce_method != "agg":

        def _gb_reduce_to_frame(buffers):
            return _gb_reduce(buffers).to_frame(name=gb_reduce_method)

        buffers_to_features = _gb_reduce_to_frame
    else:
        buffers_to_features = _gb_reduce

    # compute the features
    vector_features_df = _compute_features(
        data, sites, buffer_dists, buffers_to_features
    )

    # process column names
    if isinstance(vector_features_df.columns, pd.MultiIndex):
        # multi-index columns (feature, agg) to single level by joining with "_"
        vector_features_df.columns = [
            "_".join(map(str, col)).strip() for col in vector_features_df.columns.values
        ]
    elif gb_reduce_args:
        # when positionally passing a dict of strings to agg, add them as suffix
        # note that if a single key of the dict maps to a list-like, we will have
        # multi-index columns so we will not enter this `else` but rather the `if` above
        gb_reduce_func_arg = gb_reduce_args[0]
        if gb_reduce_method == "agg":
            if isinstance(gb_reduce_func_arg, dict):
                vector_features_df = vector_features_df.rename(
                    columns=lambda col: f"{col}_{gb_reduce_func_arg.get(col, '')}"
                )
            else:
                # assume `gb_reduce_func_arg` is a string (otherwise we would have
                # multi-index columns) and add them as suffix to all columns
                vector_features_df = vector_features_df.rename(
                    columns=lambda col: f"{col}_{gb_reduce_func_arg}"
                )

    if fillna is None:
        fillna = settings.VECTOR_FEATURES_FILLNA
    if fillna == 0 or fillna:
        vector_features_df = vector_features_df.fillna(fillna)

    return vector_features_df


## raster
def compute_raster_features(
    raster: np.ndarray | rio.io.MemoryFile | utils.PathType,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    *,
    affine: affine.Affine | None = None,
    fillna: utils.FillnaType = None,
    **zonal_stats_kwargs: utils.KwargsType,
):
    """Compute multi-scale raster statistics features.

    Parameters
    ----------
    raster : numpy.ndarray, rasterio.io.MemoryFile or path-like
        Raster data to compute features, passed to `rasterstats.zonal_stats`. If a
        `numpy.ndarray` is passed, the `affine` keyword argument is required.
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The buffer distances to compute features, in the same units as the raster CRS.
    affine: `affine.Affine`, optional
        Affine transform. Ignored if `raster` is a path-like object.
    fillna : numeric, mapping, bool, optional
        Value to use to fill NaN values in the resulting features DataFrame, passed to
        `pandas.DataFrame.fillna`. If `False`, no filling is performed. If `None`, the
        default value set in `settings.RASTER_FEATURES_FILLNA` is used.
    **zonal_stats_kwargs : mapping, optional
        Keyword arguments to pass to `rasterstats.zonal_stats`.

    Returns
    -------
    features_df : pandas.DataFrame
        The computed features for each site (first-level index) and buffer distance
        (second-level index).
    """

    def _zonal_stats(buffers, *args, **kwargs):
        return pd.DataFrame(
            rasterstats.zonal_stats(buffers, *args, **kwargs), index=buffers.index
        )

    raster_features_df = _compute_features(
        raster,
        sites,
        buffer_dists,
        _zonal_stats,
        # pass `raster` again as `buffers_to_features_args`
        raster,
        affine=affine,
        **zonal_stats_kwargs,
    )

    if fillna is None:
        fillna = settings.RASTER_FEATURES_FILLNA
    if fillna == 0 or fillna:
        raster_features_df = raster_features_df.fillna(fillna)

    return raster_features_df


def _fit_transform(X, transformer, **transformer_kwargs):
    _transformer = transformer(**transformer_kwargs).fit(X)
    # ACHTUNG: do not modify X in place to avoid side effects
    _X = _transformer.transform(X)
    if isinstance(X, pd.DataFrame):
        _X = pd.DataFrame(_X, index=X.index, columns=X.columns)

    return _X, _transformer


class FocalAnalysis:
    """Focal analysis.

    Parameters
    ----------
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    """

    def __init__(
        self,
        data: Any | Sequence | Mapping,
        sites: gpd.GeoSeries | gpd.GeoDataFrame,
        buffer_dists: float | Sequence[float] | Mapping,
        features: str | Callable | Sequence[str | Callable],
        *,
        feature_col_prefixes: str | Sequence[str] | Mapping | None = None,
        feature_methods_args: utils.KwargsType = None,
        feature_methods_kwargs: utils.KwargsType = None,
    ):
        # # ensure that we have an index name
        # site_index_name = sites.index.name
        # if site_index_name is None:
        #     site_index_name = "site_id"
        #     sites = sites.rename_axis(site_index_name)
        if isinstance(sites, gpd.GeoSeries):
            sites = sites.to_frame()
        self.sites_gdf = sites

        # process the `features` arg
        # if we only have one feature, put it as a list
        if not pd.api.types.is_list_like(features):
            features = [features]
        # get a list of feature methods (not names)
        feature_method_dict = {}
        for feature in features:
            if isinstance(feature, str):
                feature_method_dict[feature] = getattr(focalpy, feature)
            else:
                # assert it is a callable
                feature_method_dict[feature] = feature
        self.feature_method_dict = feature_method_dict

        # small utility to dry the processing of some args
        def _process_scalar_sequence_mapping_arg(arg):
            if not pd.api.types.is_list_like(arg) and not isinstance(arg, Mapping):
                # if we only have one scalar item, map it to all feature methods
                arg = {
                    feature_method: arg
                    for feature_method in feature_method_dict.values()
                }
            elif pd.api.types.is_list_like(arg):
                # for a list of scalar items, we will pass it to all feature methods -
                # note that this requires a positional match
                arg = {
                    feature_method_dict[feature]: _arg
                    for feature, _arg in zip(features, arg)
                }
            else:  # if isinstance(arg, Mapping):
                # assume that `arg` is a mapping so there is nothing to do
                pass
            return arg

        # process the `data` arg
        self.data_dict = _process_scalar_sequence_mapping_arg(data)

        # process the `buffer_dists` arg
        # we cannot use `_process_scalar_sequence_mapping_arg` because this is slightly
        # different
        if not pd.api.types.is_list_like(buffer_dists) and not isinstance(
            buffer_dists, Mapping
        ):
            # if we only have one scalar item, put it as a list so that it is properly
            # processed below
            buffer_dists = [buffer_dists]
        if pd.api.types.is_list_like(buffer_dists):
            if np.isscalar(buffer_dists[0]) and np.isreal(buffer_dists[0]):
                # for a list of scalar items, we map it to all feature methods
                # buffer_dists = {feature: buffer_dists for feature in features}
                buffer_dists = {
                    feature_method: buffer_dists
                    for feature_method in feature_method_dict.values()
                }
            else:
                # assume that we have a list of lists, i.e., each feature method has its
                # own buffer distances) - note that this requires a positional match
                buffer_dists = {
                    feature_method_dict[feature]: _buffer_dists
                    for feature, _buffer_dists in zip(features, buffer_dists)
                }
        else:
            # assume that `buffer_dists` is a mapping so there is nothing to do
            pass
        self.buffer_dists_dict = buffer_dists

        # process the `feature_col_prefixes` arg
        self.feature_col_prefix_dict = _process_scalar_sequence_mapping_arg(
            feature_col_prefixes
        )

        # process the `feature_methods_args` arg
        if feature_methods_args is None:
            feature_methods_args = {}
        if len(features) == 1 and features[0] not in feature_methods_args:
            feature_methods_args = {features[0]: feature_methods_args}
        self.feature_methods_args_dict = feature_methods_args

        # process the `feature_methods_kwargs` arg
        if feature_methods_kwargs is None:
            feature_methods_kwargs = {}
        if len(features) == 1 and features[0] not in feature_methods_kwargs:
            feature_methods_kwargs = {features[0]: feature_methods_kwargs}
        self.feature_methods_kwargs_dict = feature_methods_kwargs

        # compute the features for the initialization sites
        self.features_df = self.compute_features_df(
            sites_gdf=self.sites_gdf,
        )

    def compute_features_df(
        self,
        *,
        sites_gdf: gpd.GeoDataFrame | None = None,
        spatial_extent: pyregeon.RegionType | None = None,
        grid_res: float | None = None,
        process_region_arg_kwargs: utils.KwargsType = None,
        **generate_regular_grid_gser_kwargs: utils.KwargsType,
    ) -> pd.DataFrame:
        """Compute a data frame of multi-scale features for the provided sites."""
        if sites_gdf is None:
            if grid_res is None:
                raise ValueError(
                    "When `sites_gdf` is not provided, `grid_res` must be provided."
                )
            if process_region_arg_kwargs is None:
                process_region_arg_kwargs = {}
            # the grid geometry type must be "point"
            _generate_regular_grid_gser_kwargs = (
                generate_regular_grid_gser_kwargs.copy()
            )
            _ = _generate_regular_grid_gser_kwargs.pop("geometry_type", None)
            _generate_regular_grid_gser_kwargs["geometry_type"] = "point"
            sites_gdf = (
                pyregeon.generate_regular_grid_gser(
                    pyregeon.RegionMixin._process_region_arg(
                        spatial_extent, **process_region_arg_kwargs
                    )["geometry"],
                    grid_res,
                    **_generate_regular_grid_gser_kwargs,
                )
                .to_crs(self.sites_gdf.crs)
                .to_frame(name="geometry")
            )

        # small utility
        def _prefix_rename_dict(feature):
            feature_col_prefix = self.feature_col_prefix_dict.get(feature, "")
            if feature_col_prefix:
                return lambda feature_col: f"{feature_col_prefix}_{feature_col}"
            else:
                return {}

        # ACHTUNG: we need to unstack each `feature_df` individually because each
        # feature may have different scales/buffer distances
        features_df = pd.concat(
            [
                feature_method(
                    self.data_dict[feature_method],
                    sites_gdf,
                    self.buffer_dists_dict[feature_method],
                    *self.feature_methods_args_dict.get(feature, []),
                    **self.feature_methods_kwargs_dict.get(feature, {}),
                )
                .rename(columns=_prefix_rename_dict(feature_method))
                .unstack(level="buffer_dist")
                for feature, feature_method in self.feature_method_dict.items()
            ],
            axis="columns",
        )
        features_df.columns = [
            f"{feature_col}_{buffer_dist}"
            for feature_col, buffer_dist in features_df.columns.values
        ]
        return features_df

    def predict(
        self,
        model: BaseEstimator | Callable,
        sites: gpd.GeoDataFrame | gpd.GeoSeries,
        *,
        features: str | Sequence[str] | None = None,
    ) -> pd.Series:
        """Predict using a fitted scikit-learn-like model for the provided sites."""
        if isinstance(sites, gpd.GeoSeries):
            sites = sites.to_frame()
        # TODO: only compute requested features/scales
        # compute features
        X_df = self.compute_features_df(sites_gdf=sites.to_crs(self.sites_gdf.crs))
        # select only requested features
        if features is not None:
            if not pd.api.types.is_list_like(features):
                features = [features]
            X_df = X_df[features]

        return pd.Series(model.predict(X_df), index=X_df.index)

    def predict_raster(
        self,
        model: BaseEstimator | Callable,
        spatial_extent: pyregeon.RegionType,
        grid_res: float,
        *,
        features: str | Sequence[str] = None,
        pred_label: str = "pred",
        process_region_arg_kwargs: utils.KwargsType = None,
        **generate_regular_grid_gser_kwargs: utils.KwargsType,
    ):
        """Predict a raster grid using a fitted scikit-learn-like model."""
        if process_region_arg_kwargs is None:
            process_region_arg_kwargs = {}
        # the grid geometry type must be "point"
        _generate_regular_grid_gser_kwargs = generate_regular_grid_gser_kwargs.copy()
        _ = _generate_regular_grid_gser_kwargs.pop("geometry_type", None)
        _generate_regular_grid_gser_kwargs["geometry_type"] = "point"
        grid_sites_gdf = (
            pyregeon.generate_regular_grid_gser(
                pyregeon.RegionMixin._process_region_arg(
                    spatial_extent, **process_region_arg_kwargs
                )["geometry"],
                grid_res,
                **_generate_regular_grid_gser_kwargs,
            )
            # .to_crs(self.sites_gdf.crs)
            .to_frame(name="geometry")
        )
        pred_gdf = gpd.GeoDataFrame(
            {pred_label: self.predict(model, grid_sites_gdf, features=features)},
            geometry=grid_sites_gdf["geometry"],
        )
        for coord in ["x", "y"]:
            pred_gdf[coord] = getattr(pred_gdf["geometry"], coord)
        return (
            pred_gdf.reset_index(drop=True)
            .set_index(["y", "x"])
            .drop(columns="geometry")
            .to_xarray()[pred_label]
        )
