"""Core."""

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import affine
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterstats
import seaborn as sns

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
        .fillna(0)
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
    gb_reduce_kwargs : mapping, optional
        Additional keyword arguments to pass to the group-by reduce-like method.

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
        return getattr(
            buffers.to_frame(name="geometry")
            .sjoin(gdf)
            # remove right index resulting column in the sjoin data frame
            # see https://github.com/geopandas/geopandas/issues/498
            .drop(columns=["geometry", gdf.index.name], errors="ignore")
            .reset_index(sites.index.name)
            .groupby(by=sites.index.name),
            gb_reduce_method,
        )(*gb_reduce_args, **gb_reduce_kwargs)

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

    return vector_features_df


## raster
def compute_raster_features(
    raster: np.ndarray | rio.io.MemoryFile | utils.PathType,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    *,
    affine: affine.Affine | None = None,
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
    zonal_stats_kwargs : mapping, optional
        Additional keyword arguments to pass to `rasterstats.zonal_stats`.

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

    return _compute_features(
        raster,
        sites,
        buffer_dists,
        _zonal_stats,
        # pass `raster` again as `buffers_to_features_args`
        raster,
        affine=affine,
        **zonal_stats_kwargs,
    )


def _compute_features_df(
    sites,
    data_dict,
    buffer_dists_dict,
    feature_method_dict,
    feature_col_prefix_dict,
    feature_methods_args_dict,
    feature_methods_kwargs_dict,
):
    # small utility
    def _prefix_rename_dict(feature):
        feature_col_prefix = feature_col_prefix_dict.get(feature, "")
        if feature_col_prefix:
            return lambda feature_col: f"{feature_col_prefix}_{feature_col}"
        else:
            return {}

    # ACHTUNG: we need to unstack each `feature_df` individually because each feature
    # may have different scales/buffer distances
    features_df = pd.concat(
        [
            feature_method(
                data_dict[feature_method],
                sites,
                buffer_dists_dict[feature_method],
                *feature_methods_args_dict.get(feature, []),
                **feature_methods_kwargs_dict.get(feature, {}),
            )
            .rename(columns=_prefix_rename_dict(feature_method))
            .unstack(level="buffer_dist")
            for feature, feature_method in feature_method_dict.items()
        ],
        axis="columns",
    )
    features_df.columns = [
        f"{feature_col}_{buffer_dist}"
        for feature_col, buffer_dist in features_df.columns.values
    ]
    return features_df


def _fit_transform(X, transformer, **transformer_kwargs):
    # ACHTUNG: do not modify X in place to avoid side effects
    _X = transformer(**transformer_kwargs).fit_transform(X)
    if isinstance(X, pd.DataFrame):
        _X = pd.DataFrame(_X, index=X.index, columns=X.columns)

    return _X


class FocalAnalysis:
    """Multi-scale feature computer.

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
        self.sites = sites

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
        # self.feature_method_dict = feature_method_dict

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
        data = _process_scalar_sequence_mapping_arg(data)

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
        # self.buffer_dists_dict = buffer_dists

        # process the `feature_col_prefixes` arg
        feature_col_prefixes = _process_scalar_sequence_mapping_arg(
            feature_col_prefixes
        )

        # process the `feature_methods_args_dict` arg
        if feature_methods_args is None:
            feature_methods_args = {}
        if len(features) == 1 and features[0] not in feature_methods_args:
            feature_methods_args = {features[0]: feature_methods_args}
        # self.feature_methods_args = feature_methods_args

        # process the `feature_methods_kwargs` arg
        if feature_methods_kwargs is None:
            feature_methods_kwargs = {}
        if len(features) == 1 and features[0] not in feature_methods_kwargs:
            feature_methods_kwargs = {features[0]: feature_methods_kwargs}
        # self.feature_methods_kwargs = feature_methods_kwargs

        # compute the features
        self.features_df = _compute_features_df(
            sites,
            data,
            buffer_dists,
            feature_method_dict,
            feature_col_prefixes,
            feature_methods_args,
            feature_methods_kwargs,
        )

    def decompose(
        self,
        *,
        decomposer=None,
        preprocessor=None,
        preprocessor_kwargs=None,
        imputer=None,
        imputer_kwargs=None,
        **decomposer_kwargs,
    ):
        """Factorize the spatial signature matrix into components.

        Parameters
        ----------
        decomposer : class, optional
            A class that implements the decomposition algorithm. It can be any
            scikit-learn like transformer that implements the `fit`, `transform` and
            `fit_transform` methods and with the `components_` and `n_components`
            attributes. If no value is provided, the default value set in
            `settings.DEFAULT_DECOMPOSER` will be taken.
        preprocessor : class, optional
            A class that implements the preprocessing algorithm. It can be any
            scikit-learn like transformer that implements the `fit_transform` method.
            If no value is provided, the default value set in
            `settings.DEFAULT_PREPROCESSOR` will be taken.
        preprocessor_kwargs : dict, optional
            Keyword arguments to be passed to the initializationof `preprocessor`.
        imputer : class, optional
            A class that implements the imputation algorithm. It can be any scikit-learn
            like transformer that implements the `fit_transform` method. If no value is
            provided, no imputation will be performed.
        imputer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `imputer`. Ignored
            if `imputer` is `None`.
        **decomposer_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `decomposer`.

        Returns
        -------
        components_df : pandas.DataFrame
            A DataFrame with the components of the decomposition. Each row corresponds
            to a landscape and each column to a component.
        decomposer_model : object
            The fitted decomposer model.
        """
        # ACHTUNG: using a copy to avoid modifying the original features_df
        X = self.features_df.copy()

        if preprocessor is None:
            preprocessor = settings.DEFAULT_PREPROCESSOR

        if preprocessor:  # user can provide `preprocessor=False` to skip this step
            if preprocessor_kwargs is None:
                preprocessor_kwargs = {}
            X = _fit_transform(X, preprocessor, **preprocessor_kwargs)

        if imputer is not None:
            if imputer_kwargs is None:
                imputer_kwargs = {}
            X = _fit_transform(X, imputer, **imputer_kwargs)

        if decomposer is None:
            decomposer = settings.DEFAULT_DECOMPOSER
        try:
            # try if the model accepts nan values
            decompose_model = decomposer(**decomposer_kwargs).fit(X)
        except ValueError:
            warnings.warn(
                "The provided spatial signatures contain NaN values which are not "
                "supported by the decomposition model. In order to proceed, the NaN "
                "values will be dropped. However, you may consider either (i) changing "
                "the chosen metrics or (ii) imputing the NaN values by providing the "
                "`imputer` and `imputer_kwargs` arguments.",
                RuntimeWarning,
            )
            X = X.dropna()
            decompose_model = decomposer(**decomposer_kwargs).fit(X)
        # set X to the reduced matrix but as a data frame with the same index as the
        # original metrics' matrix (taking into account the dropped rows if any)
        return pd.DataFrame(
            decompose_model.transform(X), index=X.index
        ), decompose_model

    def get_loading_df(self, decompose_model, *, columns=None, index=None, **df_kwargs):
        """Get components loadings for each metric.

        Parameters
        ----------
        decompose_model : object
            The decomposition model fitted to the spatial signature matrix.
        columns : list-like, optional
            Column names for the components. If no value is provided, an integer range
            from 0 to `n_components_ - 1` will be used.
        index : list-like, optional
            Index names for the metrics. If no value is provided, the column names of
            `metrics_df` will be used.
        **df_kwargs : dict, optional
            Keyword arguments to be passed to the initialization of `pandas.DataFrame`

        Returns
        -------
        loading_df : pandas.DataFrame
            A DataFrame with the loadings of the components. Each row corresponds to a
            metric and each column to a component.
        """
        if df_kwargs is None:
            _df_kwargs = {}
        else:
            _df_kwargs = df_kwargs.copy()
        if columns is None:
            columns = _df_kwargs.pop(
                "columns",
                range(decompose_model.n_components_),
            )
        if index is None:
            index = _df_kwargs.pop("index", self.features_df.columns)
        return pd.DataFrame(
            decompose_model.components_.T, columns=columns, index=index, **_df_kwargs
        )

    def scatterplot_features(
        self,
        feature_x,
        feature_y,
        *,
        hue=None,
        ax=None,
        **scatterplot_kwargs,
    ):
        """Scatterplot the site samples by two selected features.

        Parameters
        ----------
        feature_x, feature_y : str
            Strings with the names of the metrics to be plotted on the x and y axes
            respectively. Should be a column of `cgram.data`. Note thus that for
            class-level metrics, the passed string should be of the form
            "{metric}_{class_val}".
        ax : matplotlib.axes.Axes, optional
            Axes object to draw the plot onto, otherwise create a new figure.
        scatterplot_kwargs : dict, optional
            Keyword arguments to be passed to `seaborn.scatterplot`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the `Axes` object with the plot drawn onto it.

        """
        if hue in self.sites.columns:
            hue = self.sites.loc[self.features_df.index, hue].values

        if ax is None:
            _, ax = plt.subplots()

        sns.scatterplot(
            x=feature_x,
            y=feature_y,
            data=self.features_df,
            hue=hue,
            **scatterplot_kwargs,
        )

        return ax
