"""Landscape metrics."""

from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
import pylandstats as pls

from focalpy import core, utils

__all__ = ["compute_landscape_metrics"]


def compute_landscape_metrics(
    landscape_filepath: utils.PathType,
    sites: gpd.GeoSeries,
    buffer_dists: float | Sequence[float],
    *,
    class_metrics: Sequence[str] | None = None,
    landscape_metrics: Sequence[str] | None = None,
    classes: Sequence[int] | None = None,
    class_metrics_fillna: float | None = None,
    class_metrics_kwargs: utils.KwargsType = None,
    landscape_metrics_kwargs: utils.KwargsType = None,
) -> pd.DataFrame:
    """
    Compute mutli-scale landscape metrics from `pylandstats`.

    Parameters
    ----------
    landscape_filepath : path-like
        Path to a raster file with the landscape data, passed as the first positional
        argument to `pylandstats.ZonalAnalysis`.
    sites : geopandas.GeoSeries or geopandas.GeoDataFrame
        Site locations (point geometries) to compute features.
    buffer_dists : list-like of numeric
        The buffer distances to compute features, in the same units as the landscape
        raster CRS.
    class_metrics : list-like of str, optional
        A list-like of strings with the names of the metrics that should be computed. If
        `None`, no class-level metric will be computed. Passed as homonymous keyword
        argument to `pylandstats.SpatialSignatureAnalysis`.
    classes : list-like, optional
        A list-like of ints or strings with the class values that should be considered
        in the context of this analysis case. If `None` and class-level metrics are
        computed, all unique class values will be considered. Ignored if no class-level
        metrics are computed. Passed as homonymous keyword argument to
        `pylandstats.SpatialSignatureAnalysis`.
    class_metrics_fillna : bool, optional
        Whether `NaN` values representing landscapes with no occurrences of patches of
        the provided class should be replaced by zero when appropriate, e.g., area and
        edge metrics (no occurrences mean zero area/edge). If the provided value is
        `None` (default), the value will be taken from
        `pylandstats.settings.CLASS_METRICS_DF_FILLNA`. Passed as homonymous keyword
        argument to `pylandstats.SpatialSignatureAnalysis`.
    class_metrics_kwargs, landscape_metrics_kwargs : mapping, optional
        Dictionary mapping the keyword arguments (values) that should be passed to each
        metric method (key) for the class and landscape-level metrics respectively. For
        instance, to exclude the boundary from the computation of `total_edge`,
        metric_kwargs should map the string 'total_edge' (method name) to
        {'count_boundary': False}. If `None`, each metric will be computed according to
        FRAGSTATS defaults.  Passed as homonymous keyword argument to
        `pylandstats.SpatialSignatureAnalysis`.

    Returns
    -------
    features_df : pandas.DataFrame
        The computed landscape metrics for each site (first-level index) and buffer
        distance (second-level index).
    """

    def _compute_landscape_metrics(buffers):
        return pls.SpatialSignatureAnalysis(
            pls.ZonalAnalysis(
                landscape_filepath,
                buffers,
            ),
            class_metrics=class_metrics,
            landscape_metrics=landscape_metrics,
            classes=classes,
            class_metrics_fillna=class_metrics_fillna,
            class_metrics_kwargs=class_metrics_kwargs,
            landscape_metrics_kwargs=landscape_metrics_kwargs,
        ).metrics_df.fillna(0)

    return core._compute_features(
        landscape_filepath, sites, buffer_dists, _compute_landscape_metrics
    )
