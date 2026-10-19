"""Spatial diagnostics for regression residuals."""

import geopandas as gpd
import numpy as np
from esda import moran
from libpysal import weights
from scipy import spatial
from scipy.sparse import csgraph
from sklearn.preprocessing import StandardScaler

from focalpy import utils

__all__ = ["feature_space_coverage", "lisa_residuals", "moran_residuals"]


def _digitize_multidim(X, bin_edges):
    """Return per-row multidimensional bin indices."""
    bin_idx = np.empty(X.shape, dtype=np.int64)

    for j, edges in enumerate(bin_edges):
        idx = np.searchsorted(edges, X[:, j], side="right") - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        bin_idx[:, j] = idx

    return bin_idx


def feature_space_coverage(X_sample, X_reference, *, bins="auto"):
    """Proportion of occupied reference bins also occupied by the sample.

    Features are standardized using the reference statistics, then a
    per-dimension histogram is built from the reference.  A bin is
    "occupied" if it contains at least one reference point.  Coverage is
    the fraction of those occupied bins that also contain at least one
    sample point.

    Bin edges are fixed from the reference distribution (using
    ``numpy.histogram_bin_edges`` with ``bins`` applied independently per
    dimension), so the metric is monotonically non-decreasing as the sample
    grows: adding any station can only fill more bins, never un-fill them.

    Parameters
    ----------
    X_sample : array-like of shape (n_sample, n_features)
        Feature matrix of the sample (e.g., station locations).
    X_reference : array-like of shape (n_reference, n_features)
        Feature matrix of the reference domain (e.g., prediction grid).
    bins : int or str, default "auto"
        Passed to :func:`numpy.histogram_bin_edges` for each feature
        dimension independently.  ``"auto"`` uses the maximum of the
        Sturges and Freedman–Diaconis estimators.

    Returns
    -------
    dict
        ``{"score": float}`` — fraction of occupied reference bins that
        contain at least one sample point.
    """
    X_sample = np.asarray(X_sample, dtype=float)
    X_reference = np.asarray(X_reference, dtype=float)

    scaler = StandardScaler().fit(X_reference)
    X_ref_scaled = scaler.transform(X_reference)
    X_samp_scaled = scaler.transform(X_sample)

    # per-dimension bin edges derived from the reference distribution
    bin_edges = [
        np.histogram_bin_edges(X_ref_scaled[:, j], bins=bins)
        for j in range(X_ref_scaled.shape[1])
    ]

    ref_bin_idx = _digitize_multidim(X_ref_scaled, bin_edges)
    samp_bin_idx = _digitize_multidim(X_samp_scaled, bin_edges)

    occupied_ref = {tuple(idx) for idx in ref_bin_idx}
    occupied_samp = {tuple(idx) for idx in samp_bin_idx}
    score = float(len(occupied_ref & occupied_samp) / len(occupied_ref))

    return {"score": score}


def _fully_connected_threshold(coords):
    """Minimum distance threshold ensuring full connectivity.

    Uses the maximum edge length of the minimum spanning tree.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n, 2)
        Point coordinates.

    Returns
    -------
    float
        Distance threshold.
    """
    dist_matrix = spatial.distance_matrix(coords, coords)
    mst = csgraph.minimum_spanning_tree(dist_matrix)
    return mst.max()


def distance_band_weights(coords, threshold=None):
    """Build a row-standardized distance-band spatial weights matrix.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n, 2)
        Point coordinates.
    threshold : float, optional
        Distance threshold. If ``None``, computed via
        :func:`_fully_connected_threshold`.

    Returns
    -------
    w : libpysal.weights.DistanceBand
        Row-standardized spatial weights.
    """
    if threshold is None:
        threshold = _fully_connected_threshold(coords)
    w = weights.DistanceBand(coords.tolist(), threshold=threshold)
    w.transform = "r"
    return w


def knn_weights(coords, k=4):
    """Build a row-standardized k-nearest neighbors spatial weights matrix.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n, 2)
        Point coordinates.
    k : int, default 4
        Number of nearest neighbors.

    Returns
    -------
    w : libpysal.weights.KNN
        Row-standardized spatial weights.
    """
    n = len(coords)
    w = weights.KNN(coords, k=min(k, n - 1), ids=list(range(n)))
    w.transform = "r"
    return w


def _ensure_weights(w_or_gser, *, unique_groups=None, weights_fn=None):
    """Return a spatial weights matrix, building one from a GeoSeries if needed.

    Parameters
    ----------
    w_or_gser : libpysal.weights.W or geopandas.GeoSeries
        Either a pre-built weights matrix (returned as-is) or a GeoSeries of
        site geometries from which weights are constructed via ``weights_fn``.
    unique_groups : array-like, optional
        Site identifiers used to filter ``w_or_gser`` before building weights.
        Ignored when ``w_or_gser`` is already a weights matrix.
    weights_fn : callable, optional
        ``(geopandas.GeoSeries) -> libpysal.weights.W`` used to build weights
        from the (filtered) GeoSeries. Defaults to KNN with k=4 when ``None``.
        Ignored when ``w_or_gser`` is already a weights matrix.

    Returns
    -------
    libpysal.weights.W
    """
    if not isinstance(w_or_gser, gpd.GeoSeries):
        return w_or_gser

    gser = w_or_gser if unique_groups is None else w_or_gser.loc[unique_groups]

    if weights_fn is None:
        return knn_weights(utils.as_coord_arr(gser))

    return weights_fn(gser)


def _aggregate_per_site(residuals, groups):
    """Average residuals per site, returning ``(site_residuals, unique_groups)``."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    site_residuals = np.array([residuals[groups == g].mean() for g in unique_groups])
    return site_residuals, unique_groups


def lisa_residuals(
    residuals, w, *, groups=None, permutations=999, weights_fn=None, **kwargs
):
    """Compute Local Moran's I on regression residuals (LISA).

    When observations have repeated measurements per site (e.g., multiple
    time steps per station), pass ``groups`` to aggregate residuals per site
    before computing the statistic.

    Parameters
    ----------
    residuals : array-like of shape (n_samples,)
        Regression residuals (``y_true - y_pred``).
    w : libpysal.weights.W or geopandas.GeoSeries
        Spatial weights matrix, or a GeoSeries of site geometries from which
        weights are built via ``weights_fn``. When ``groups`` is provided, the
        GeoSeries is filtered to the unique sites before building weights.
    groups : array-like of shape (n_samples,), optional
        Site identifiers for each observation. When provided, residuals
        are averaged per group before computing Local Moran's I.
    permutations : int, default 999
        Number of random permutations for significance testing.
    weights_fn : callable, optional
        ``(geopandas.GeoSeries) -> libpysal.weights.W`` used when ``w`` is a
        GeoSeries. Defaults to KNN with k=4 when ``None``.
    **kwargs
        Extra keyword arguments forwarded to :class:`esda.moran.Moran_Local`
        (e.g. ``seed`` for reproducible permutations).

    Returns
    -------
    dict
        ``{"Is": ndarray, "p_sim": ndarray, "q": ndarray}`` — per-site
        arrays. ``q`` follows the esda convention: 1=HH, 2=LH, 3=LL, 4=HL.

    See Also
    --------
    moran_residuals : Global Moran's I on residuals.
    """
    residuals = np.asarray(residuals)

    if groups is not None:
        site_residuals, unique_groups = _aggregate_per_site(residuals, groups)
    else:
        site_residuals, unique_groups = residuals, None

    if np.var(site_residuals) == 0:
        n_sites = len(site_residuals)
        return {
            "Is": np.full(n_sites, np.nan),
            "p_sim": np.full(n_sites, np.nan),
            "q": np.zeros(n_sites, dtype=int),
        }

    w = _ensure_weights(w, unique_groups=unique_groups, weights_fn=weights_fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        ml = moran.Moran_Local(site_residuals, w, permutations=permutations, **kwargs)
    return {"Is": ml.Is, "p_sim": ml.p_sim, "q": ml.q}


def moran_residuals(
    residuals, w, *, groups=None, permutations=999, weights_fn=None, **kwargs
):
    """Compute Moran's I on regression residuals.

    When observations have repeated measurements per site (e.g., multiple
    time steps per station), pass ``groups`` to aggregate residuals per site
    before computing the statistic.

    Parameters
    ----------
    residuals : array-like of shape (n_samples,)
        Regression residuals (``y_true - y_pred``).
    w : libpysal.weights.W or geopandas.GeoSeries
        Spatial weights matrix, or a GeoSeries of site geometries from which
        weights are built via ``weights_fn``. When ``groups`` is provided, the
        GeoSeries is filtered to the unique sites before building weights.
    groups : array-like of shape (n_samples,), optional
        Site identifiers for each observation. When provided, residuals
        are averaged per group before computing Moran's I.
    permutations : int, default 999
        Number of random permutations for significance testing.
    weights_fn : callable, optional
        ``(geopandas.GeoSeries) -> libpysal.weights.W`` used when ``w`` is a
        GeoSeries. Defaults to KNN with k=4 when ``None``.
    **kwargs
        Extra keyword arguments forwarded to :class:`esda.moran.Moran`
        (e.g. ``rng`` for reproducible permutations).

    Returns
    -------
    dict
        ``{"I": float, "p_sim": float}`` — Moran's I statistic and
        simulated p-value.

    Examples
    --------
    Pass a pre-built weights matrix:

    >>> result = moran_residuals(residuals, w, groups=station_ids)

    Or let the function build KNN weights from a GeoSeries:

    >>> result = moran_residuals(residuals, station_gser, groups=station_ids)

    Or supply a custom weights function:

    >>> from libpysal.weights import DistanceBand
    >>> from focalpy.utils import as_coord_arr
    >>> result = moran_residuals(
    ...     residuals,
    ...     station_gser,
    ...     groups=station_ids,
    ...     weights_fn=lambda gser: DistanceBand(
    ...         as_coord_arr(gser).tolist(),
    ...         threshold=2000,
    ...     ),
    ... )

    See Also
    --------
    focalpy.model_selection.SpatialKFold :
        Spatial cross-validator using the same ``groups`` convention.
    lisa_residuals : Local Moran's I on residuals.
    """
    residuals = np.asarray(residuals)

    if groups is not None:
        site_residuals, unique_groups = _aggregate_per_site(residuals, groups)
    else:
        site_residuals, unique_groups = residuals, None

    if np.var(site_residuals) == 0:
        return {"I": np.nan, "p_sim": np.nan}

    w = _ensure_weights(w, unique_groups=unique_groups, weights_fn=weights_fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = moran.Moran(site_residuals, w, permutations=permutations, **kwargs)
    return {"I": m.I, "p_sim": m.p_sim}
