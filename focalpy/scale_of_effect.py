"""Scale-of-effect evaluation."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import typing as npt
from scipy import stats
from statsmodels.base.model import Model

from focalpy import settings, utils

__all__ = [
    "scale_eval_ser",
    "scale_of_effect_features",
]


def _scipy_statistic(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    func: Callable,
    *args: Sequence,
    **kwargs: utils.KwargsType,
) -> float:
    # raise for stats.ConstantInputWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # TODO: do we need a try/except? if so, which error to catch?
        # try:
        return func(X, y, **kwargs).statistic
        # except:
        #     return np.nan


def _model_attr(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    model: Model,
    attr: str,
    *,
    add_constant: bool = True,
    **model_kwargs: utils.KwargsType,
) -> float:
    """
    Extract an attribute from a (fitted) model (statsmodels or spreg).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    model : Model
        A statsmodels or spreg model class.
    attr : str
        The attribute to extract from the model.
    add_constant : bool, default True

    Returns
    -------
    float
        The value of the specified attribute.
    """
    if add_constant:
        X = sm.add_constant(X)
    _model = model(y, X, **model_kwargs)
    if hasattr(_model, "fit"):
        # compat between statsmodels and spreg
        _model = _model.fit()
    return getattr(_model, attr)


def scale_eval_ser(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    *,
    criteria: str | None = None,
    model: Model | None = None,
    **eval_func_kwargs: utils.KwargsType,
) -> pd.Series:
    """
    Evaluate the scale effect for each feature in `X_df` against `y_ser`.

    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature data frame where each column represents a feature at a specific scale,
        following the naming pattern "{feature}_{scale}" (e.g., `"density_500"`).
    y_ser : pandas.Series
        Response variable, as pandas Series with the same index as `X_df`.
    criteria : str, optional
        The evaluation criteria to use, which can be either a statistical test from
        `scipy.stats` (e.g., `"pearsonr"`, `"spearmanr"`) or a model attribute from a
        statsmodels or spreg model (e.g., `"rsquared"`, `"aic"`). If `None`, defaults
        to `settings.SCALE_OF_EFFECT_CRITERIA`.
    model : statsmodels or spreg Model class, optional
        The model class to use if `criteria` is a model attribute. If `None`, defaults
        to `settings.SCALE_OF_EFFECT_MODEL`. Ignored if `criteria` is a `scipy.stats`
        function.
    **eval_func_kwargs : mapping, optional
        Keyword arguments to pass to the evaluation function.

    Returns
    -------
    pandas.Series
        A Series with MultiIndex (feature group, scale) containing the evaluation scores
        for each feature at each scale.
    """
    # process criteria arg
    if criteria is None:
        criteria = settings.SCALE_OF_EFFECT_CRITERIA
    if hasattr(stats, criteria):
        # non-parametric with scipy.stats
        eval_func = _scipy_statistic
        scipy_func = getattr(stats, criteria)
        extra_eval_func_args = [scipy_func]
    else:
        # parametric with statsmodels or spreg
        eval_func = _model_attr
        if model is None:
            model = settings.SCALE_OF_EFFECT_MODEL
        extra_eval_func_args = [model, criteria]

    return (
        X_df.T.groupby(
            by=X_df.columns.str.split("_")
            .str[:-1]
            .map(lambda col_parts: "_".join(col_parts))
        )
        .apply(
            lambda group_df: group_df.T.apply(
                eval_func, args=(y_ser, *extra_eval_func_args)
            )
        )
        .rename(criteria)
    )


def scale_of_effect_features(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    *,
    criteria: str | None = None,
    direction: str | None = None,
    model: Model | None = None,
    **eval_func_kwargs: utils.KwargsType,
) -> np.ndarray:
    """
    Identify the scale-of-effect for each feature in `X_df` against `y_ser`.

    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature data frame where each column represents a feature at a specific scale,
        following the naming pattern "{feature}_{scale}" (e.g., `"density_500"`).
    y_ser : pandas.Series
        Response variable, as pandas Series with the same index as `X_df`.
    criteria : str, optional
        The evaluation criteria to use, which can be either a statistical test from
        `scipy.stats` (e.g., `"pearsonr"`, `"spearmanr"`) or a model attribute from a
        statsmodels or spreg model (e.g., `"rsquared"`, `"aic"`). If `None`, defaults
        to `settings.SCALE_OF_EFFECT_CRITERIA`.
    direction : str, optional
        The direction of the criteria, either `"max"` (higher is better) or `"min"`
        (lower is better). If `None`, the direction is inferred from
        `settings.SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT`. Required if the direction
        cannot be inferred.
    model : statsmodels or spreg Model class, optional
        The model class to use if `criteria` is a model attribute. If `None`, defaults
        to `settings.SCALE_OF_EFFECT_MODEL`. Ignored if `criteria` is a `scipy.stats`
        function.
    **eval_func_kwargs : mapping, optional
        Keyword arguments to pass to the evaluation function.

    Returns
    -------
    numpy.ndarray
        An array of scale-of-effect feature names for each feature in `X_df`.
    """
    eval_ser = scale_eval_ser(
        X_df,
        y_ser,
        criteria=criteria,
        model=model,
        **eval_func_kwargs,
    )
    # TODO: manage index level names better than default "level_0", "level_1", etc?
    eval_ser_gb = eval_ser.reset_index(level=0).groupby("level_0")

    if direction is None:
        direction = settings.SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT.get(
            eval_ser.name, None
        )
    if direction is None:
        raise ValueError(
            f"Direction (min/max) for criteria '{criteria}' is not specified and not "
            "found in settings."
        )
    else:
        return getattr(eval_ser_gb, f"idx{direction}")().values.flatten()
