"""Settings."""

import statsmodels.api as sm
from sklearn import linear_model, pipeline

# compute features
# TODO: dict-like fillna for different statistics?
VECTOR_FEATURES_FILLNA = 0
RASTER_FEATURES_FILLNA = 0

# focal analysis
# FEATURE_PREPROCESSOR = preprocessing.StandardScaler
# FEATURE_DECOMPOSER = decomposition.PCA
INFERENCE_PIPELINE_STEPS = [
    # ("scaler", preprocessing.StandardScaler()),
    ("model", linear_model.LinearRegression()),
]
INFERENCE_PIPELINE = pipeline.Pipeline

# scale of effect feature selection
SCALE_OF_EFFECT_CRITERIA = "rsquared"
SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT = {
    # statsmodels
    "rsquared": "max",
    "aic": "min",
    "bic": "min",
    # scipy.stats
    "spearmanr": "max",
}
SCALE_OF_EFFECT_MODEL = sm.OLS
SCALE_OF_EFFECT_HOW = "global"
