"""Settings."""

import statsmodels.api as sm
from sklearn import decomposition, preprocessing

# focal analysis
FEATURE_PREPROCESSOR = preprocessing.StandardScaler
FEATURE_DECOMPOSER = decomposition.PCA

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
