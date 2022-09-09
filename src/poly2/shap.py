

import shap
from joblib import Memory

from xgboost import XGBRegressor


memory = Memory('../joblib_cache/', verbose=1)

#
#
#


@memory.cache
def get_shap_values(X, y, filename):

    model = XGBRegressor()

    model.load_model(filename)

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    return shap_values
