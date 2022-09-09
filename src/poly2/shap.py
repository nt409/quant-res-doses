import shap
from joblib import Memory

from xgboost import XGBRegressor

from poly2.utils import load_data, object_dump

memory = Memory('../joblib_cache/', verbose=1)

#
#
#


# @memory.cache
def get_shap_values(X_in, y_in, filename):

    model = XGBRegressor()

    model.load_model(filename)

    model.fit(X_in, y_in)

    explainer = shap.TreeExplainer(model)

    shap_vals = explainer(X_in)

    return shap_vals


if __name__ == "__main__":
    MODEL = 'all'
    # MODEL = 'Y10'
    # MODEL = 'cumulative'
    # MODEL = 'asymp'

    X, y = load_data(MODEL)

    shap_values = get_shap_values(X, y, f'../outputs/xgb/{MODEL}.json')

    object_dump(shap_values, f'../outputs/SHAP/{MODEL}.pickle')
