import shap
from joblib import Memory
import warnings

from xgboost import XGBRegressor

from poly2.utils import load_data, object_dump

memory = Memory('../joblib_cache/', verbose=1)


# ignore warning about Int64Index
warnings.simplefilter(action='ignore', category=FutureWarning)


#
#
#


# @memory.cache
def get_shap_values(X_in, y_in, filename):

    model = XGBRegressor()

    model.load_model(filename)

    print('fit')

    model.fit(X_in, y_in)

    explainer = shap.TreeExplainer(model)

    print('get shap values')

    shap_vals = explainer(X_in)

    return shap_vals


if __name__ == "__main__":
    # MODEL = 'all'
    MODEL = 'Y10'
    # MODEL = 'cumulative'
    # MODEL = 'asymp'

    X, y = load_data(MODEL, include_run=False)

    shap_values = get_shap_values(X, y, f'../outputs/xgb/{MODEL}.json')

    object_dump(shap_values, f'../outputs/SHAP/{MODEL}.pickle')
