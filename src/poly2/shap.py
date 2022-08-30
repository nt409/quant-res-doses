from xgboost import XGBRegressor
import shap
from joblib import Memory


memory = Memory('../joblib_cache/', verbose=1)

#
#
#


@memory.cache
def get_shap_values(X):

    print('start')

    model = XGBRegressor()

    model.load_model('xgb_scan_all.json')

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    print('done')

    # shap_values = []

    return shap_values
