"""Get SHAP values for each model

Example
-------
conda activate poly2
cd src
python -m poly2.shap asymp
"""
import sys

import shap
from joblib import Memory
import warnings

from poly2.utils import get_best_model, load_data, object_dump

memory = Memory('../joblib_cache/', verbose=1)

# ignore warning about Int64Index
warnings.simplefilter(action='ignore', category=FutureWarning)


#
#
#


# @memory.cache
def get_shap_values(Xd, yd, model):

    best_model = get_best_model(model)

    print('fit')

    best_model.fit(Xd, yd)

    explainer = shap.TreeExplainer(best_model)

    print('get shap values')

    shap_vals = explainer(Xd)

    return shap_vals


if __name__ == "__main__":
    # MODEL = 'all' / 'Y10' / 'cumulative' / 'asymp'

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: the model name")

    MODEL = sys.argv[1]

    X, y = load_data(MODEL, include_run=False)

    shap_values = get_shap_values(X, y, MODEL)

    object_dump(shap_values, f'../outputs/SHAP/{MODEL}.pickle')
