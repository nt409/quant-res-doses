
import numpy as np

from tqdm import tqdm

import shap
from joblib import Memory

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


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


class HyperparamsObj:
    def __init__(self, X_in, y_in) -> None:
        self.X = X_in
        self.y = y_in

    #
    #

    def __call__(self, trial):

        params = self.get_params(trial)

        score = self.run_model(params)

        return score

    #
    #

    def run_model(self, params):

        X, y = self.X, self.y

        rmse_list = []

        kf = KFold(n_splits=5, random_state=0, shuffle=True)

        for train_ind, val_ind in kf.split(X):

            X_tr = X.iloc[train_ind]
            y_tr = y.iloc[train_ind]

            X_v = X.iloc[val_ind]
            y_v = y.iloc[val_ind]

            y_tr = np.array(y_tr)
            y_v = np.array(y_v)

            model = XGBRegressor(**params).fit(X_tr, y_tr)

            y_p = model.predict(X_v)

            rmse = mean_squared_error(y_p, y_v, squared=False)

            rmse_list.append(rmse)

        score = sum(rmse_list)/len(rmse_list)

        return score

    #
    #

    def get_params(self, trial):
        params = {
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1),
        }
        return params


def check_default_model(X_cv, y_cv):
    rmse_list = []

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for train_ind, val_ind in tqdm(kf.split(X_cv)):

        X_tr = X_cv.iloc[train_ind]
        y_tr = y_cv.iloc[train_ind]

        X_v = X_cv.iloc[val_ind]
        y_v = y_cv.iloc[val_ind]

        y_tr = np.array(y_tr)
        y_v = np.array(y_v)

        model = XGBRegressor().fit(X_tr, y_tr)

        y_p = model.predict(X_v)

        rmse = mean_squared_error(y_p, y_v, squared=False)

        rmse_list.append(rmse)

    score = sum(rmse_list)/len(rmse_list)

    return score
