import pandas as pd
import warnings

import optuna
from optuna.samplers import TPESampler

import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from poly2.xgb_utils import HyperparamsObj, check_default_model
from poly2.utils import load_data


def main(model):
    X, y = load_data(model)

    X_cv = X.loc[lambda x: (x.run < 8000)].drop('run', axis=1)
    y_cv = y.loc[lambda x: (x.run < 8000)].drop('run', axis=1)

    X_test = X.loc[lambda x: (x.run >= 8000)].drop('run', axis=1)
    y_test = np.array(y.loc[lambda x: (x.run >= 8000)].drop('run', axis=1))

    default_score = check_default_model(X_cv, y_cv)

    best_score, best_pars = run_optuna(X_cv, y_cv)

    rmse_train, rmse_test, rmse_def_test = train_test_scores(
        best_pars,
        X_cv,
        X_test,
        y_cv,
        y_test
    )

    (
        pd.DataFrame(dict(
            model=model,
            default_cv_score=default_score,
            best_cv_score=best_score,
            rmse_test=rmse_test,
            rmse_train=rmse_train,
            rmse_test_def=rmse_def_test,
        ), index=[0]
        )
        .to_csv(f'../outputs/scores/{model}.csv', index=False)
    )

    return None


def run_optuna(X_cv, y_cv):
    np.random.seed(0)

    optuna.logging.set_verbosity(0)

    # ignore warning about Int64Index
    warnings.simplefilter(action='ignore', category=FutureWarning)

    obj = HyperparamsObj(X_cv, y_cv)

    sampler = TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler)

    study.optimize(obj, n_trials=1)

    best_pars = study.best_params
    best_value = study.best_value

    (
        pd.DataFrame(best_pars, index=[0])
        .to_csv(f'../outputs/hyperparams/{MODEL}.csv', index=False)
    )

    return best_value, best_pars


def train_test_scores(best_pars, X_train, X_test, y_train, y_test):

    best_model = XGBRegressor(**best_pars).fit(X_train, y_train)

    yp_train = best_model.predict(X_train)
    yp_test = best_model.predict(X_test)

    rmse_train = mean_squared_error(yp_train, y_train, squared=False)
    rmse_test = mean_squared_error(yp_test, y_test, squared=False)

    default_model = XGBRegressor().fit(X_train, y_train)

    yp_def = default_model.predict(X_test)

    rmse_def_test = mean_squared_error(yp_def, y_test, squared=False)

    return rmse_train, rmse_test, rmse_def_test


if __name__ == "__main__":
    MODEL = 'all'
    # MODEL = 'Y10'
    # MODEL = 'cumulative'
    # MODEL = 'asymp'

    main(MODEL)
