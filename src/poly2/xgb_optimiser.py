from xgboost import XGBRegressor
import warnings

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler


from sklearn.metrics import mean_squared_error

from poly2.xgb_utils import HyperparamsObj, check_default_model
from poly2.utils import load_train_test_data

# ignore warning about Int64Index
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(model):

    print(f'{model=}')

    print('load data')
    X_cv, y_cv, X_test, y_test = load_train_test_data(model)

    print('check default model')
    default_score = check_default_model(X_cv, y_cv)

    print('optimise hyperparameters')
    best_score, best_pars = run_optuna(X_cv, y_cv, model)

    print('get train and test scores')
    rmse_train, rmse_test, rmse_def_test = train_test_scores(
        best_pars,
        X_cv,
        X_test,
        y_cv,
        y_test,
        model,
    )

    print('Save scores')
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


def run_optuna(X_cv, y_cv, model):
    np.random.seed(0)

    optuna.logging.set_verbosity(0)

    obj = HyperparamsObj(X_cv, y_cv)

    sampler = TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler)

    study.optimize(obj, n_trials=300)

    best_pars = study.best_params
    best_value = study.best_value

    (
        pd.DataFrame(best_pars, index=[0])
        .to_csv(f'../outputs/hyperparams/{model}.csv', index=False)
    )

    return best_value, best_pars


def train_test_scores(best_pars, X_train, X_test, y_train, y_test, model):

    best_model = XGBRegressor(**best_pars).fit(X_train, y_train)

    yp_train = best_model.predict(X_train)
    yp_test = best_model.predict(X_test)

    rmse_train = mean_squared_error(yp_train, y_train, squared=False)
    rmse_test = mean_squared_error(yp_test, y_test, squared=False)

    best_model.save_model(f'../outputs/xgb/{model}.json')

    default_model = XGBRegressor().fit(X_train, y_train)

    yp_def = default_model.predict(X_test)

    rmse_def_test = mean_squared_error(yp_def, y_test, squared=False)

    return rmse_train, rmse_test, rmse_def_test


if __name__ == "__main__":
    # MODEL = 'all'
    # MODEL = 'Y10'
    # MODEL = 'cumulative'
    MODEL = 'asymp'

    main(MODEL)
