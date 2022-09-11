import sys

from xgboost import XGBRegressor
import warnings

import numpy as np
import pandas as pd

import optuna


from sklearn.metrics import mean_squared_error

from poly2.utils import HyperparamsObj, get_model_cv_score, load_train_test_data

# ignore warning about Int64Index
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(model, seed):

    print(f'{model=}, {seed=}')

    # load data
    X_cv, y_cv, X_test, y_test = load_train_test_data(model)

    # check default model
    default_score = get_model_cv_score(X_cv, y_cv, {})

    # optimise hyperparameters
    best_score, best_pars, best_number = run_optuna(X_cv, y_cv, seed)

    # Save best pars
    (
        pd.DataFrame(best_pars, index=[0])
        .to_csv(f'../outputs/hyperparams/{model}_{seed}.csv', index=False)
    )

    # get train and test scores
    rmse_train, rmse_test, rmse_test_def = train_test_scores(
        best_pars,
        X_cv,
        y_cv,
        X_test,
        y_test,
        # model,
    )

    # Save scores
    (
        pd.DataFrame(dict(
            model=model,
            default_cv_score=default_score,
            best_cv_score=best_score,
            rmse_test=rmse_test,
            rmse_train=rmse_train,
            rmse_test_def=rmse_test_def,
            number=best_number,
        ), index=[0]
        )
        .to_csv(f'../outputs/scores/{model}_{seed}.csv', index=False)
    )

    return None


def run_optuna(X_cv, y_cv, seed):
    np.random.seed(seed)

    optuna.logging.set_verbosity(0)

    obj = HyperparamsObj(X_cv, y_cv)

    sampler = optuna.samplers.RandomSampler(seed=seed)

    study = optuna.create_study(sampler=sampler)

    # study.optimize(obj, n_trials=40)
    study.optimize(obj, n_trials=1)

    best_pars = study.best_params
    best_value = study.best_value
    best_number = study.best_trial.number

    return best_value, best_pars, best_number


def train_test_scores(
    best_pars,
    X_train,
    y_train,
    X_test,
    y_test,
    # model
):

    best_model = XGBRegressor(**best_pars).fit(X_train, y_train)

    yp_train = best_model.predict(X_train)
    yp_test = best_model.predict(X_test)

    rmse_train = mean_squared_error(yp_train, y_train, squared=False)
    rmse_test = mean_squared_error(yp_test, y_test, squared=False)

    # best_model.save_model(f'../outputs/xgb/{model}.json')

    default_model = XGBRegressor().fit(X_train, y_train)

    yp_def = default_model.predict(X_test)

    rmse_def_test = mean_squared_error(yp_def, y_test, squared=False)

    return rmse_train, rmse_test, rmse_def_test


if __name__ == "__main__":
    # MODEL = 'all'
    # MODEL = 'Y10'
    # MODEL = 'cumulative'
    # MODEL = 'asymp'

    if len(sys.argv) != 3:
        raise Exception("Supply one argument: the model name")

    MODEL = sys.argv[1]
    SEED = int(sys.argv[2])

    main(MODEL, SEED)
