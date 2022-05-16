import pandas as pd
import numpy as np

from poly2.config import Config
from poly2.consts import MUTATION_PROP
from poly2.run import no_joblib_simulations_run
from poly2.utils import get_dist_mean, get_dist_var


def main(n_years=15):

    cmh = Config(
        type='single',
        sprays=[1, 2, 3],
        host_on=[False],
        n_years=n_years,
        n_k=500,
        n_l=50,
    )

    # cmh.mutation_proportion = 2e-1
    cmh.mutation_proportion = 0

    out = pd.DataFrame()

    # doses = np.linspace(0.1, 1, 10)
    doses = [0.5, 1]

    for dose in doses:

        run_output = no_joblib_simulations_run(
            cmh,
            doses=dose*np.ones(n_years)
        )

        output1 = run_output['spray_1_host_N']
        output2 = run_output['spray_2_host_N']
        output3 = run_output['spray_3_host_N']

        out = pd.concat(
            [
                out,
                get_dataframe_fig1(output1, 1, dose),
                get_dataframe_fig1(output2, 2, dose),
                get_dataframe_fig1(output3, 3, dose)
            ],
            ignore_index=True
        )

    out = out.reset_index(drop=True)

    conf_str = f'{cmh.n_k}_{cmh.n_l}'
    fn = f'../outputs/combined/fig1_{conf_str}.csv'

    print(f'Saving to {fn}')
    out.to_csv(fn, index=False)

    return None


def get_dataframe_fig1(output, sprays, dose):
    """Get dataframe summarising output

    Parameters
    ----------
    output : list of dicts
        --
    run : int
        --
    sprays : int
        --
    dose : float
        --

    Returns
    -------
    out : pd.DataFrame
        Columns:
        - run
        - sprays
        - dose
        - year
        - sev
        - yld
        - econ

    """

    mean = get_dist_mean(output['fung_dists'], output['k_vec'])
    var = get_dist_var(output['fung_dists'], output['k_vec'])

    out = pd.DataFrame(
        {
            'sprays': int(sprays),
            'dose': dose,
            'year': np.arange(1+len(output['yield_vec'])),
            'sev': np.concatenate([[np.nan], 100*output['dis_sev']]),
            'fung_mean': mean,
            'fung_var': var,
        }
    )

    return out


if __name__ == '__main__':
    main()
