import pandas as pd


class Constants:
    def __init__(self):
        """
        Economic params, running params, setup etc
        """

        consts = pd.read_csv('../data/03_model_inputs/constants.csv')

        self.wheat_price = float(
            consts[consts['Parameter'] == 'Price of wheat'].Value
        )

        self.spray_costs = {
            '0': 0,
            '1': float(consts[consts['Parameter'] == 'Cost 1 spray'].Value),
            '2': float(consts[consts['Parameter'] == 'Cost 2 sprays'].Value),
            '3': float(consts[consts['Parameter'] == 'Cost 3 sprays'].Value)
        }

        # times
        self.T_1 = float(consts[consts['Parameter'] == 'T1'].Value)
        self.T_2 = float(consts[consts['Parameter'] == 'T2'].Value)
        self.T_3 = float(consts[consts['Parameter'] == 'T3'].Value)
        self.T_end = float(consts[consts['Parameter'] == 'T_end'].Value)

        # host growth
        self.host_growth_rate = float(
            consts[consts['Parameter'] == 'Host_growth_rate'].Value
        )

        self.host_growth_initial_area = float(
            consts[consts['Parameter'] == 'Host_growth_initial_area'].Value
        )

        # # mapping between strat and index
        # self.strategy_mapping = dict(
        #     spray_N_host_N=0,
        #     spray_N_host_Y=1,
        #     spray_Y1_host_N=2,
        #     spray_Y1_host_Y=3,
        #     spray_Y2_host_N=4,
        #     spray_Y2_host_Y=5,
        #     spray_Y3_host_N=6,
        #     spray_Y3_host_Y=7
        # )


PARAMS = Constants()
