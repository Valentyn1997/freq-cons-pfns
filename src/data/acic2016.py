import glob
import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler

from src import ROOT_PATH


class ACIC2016:
    """
    Collections of the semi-synthetic datasets from ACIC competitions of 2016
    """

    def __init__(self, **kwargs):
        self.cov_data_path = f'{ROOT_PATH}/data/acic_2016/x.csv'
        self.out_data_path = f'{ROOT_PATH}/data/acic_2016/synth_outcomes'

        self.simulation_files = sorted(glob.glob(f'{self.out_data_path}/*'))

    # def _postprocess_data(self, x, t, y_f, y_cf, dataset):
    #     y0_pot = np.where(t == 0, y_f, y_cf)
    #     y1_pot = np.where(t == 1, y_f, y_cf)
    #
    #     return {
    #         'cov_f': x,
    #         'treat_f': t,
    #         'out_f': y_f,
    #         'out_pot_0': y0_pot,
    #         'out_pot_1': y1_pot
    #     }

    # def load_treatment_and_outcome_f_cf(self, covariates, file_f, file_cf):
    #     out_f = pd.read_csv(file_f, index_col='sample_id', header=0, sep=',')
    #     out_cf = pd.read_csv(file_cf, index_col='sample_id', header=0, sep=',')
    #     dataset = covariates.join(out_f, how='inner').join(out_cf, how='inner')
    #     t = dataset['z'].values
    #     y_f = dataset['y'].values
    #     y_cf = np.where(t == 0, dataset['y1'].values, dataset['y0'].values)
    #     x = dataset.values[:, :-4]
    #     t, y_f, y_cf, x = t.reshape(-1, 1).astype(float), y_f.reshape(-1, 1), y_cf.reshape(-1, 1), x
    #     return self._postprocess_data(x, t, y_f, y_cf, dataset)

    def load_treatment_and_outcome(self, covariates, file):
        out = pd.read_csv(file, header=0, sep=',')
        dataset = covariates.join(out, how='inner')#.drop(columns=['mu0', 'mu1'])
        t = dataset['z'].values
        y_f = np.where(t == 1, dataset['y1'].values, dataset['y0'].values)
        y_cf = np.where(t == 1, dataset['y0'].values, dataset['y1'].values)
        mu0, mu1 = dataset['mu0'].values, dataset['mu1'].values
        x = dataset.values[:, :-5]
        t, y_f, y_cf, x = t.astype(float), y_f, y_cf, x.astype(float)
        mu0, mu1 = mu0, mu1
        y0_pot = np.where(t == 0, y_f, y_cf)
        y1_pot = np.where(t == 1, y_f, y_cf)

        data = {
            'cov_f': x,
            'treat_f': t,
            'out_f': y_f,
            'out_pot0': y0_pot,
            'out_pot1': y1_pot,
            'mu0': mu0,
            'mu1': mu1
        }

        # Standard scaling
        out_scaler = StandardScaler()
        out_scaler.fit(data['out_f'].reshape(-1, 1))
        data['out_f_scaled'] = out_scaler.transform(data['out_f'].reshape(-1, 1)).reshape(-1)
        data['out_pot0_scaled'] = out_scaler.transform(data['out_pot0'].reshape(-1, 1)).reshape(-1)
        data['out_pot1_scaled'] = out_scaler.transform(data['out_pot1'].reshape(-1, 1)).reshape(-1)

        return data

    def get_data(self) -> List[dict]:
        x_raw = pd.read_csv(self.cov_data_path, header=0, sep=',')
        x_raw = pd.get_dummies(x_raw)

        datasets = []
        for file in self.simulation_files:
            datasets.append(self.load_treatment_and_outcome(x_raw, file))
        return datasets