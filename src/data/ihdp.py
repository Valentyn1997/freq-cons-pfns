import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from typing import Tuple
import torch
import pandas as pd
from pyro.distributions.transforms.spline import ConditionedSpline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, expon

from src import ROOT_PATH


class IHDP:

    def __init__(self, **kwargs):
        self.train_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.train.npz"
        self.test_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.test.npz"

    def get_data(self):
        train_data = np.load(self.train_data_path, 'r')
        test_data = np.load(self.test_data_path, 'r')

        datasets = []

        for i in range(train_data['x'].shape[-1]):

            data_dicts = []
            out_scaler = None

            for kind, data in zip(['train', 'test'], [train_data, test_data]):
                if kind == 'train':
                    out_scaler = StandardScaler()
                    out_scaler.fit(data['yf'][:, i].reshape(-1, 1))

                data = {
                    'cov_f': data['x'][:, :, i],
                    'treat_f': data['t'][:, i],
                    'out_f': data['yf'][:, i],
                    'out_f_scaled': out_scaler.transform(data['yf'][:, i].reshape(-1, 1)).reshape(-1),
                    'out_pot0': np.where(1.0 - data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'out_pot1': np.where(data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'out_scaler.scale_': float(out_scaler.scale_),
                    'out_scaler.mean_': float(out_scaler.mean_),
                    'mu0': data['mu0'][:, i],
                    'mu1': data['mu1'][:, i]
                }

                data['out_pot0_scaled'] = out_scaler.transform(data['out_pot0'].reshape(-1, 1)).reshape(-1)
                data['out_pot1_scaled'] = out_scaler.transform(data['out_pot1'].reshape(-1, 1)).reshape(-1)

                data_dicts.append(data)

            datasets.append(data_dicts)

        return datasets