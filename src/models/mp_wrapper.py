import numpy as np
from omegaconf import DictConfig
from typing import List
from tqdm import tqdm
from copy import deepcopy
import logging
from hydra.utils import instantiate
from joblib import Parallel, delayed
import torch
from scipy.stats import Normal, multivariate_normal
from sklearn.preprocessing import StandardScaler

from src.models.pfns import PFN
from src.models.utils import NormalScoreMarginal

logger = logging.getLogger(__name__)


class PFNsMartingalePosteriorWrapper(object):

    def __init__(self, args: DictConfig = None, pfn_config: DictConfig = None, **kwargs):
        self.hparams = args
        self.pfn_config = pfn_config

        assert pfn_config.backbone in ['tabpfn']

        # Hprams
        self.mp_steps = args.mp_wrapper.mp_steps
        self.dim_cov = args.dataset.dim_cov


    def get_posterior_paths(self, train_data_dict: dict, num_paths: int, nuisance: str, n_jobs=5) -> List[PFN]:

        assert nuisance in ['mu', 'prop']

        def mp_pfn():
            mp_data_dict = deepcopy(train_data_dict)
            pfn = instantiate(self.pfn_config, self.hparams, _recursive_=False)

            for step in range(self.mp_steps):
                # Fitting a new PFN
                pfn.fit(mp_data_dict)

                rand_ix, rand_q = np.random.randint(0, len(train_data_dict['cov_f'])), np.random.uniform()
                one_point_dict = {
                    'cov_f': mp_data_dict['cov_f'][rand_ix].reshape(-1, self.dim_cov),
                    'treat_f': mp_data_dict['treat_f'][rand_ix].reshape(-1),
                }

                if nuisance in ['mu']:
                    # Sampling a new datapoint from PPD
                    out_f_sample = pfn.get_posterior_predictive(one_point_dict,
                                                                treat=mp_data_dict['treat_f'][rand_ix],
                                                                output_type="quantiles",
                                                                quantiles=[rand_q])[0].squeeze()

                    # Extending a dataset
                    mp_data_dict['cov_f'] = np.concatenate([mp_data_dict['cov_f'], one_point_dict['cov_f']])
                    mp_data_dict['treat_f'] = np.concatenate([mp_data_dict['treat_f'], one_point_dict['treat_f']])
                    mp_data_dict['out_f'] = np.append(mp_data_dict['out_f'], out_f_sample)

                elif nuisance in ['prop']:
                    # Sampling a new datapoint from PPD
                    prop_sample = pfn.get_posterior_predictive(one_point_dict)
                    treat_f_sample = np.random.binomial(1, prop_sample)

                    # Extending a dataset
                    mp_data_dict['cov_f'] = np.concatenate([mp_data_dict['cov_f'], one_point_dict['cov_f']])
                    mp_data_dict['treat_f'] = np.append(mp_data_dict['treat_f'], treat_f_sample)

                else:
                    raise NotImplementedError()

            return pfn

        logger.info(f'Sampling posterior {num_paths} paths with {n_jobs} parallel jobs.')

        mp_pfns = Parallel(n_jobs=n_jobs)(delayed(mp_pfn)() for _ in tqdm(range(num_paths)))

        return mp_pfns, True


class ConditionedRecursiveCopula(object):
    def __init__(self, base_pfn: PFN, mp_steps: int, train_data_dict: dict, rho: float, nuisance: str, reduce_grid: int = 1, mode: str = 'natural'):
        self.base_pfn = base_pfn
        self.mp_steps = mp_steps
        self.train_data_dict = train_data_dict
        self.rho = rho
        self.nuisance = nuisance
        self.reduce_grid = reduce_grid
        self.mode = mode

        # Using data scaler for a better performance of copulas
        self.cov_scaler = NormalScoreMarginal()
        if self.nuisance == 'prop':
            self.cov_scaler.fit(self.train_data_dict['cov_f'])
        elif self.nuisance == 'mu':
            self.cov_scaler.fit(np.append(self.train_data_dict['cov_f'], self.train_data_dict['treat_f'][:, None], 1))

    def _get_alpha(self, new_points_dict, data_dict, mp_step, n, treat=None):
        # Appending treatment to covariates
        if 'treat_f' in new_points_dict:
            cov_f = np.append(data_dict['cov_f'], treat * np.ones((data_dict['cov_f'].shape[0], 1)), 1)
            cov_f_new_points = np.append(new_points_dict['cov_f'], new_points_dict['treat_f'][:, None], 1)
        else:
            cov_f = data_dict['cov_f']
            cov_f_new_points = new_points_dict['cov_f']

        # Scaling
        cov_f_scaled = self.cov_scaler.transform(cov_f)
        cov_f_new_points_scaled = self.cov_scaler.transform(cov_f_new_points)

        cov_f_new_points_scaled = np.repeat(cov_f_new_points_scaled[None], cov_f_scaled.shape[0], 0)
        cov_f_scaled = np.repeat(cov_f_scaled[:, None], cov_f_new_points_scaled.shape[1], 1)

        cov_joint = np.stack([cov_f_scaled, cov_f_new_points_scaled], -1)
        # print(cov_joint.shape)

        i = mp_step + n
        alpha = (2 - 1 / i) / (i + 1)

        bivar_norm = multivariate_normal(np.zeros((2, )), np.array([[1.0, self.rho], [self.rho,  1.0]]))
        joint_pdf = bivar_norm.pdf(cov_joint)
        joint_pdf = joint_pdf if len(joint_pdf.shape) == 3 else joint_pdf[:, :, None]
        c = joint_pdf / (Normal().pdf(cov_f_scaled) + 1e-9) / (Normal().pdf(cov_f_new_points_scaled) + 1e-9)

        alpha_x_x = alpha * np.prod(c, -1) / (1 - alpha + alpha * np.prod(c, -1))
        return alpha_x_x

    def _get_cdf_recursive(self, values, data_dict, treat, mp_step, n_samples):
        assert self.base_pfn.kind == 'outcome_pfn' and self.nuisance == 'mu'

        # Recursive update
        if mp_step == 0:
            out = self.base_pfn.get_posterior_predictive(data_dict, treat, output_type="full")
            cdf_prev = out["criterion"].cdf(out['logits'], torch.tensor(values).to(out['logits'].device)).cpu().detach().numpy()
        else:
            cdf_prev = self._get_cdf_recursive(values, data_dict, treat, mp_step - 1, n_samples)

        # Clipping for stability
        cdf_prev = np.clip(cdf_prev, 1e-6, 1 - 1e-6)

        # Sampling random noise
        rand_ix = np.random.randint(0, len(self.train_data_dict['cov_f']), size=n_samples)
        if self.mode == 'natural':
            u = np.random.uniform(size=(1, n_samples, values.shape[0]))
        elif self.mode == 'independent':
            u = np.random.uniform(size=(cdf_prev.shape[0], n_samples, 1))
        elif self.mode == 'parallel':
            u = np.random.uniform(size=(1, n_samples, 1))
        else:
            raise NotImplementedError()

        new_points_dict = {
            'cov_f': self.train_data_dict['cov_f'][rand_ix].reshape(-1, self.train_data_dict['cov_f'].shape[1]),
            'treat_f': self.train_data_dict['treat_f'][rand_ix].reshape(-1),
        }

        # Inferring a learning rate
        alpha = self._get_alpha(new_points_dict, data_dict, mp_step, len(self.train_data_dict['cov_f']), treat)

        # reshaping
        alpha = alpha[:, :, None]
        cdf_prev = cdf_prev[:, None] if len(cdf_prev.shape) == 2 else cdf_prev
        # u = u[None, :, None]

        H = Normal().cdf((Normal().icdf(cdf_prev) - self.rho * Normal().icdf(u)) / np.sqrt(1 - self.rho ** 2))

        logger.info(f'MP step: {mp_step}')
        return (1 - alpha) * cdf_prev + alpha * H

    def _get_pdf_recursive(self, data_dict, mp_step, n_samples, mp_batch=1):
        assert self.base_pfn.kind == 'prop_pfn' and self.nuisance == 'prop'

        # Sampling random noise
        rand_ix = np.random.randint(0, len(self.train_data_dict['cov_f']), n_samples * mp_batch)
        new_points_dict = {
            'cov_f': self.train_data_dict['cov_f'][rand_ix].reshape(-1, self.train_data_dict['cov_f'].shape[1]),
        }

        # Recursive update
        merged_dicts = {
            'cov_f': np.concatenate([data_dict['cov_f'], new_points_dict['cov_f']])
        }

        if mp_step == 0:
            out = self.base_pfn.get_posterior_predictive(merged_dicts)
        else:
            out = self._get_pdf_recursive(merged_dicts, mp_step - 1, n_samples, mp_batch)

        pdf_prev, pdf_prev_next = out[:-n_samples * mp_batch], out[-n_samples * mp_batch:]
        pdf_prev_next = np.diag(pdf_prev_next) if len(pdf_prev_next.shape) == 2 else pdf_prev_next

        # Sampling from a prev_pdf
        if self.mode == 'natural' or self.mode == 'parallel':
            treat_f_sample = np.random.binomial(1, pdf_prev_next, size=(1, n_samples * mp_batch)).astype(float)
        elif self.mode == 'independent':
            treat_f_sample = np.random.binomial(1, pdf_prev_next, size=(pdf_prev.shape[0], n_samples * mp_batch)).astype(float)
        else:
            raise NotImplementedError()
        r = np.where(treat_f_sample == 1.0, pdf_prev_next, 1.0 - pdf_prev_next)
        q = pdf_prev[:, None] if len(pdf_prev.shape) == 1 else pdf_prev

        # Inferring a learning rate
        alpha = self._get_alpha(new_points_dict, data_dict, mp_step, len(self.train_data_dict['cov_f']))

        c = np.where(treat_f_sample == 1.0,
                     1.0 - self.rho + self.rho * np.minimum(r, q) / r / q,
                     1.0 - self.rho + self.rho * (q - np.minimum(q, (1.0 - r))) / r / q)
        logger.info(f'MP step: {mp_step}')
        return (1 - alpha + alpha * c) * q

    def get_posterior_predictive(self, data_dict: dict, treat: int = None, n_samples: int = None, **kwargs) -> np.ndarray:
        logger.info(f'Inferring posterior predictive for: {self.nuisance}')

        if self.nuisance == 'mu':
            out = self.base_pfn.get_posterior_predictive(data_dict, treat, output_type="full")
            values_grid = out["criterion"].borders.cpu().detach().numpy()[::self.reduce_grid]
            cdfs = self._get_cdf_recursive(values_grid, data_dict, treat, self.mp_steps, n_samples)
            d_cdfs = np.diff(cdfs)
            values_grid_mid = 0.5 * (values_grid[1:] + values_grid[:-1])
            result = np.nansum(values_grid_mid * d_cdfs, axis=-1)
        elif self.nuisance == 'prop':
            result = self._get_pdf_recursive(data_dict, self.mp_steps, n_samples)
        else:
            raise NotImplementedError()
        return result


class CopulaMartingalePosteriorWrapper(object):

    def __init__(self, args: DictConfig = None, pfn_config: DictConfig = None, **kwargs):
        self.hparams = args
        self.pfn_config = pfn_config

        assert pfn_config.backbone in ['tabpfn']

        # Hprams
        self.mp_steps = args.mp_wrapper.mp_steps
        self.reduce_grid = args.mp_wrapper.reduce_grid
        self.dim_cov = args.dataset.dim_cov
        self.rho = args.mp_wrapper.rho
        self.mode = args.mp_wrapper.mode

    def get_posterior_paths(self, train_data_dict: dict, num_paths: int, nuisance: str) -> List[ConditionedRecursiveCopula]:
        assert nuisance in ['mu', 'prop']

        base_pfn = instantiate(self.pfn_config, self.hparams, _recursive_=False).fit(train_data_dict)
        mp_copula = ConditionedRecursiveCopula(base_pfn, self.mp_steps, train_data_dict, self.rho, nuisance, self.reduce_grid, self.mode)

        return mp_copula, False