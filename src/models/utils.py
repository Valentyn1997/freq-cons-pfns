import numpy as np
import torch
from copy import deepcopy
from sklearn.preprocessing import QuantileTransformer
from causalpfn import CATEEstimator
from causalfm import StandardCATEModel
from causalfm.data import normalize_data
from scipy.stats import gaussian_kde
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from typing import Dict
from functools import partial


class CEPOsEstimator(CATEEstimator):
    def estimate_cepos(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the conditional average treatment effect (CATE) using the fitted model.
        Args:
            X (np.ndarray): The input data with shape [N', D].
        """
        self._check_fitted()

        X_context = self.X_train
        t_context = self.t_train
        y_context = self.y_train
        X_query = X
        if self.max_feature_size is not None and X_query.shape[1] > self.max_feature_size:
            X_query = self.x_dim_transformer.transform(X_query)

        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)

        mu_0_and_1 = self._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            temperature=1.0,
        )

        mu_0 = mu_0_and_1[: X_query.shape[0]]
        mu_1 = mu_0_and_1[X_query.shape[0] :]
        return mu_0, mu_1

    def sample_cepos(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        self._check_fitted()

        X_context = self.X_train
        t_context = self.t_train
        y_context = self.y_train
        X_query = X
        if self.max_feature_size is not None and X_query.shape[1] > self.max_feature_size:
            X_query = self.x_dim_transformer.transform(X_query)

        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)

        _, samples = self._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            temperature=1.0,
            n_samples=n_samples,
        )

        samples_0 = samples[: X_query.shape[0]]
        samples_1 = samples[X_query.shape[0]:]
        return samples_0, samples_1


class CATEEstimator(StandardCATEModel):

    def fit(self, cov_f, treat_f, out_f):
        cov_f_scaled, out_f_scaled, _, self.out_scaler = normalize_data(cov_f, out_f)

        self.estimate_cate_fitted = partial(self.estimate_cate,
                                            x_train=torch.tensor(cov_f_scaled),
                                            a_train=torch.tensor(treat_f).reshape(-1, 1),
                                            y_train=torch.tensor(out_f_scaled).reshape(-1, 1))
        self.sample_cate_fitted = partial(self.sample_cate,
                                          x_train=torch.tensor(cov_f_scaled),
                                          a_train=torch.tensor(treat_f).reshape(-1, 1),
                                          y_train=torch.tensor(out_f_scaled).reshape(-1, 1))

    def estimate_cate(self, x_train: torch.Tensor, a_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_test_scaled, _, _, _ = normalize_data(x_test.cpu().numpy(), y_train.cpu().numpy())
        x_test_scaled = torch.tensor(x_test_scaled)
        out = super().estimate_cate(x_train, a_train, y_train, x_test_scaled)
        out['cate'] = out['cate'] * self.out_scaler.scale_[0]
        return out

    def sample_cate(self, x_train: torch.Tensor, a_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:
        self.model.eval()

        # Ensure tensors are on the correct device
        x_train = self._to_device(x_train)
        a_train = self._to_device(a_train)
        y_train = self._to_device(y_train)
        x_test = self._to_device(x_test)

        with torch.no_grad():
            out = self.model.estimate_cate(x_train, a_train, y_train, x_test, return_gmm=True)

        mix = Categorical(probs=out['gmm_pi'])  # component index distribution
        comp = Independent(Normal(loc=out['gmm_mu'].unsqueeze(-1), scale=out['gmm_sigma'].unsqueeze(-1)), 1)  # D-dim diagonal Gaussian
        gmm = MixtureSameFamily(mix, comp)
        out['sample'] = gmm.sample((n_samples,)).squeeze(-1) * self.out_scaler.scale_[0]
        return out


def subset_by_indices(data_dict: dict, indices: list):
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        if not isinstance(data_dict[k], float) and not isinstance(data_dict[k], str):
            subset_data_dict[k] = np.copy(data_dict[k][indices])
        else:
            subset_data_dict[k] = deepcopy(data_dict[k])
    return subset_data_dict


def ecdf(x):
    x = np.sort(x)
    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result


def tv_hist(x, y, bins="fd", data_range=None):
    x = np.asarray(x); y = np.asarray(y)
    if data_range is None:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        data_range = (lo, hi)

    cx, edges = np.histogram(x, bins=bins, range=data_range, density=False)
    cy, _     = np.histogram(y, bins=edges, range=data_range, density=False)

    px = cx / cx.sum()
    py = cy / cy.sum()
    return 0.5 * np.abs(px - py).sum()


def tv_kde(x, y, grid_size=4096, bw_method="scott", cut=3.0):
    x = np.asarray(x); y = np.asarray(y)

    kde_x = gaussian_kde(x, bw_method=bw_method)
    kde_y = gaussian_kde(y, bw_method=bw_method)

    # grid covering both samples with some padding
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pad = cut * max(sx, sy, 1e-12)
    lo = min(x.min(), y.min()) - pad
    hi = max(x.max(), y.max()) + pad
    grid = np.linspace(lo, hi, grid_size)

    px = kde_x(grid)
    py = kde_y(grid)
    return 0.5 * np.trapz(np.abs(px - py), grid)


class NormalScoreMarginal:
    def __init__(self, random_state=0, eps=1e-6):
        self.random_state = random_state
        self.eps = eps
        self.qt = None

    def fit(self, X):
        n = X.shape[0]
        self.qt = QuantileTransformer(
            n_quantiles=min(1000, n),
            output_distribution="normal",
            subsample=int(1e9),          # disable subsampling
            random_state=self.random_state,
            copy=True,
        )
        self.qt.fit(X)
        return self

    def transform(self, X):
        Z = self.qt.transform(X)
        # QuantileTransformer shouldn't produce inf, but it can if extrapolation is extreme; clip defensively.
        Z = np.clip(Z, -8.0, 8.0)
        assert X.shape == Z.shape
        return Z