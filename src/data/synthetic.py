import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

from src import ROOT_PATH


class SyntheticData:

    instrument = None
    mode = 'normal_uniform'

    def __init__(self, n_samples_train, n_samples_test, **kwargs):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

    def get_data(self):
        data_dicts = []
        for n_samples in [self.n_samples_train, self.n_samples_test]:
            if self.mode == 'normal_uniform':
                u = np.random.normal(0.0, 1.0, n_samples)
                x = np.random.uniform(-2.0, 2.0, n_samples)
            elif self.mode == 'moons':
                ux, _ = make_moons(n_samples=n_samples, noise=0.125)  # noise=0.125
                u, x = ux[:, 0] - 0.6, 1.5 * ux[:, 1] - 0.5
            else:
                raise NotImplementedError()
            prop = 1.0 / (1.0 + np.exp(- (0.75 * x - u + 0.5)))
            t = np.random.binomial(1, prop, n_samples)

            y_pot0 = self.get_mu(0, x, u) + np.random.normal(0.0, 1.0, n_samples)
            y_pot1 = self.get_mu(1, x, u) + np.random.normal(0.0, 1.0, n_samples)
            y = y_pot0 * (1 - t) + y_pot1 * t
            # y = (2 * t - 1) * x + (2 * t - 1) - 2 * np.sin(2 * (2 * t - 1) * x + u) - 2 * u * (1 + 0.5 * x) + y_eps
            data_dicts.append({
                'cov_f': np.stack([x, u], -1),
                'prop': prop,
                'treat_f': t,
                'out_f': y,
                'out_pot0': y_pot0,
                'out_pot1': y_pot1,
                'mu0': self.get_mu(0, x, u),
                'mu1': self.get_mu(1, x, u),
            })
        return data_dicts

    def get_mu(self, treat, x, u):
        if treat == 0:
            return - 1 * x - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)
        else:
            return 1 * x + 1 - 2 * np.sin(2 * x + u) - 2 * u * (1 + 0.5 * x)


class SyntheticCurthVDS:
    """
    Synthetic benchmark from:
    Curth & van der Schaar (2021), arXiv:2101.10943, Supplement D.1 "Simulation set-up".

    Settings:
      (i)  tau(x)=0, confounding present
      (ii) tau(x) nonzero (5 additional predictive covariates), confounding present
      (iii) no confounding (pi(x)=0.5), mu0 and mu1 supported on disjoint covariate sets
    """

    instrument = None

    def __init__(
        self,
        n_samples_train: int,
        n_samples_test: int,
        setting: str = "ii",          # "i", "ii", or "iii"
        dim_cov: int = 25,
        xi: float = 3.0,              # selection bias strength (paper uses 3)
        omega_quantile: float = 0.5,  # 0.5 == median (paper uses median)
        noise_std: float = 1.0,       # epsilon ~ N(0, 1)
        **kwargs
    ):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

        if setting not in {"i", "ii", "iii"}:
            raise ValueError("setting must be one of {'i','ii','iii'}")
        # if d != 25:
        #     # The benchmark is defined for d=25 in the paper; allow override but keep explicit.
        #     raise ValueError("This benchmark is defined with d=25 (paper setup).")

        self.setting = setting
        self.d = dim_cov
        self.xi = float(xi)
        self.omega_quantile = float(omega_quantile)
        self.noise_std = float(noise_std)
        self._rng = None

        # --- small RNG wrappers (support both module-level RNG and Generator) ---
    def _normal(self, loc, scale, size):
        if self._rng is None:
            return np.random.normal(loc, scale, size)
        return self._rng.normal(loc, scale, size)

    def _binomial(self, n, p, size):
        if self._rng is None:
            return np.random.binomial(n, p, size)
        return self._rng.binomial(n, p, size)

    # --- DGP helpers ---
    @staticmethod
    def _expit(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_covariates(self, n_samples: int) -> np.ndarray:
        # X ~ N(0, I_d)
        return self._normal(0.0, 1.0, (n_samples, self.d))

    def _split_indices(self):
        """
        Fixed disjoint subsets consistent with the paper’s description:
          Settings (i)/(ii): X_C (5), X_O (5), X_tau (5), noise (10)
          Setting (iii): X_mu0 (10), X_mu1 (10), noise (5)
        """
        if self.setting in {"i", "ii"}:
            idx_c = np.arange(0, 5)     # confounders
            idx_o = np.arange(5, 10)    # outcome-only
            idx_tau = np.arange(10, 15) # predictive features for treatment effect (used in ii)
            return idx_c, idx_o, idx_tau
        else:
            idx_mu0 = np.arange(0, 10)
            idx_mu1 = np.arange(10, 20)
            return idx_mu0, idx_mu1

    def get_mu(self, treat: int, X: np.ndarray) -> np.ndarray:
        """
        Implements equations (13), (15), (16) from the supplement.
        """
        if self.setting in {"i", "ii"}:
            idx_c, idx_o, idx_tau = self._split_indices()
            X_co = np.concatenate([X[:, idx_c], X[:, idx_o]], axis=1)  # 10 dims
            mu0 = np.sum(X_co ** 2, axis=1)  # 1^T X_CO^2

            if treat == 0:
                return mu0

            # treat == 1
            if self.setting == "i":
                return mu0  # mu1 = mu0
            else:
                # mu1 = mu0 + 1^T X_tau^2
                mu1 = mu0 + np.sum(X[:, idx_tau] ** 2, axis=1)
                return mu1

        # setting == "iii"
        idx_mu0, idx_mu1 = self._split_indices()
        if treat == 0:
            return np.sum(X[:, idx_mu0] ** 2, axis=1)
        else:
            return np.sum(X[:, idx_mu1] ** 2, axis=1)

    def _propensity(self, X: np.ndarray, omega: float | None) -> np.ndarray:
        """
        Implements equation (14) for settings (i)/(ii), and pi(x)=0.5 for (iii).
        """
        if self.setting == "iii":
            return np.full(X.shape[0], 0.5, dtype=float)

        idx_c, _, _ = self._split_indices()
        m = np.mean(X[:, idx_c] ** 2, axis=1)  # (1/dc) * 1^T X_c^2
        if omega is None:
            omega = float(np.quantile(m, self.omega_quantile))
        return self._expit(self.xi * (m - omega))

    def get_data(self):
        """
        Returns:
          [train_dict, test_dict]
        with keys matching your existing benchmark:
          cov_f, prop, treat_f, out_f, out_pot0, out_pot1, mu0, mu1
        """
        # Generate covariates first so we can share omega (centering) across train/test in a run.
        X_train = self._sample_covariates(self.n_samples_train)
        X_test = self._sample_covariates(self.n_samples_test)

        omega = None
        if self.setting in {"i", "ii"}:
            idx_c, _, _ = self._split_indices()
            m_train = np.mean(X_train[:, idx_c] ** 2, axis=1)
            omega = float(np.quantile(m_train, self.omega_quantile))  # median if omega_quantile=0.5

        data_dicts = []
        for X in [X_train, X_test]:
            n_samples = X.shape[0]

            prop = self._propensity(X, omega=omega)
            t = self._binomial(1, prop, n_samples)

            mu0 = self.get_mu(0, X)
            mu1 = self.get_mu(1, X)

            # Paper defines: Y = W*mu1(X) + (1-W)*mu0(X) + eps, eps ~ N(0,1).
            # In potential outcomes form, use the same eps for Y(0) and Y(1) for each unit.
            eps = self._normal(0.0, self.noise_std, n_samples)
            y_pot0 = mu0 + eps
            y_pot1 = mu1 + eps
            y = y_pot0 * (1 - t) + y_pot1 * t

            data_dicts.append(
                {
                    "cov_f": X,
                    "prop": prop,
                    "treat_f": t,
                    "out_f": y,
                    "out_pot0": y_pot0,
                    "out_pot1": y_pot1,
                    "mu0": mu0,
                    "mu1": mu1,
                }
            )

        return data_dicts
