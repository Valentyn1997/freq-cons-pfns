import numpy as np
import torch
from omegaconf import DictConfig
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
from typing import List

from src import ROOT_PATH
from src.models.utils import CEPOsEstimator, CATEEstimator


class PFN(object):
    kind = None

    def __init__(self, args: DictConfig = None, **kwargs):
        self.hparams = args
        self.backbone = args[self.kind].backbone
        assert self.backbone in ['tabpfn', 'causalpfn', 'causalfm']

    def fit(self, train_data_dict: dict):
        raise NotImplementedError()

    def get_posterior_predictive(self, data_dict: dict, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class PropensityPFN(PFN):
    kind = 'prop_pfn'

    def __init__(self, args: DictConfig = None, **kwargs):
        super().__init__(args, **kwargs)

        # Models init
        self.backbone_pfn = None
        if self.backbone == 'tabpfn':
            self.backbone_pfn = TabPFNClassifier(device=args.exp.device, ignore_pretraining_limits=True)
        else:
            raise NotImplementedError()

    def fit(self, train_data_dict: dict):
        cov_f, treat_f = train_data_dict['cov_f'], train_data_dict['treat_f']

        if self.backbone == 'tabpfn':
            self.backbone_pfn.fit(cov_f, treat_f)
        else:
            raise NotImplementedError()

        return self

    def get_posterior_predictive(self, data_dict: dict, **kwargs) -> np.ndarray:
        cov_f = data_dict['cov_f']

        if self.backbone == 'tabpfn':
            return self.backbone_pfn.predict_proba(cov_f)[:, 1]
        else:
            raise NotImplementedError()


class OutcomePFN(PFN):
    kind = 'outcome_pfn'

    def __init__(self, args: DictConfig = None, **kwargs):
        super().__init__(args, **kwargs)

        # Hparams
        self.learner_type = args[self.kind].learner_type
        self.treat_options = [0.0, 1.0]

        # Models init
        self.backbone_pfn = None
        if self.backbone == 'tabpfn':
            if self.learner_type == 's':
                self.backbone_pfn = TabPFNRegressor(device=args.exp.device, ignore_pretraining_limits=True)
            elif self.learner_type == 't':
                self.backbone_pfn = [TabPFNRegressor(device=args.exp.device, ignore_pretraining_limits=True) for _ in range(len(self.treat_options))]
            else:
                raise NotImplementedError()

        elif self.backbone == 'causalpfn':
            self.backbone_pfn = CEPOsEstimator(device=args.exp.device, verbose=False)

        elif self.backbone == 'causalfm':
            self.kind = 'cate_pfn'
            self.backbone_pfn = CATEEstimator.from_pretrained(f"{ROOT_PATH}/causalfm/checkpoints/checkpoints_standard/best_model.pth")
        else:
            raise NotImplementedError()

    def fit(self, train_data_dict: dict):
        cov_f, treat_f, out_f = train_data_dict['cov_f'], train_data_dict['treat_f'], train_data_dict['out_f']

        if self.backbone == 'tabpfn':

            if self.learner_type == 's':
                inp_f = np.concatenate([cov_f, treat_f.reshape(-1, 1)], axis=1)
                self.backbone_pfn.fit(inp_f, out_f)
            elif self.learner_type == 't':
                for treat in self.treat_options:
                    self.backbone_pfn[int(treat)].fit(cov_f[treat_f == treat], out_f[treat_f == treat])

        elif self.backbone == 'causalpfn' or self.backbone == 'causalfm':
            self.backbone_pfn.fit(cov_f, treat_f, out_f)

        else:
            raise NotImplementedError()

        return self

    def get_posterior_predictive(self, data_dict: dict, treat: float = None, output_type: str = 'mean', quantiles: List[float] = None, **kwargs) -> np.ndarray:
        cov_f = data_dict['cov_f']

        # if treat is None: # Factual outcomes
        #     treat_f = data_dict['treat_f']

        if self.backbone == 'tabpfn':
            if self.learner_type == 's':
                if treat is not None:  # Potential outcomes
                    inp = np.concatenate([cov_f, treat * np.ones((cov_f.shape[0], 1))], axis=1)
                # else:  # Factual outcomes
                #     inp = np.concatenate([cov_f, treat_f.reshape(-1, 1)], axis=1)
                return self.backbone_pfn.predict(inp, output_type=output_type, quantiles=quantiles)

            elif self.learner_type == 't':
                if treat is not None:  # Potential outcomes
                    return self.backbone_pfn[int(treat)].predict(cov_f, output_type=output_type, quantiles=quantiles)
                # else:  # Factual outcomes
                #     out = np.empty((cov_f.shape[0], 1))
                #     for treat in self.treat_options:
                #         out[treat_f == treat] = self.backbone_pfn[int(treat)].predict(cov_f[treat_f == treat], output_type=output_type, quantiles=quantiles)
                #     return out

        elif self.backbone == 'causalpfn':
            if output_type == 'mean':
                if treat is not None:  # Potential outcomes
                    return self.backbone_pfn.estimate_cepos(cov_f)[int(treat)]
                # else:  # Factual outcomes
                #     mu0, mu1 = self.backbone_pfn.estimate_cepos(cov_f)
                #     return mu0 * (treat_f == 0.0) + mu1 * (treat_f == 1.0)
            elif output_type == 'quantiles' and kwargs['kind'] == 'sampling':
                if treat is not None:  # Potential outcomes
                    return self.backbone_pfn.sample_cepos(cov_f, n_samples=len(quantiles))[int(treat)].T
            else:
                raise NotImplementedError()

        elif self.backbone == 'causalfm':
            if output_type == 'mean':
                return self.backbone_pfn.estimate_cate_fitted(x_test=torch.tensor(cov_f))['cate'].cpu().numpy()
            elif output_type == 'quantiles' and kwargs['kind'] == 'sampling':
                return self.backbone_pfn.sample_cate_fitted(x_test=torch.tensor(cov_f), n_samples=len(quantiles))['sample'].cpu().numpy()

        else:
            raise NotImplementedError()



