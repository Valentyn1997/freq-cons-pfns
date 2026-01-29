import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from lightning.fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit, KFold
import os
import numpy as np
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm

import sys

from src.models.utils import subset_by_indices

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of train_data_dict
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    n_1_4 = args.dataset.n_samples_train ** (1 / 4)
    data_dicts = dataset.get_data() if args.dataset.collection else [dataset.get_data()]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        train_test_splits = []

        # Train-test split
        if args.dataset.train_test_splitted:
            train_data_dict, test_data_dict = data_dict[0], data_dict[1]
            train_test_splits.append((train_data_dict, test_data_dict))
        else:
            if hasattr(args.dataset, 'k_fold'):
                rs = KFold(n_splits=args.dataset.k_fold, random_state=args.exp.seed, shuffle=True)
            else:
                rs = ShuffleSplit(n_splits=args.dataset.n_shuffle_splits, random_state=args.exp.seed,
                                  test_size=args.dataset.test_size)
            for split_ix, (train_index, test_index) in enumerate(rs.split(data_dict['cov_f'])):

                train_data_dict, test_data_dict = subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)
                train_test_splits.append((train_data_dict, test_data_dict))

        # Experiments
        for sub_ix, (train_data_dict, test_data_dict) in enumerate(train_test_splits):
            # Mlflow init
            experiment_name = f'convergence/{args.dataset.name}/{args.outcome_pfn.backbone}/{args.mp_wrapper.name}'

            mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                         tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

            mlflow_logger.log_hyperparams(args) if args.exp.logging else None

            # Initialisation of model
            args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix

            # ============================= Frequentist consistency =============================
            logger.info('Frequentist consistency check')
            mu_pfn = instantiate(args.outcome_pfn, args, _recursive_=False).fit(train_data_dict)
            prop_pfn = instantiate(args.prop_pfn, args, _recursive_=False).fit(train_data_dict)

            mu_0_err = np.sqrt(((test_data_dict['mu0'] - mu_pfn.get_posterior_predictive(test_data_dict, 0.0)) ** 2).mean())
            mu_1_err = np.sqrt(((test_data_dict['mu1'] - mu_pfn.get_posterior_predictive(test_data_dict, 1.0)) ** 2).mean())
            prop_error = np.sqrt(((prop_pfn.get_posterior_predictive(test_data_dict) - test_data_dict['prop']) ** 2).mean())
            r2_error = mu_0_err * prop_error + mu_1_err * prop_error
            freq_results = {'scaled_freq_mse_mu0': mu_0_err * n_1_4,
                            'scaled_freq_mse_mu1': mu_1_err * n_1_4,
                            'scaled_freq_mse_prop': prop_error * n_1_4,
                            'scaled_freq_r2': r2_error * (n_1_4 ** 2)}
            mlflow_logger.log_metrics(freq_results) if args.exp.logging else None
            logger.info(f'Frequentist results: {freq_results}')

            # ============================= Bayesian consistency =============================
            logger.info('Bayesian consistency check')
            mp_mu_pfn = instantiate(args.mp_wrapper, args, pfn_config=args.outcome_pfn, _recursive_=False)
            prop_mu_pfn = instantiate(args.mp_wrapper, args, pfn_config=args.prop_pfn, _recursive_=False)

            mu_posterior_sample, iterable = mp_mu_pfn.get_posterior_paths(train_data_dict, num_paths=args.exp.num_post_samples, nuisance='mu')
            prop_posterior_sample, iterable = prop_mu_pfn.get_posterior_paths(train_data_dict, num_paths=args.exp.num_post_samples, nuisance='prop')

            if iterable:
                mu_0_errs, mu_1_errs, prop_errs, r2_errors = [], [], [], []
                for mu_posterior, prop_posterior in tqdm(zip(mu_posterior_sample, prop_posterior_sample), total=args.exp.num_post_samples):
                    mu_0_errs.append(np.sqrt(((test_data_dict['mu0'] - mu_posterior.get_posterior_predictive(test_data_dict, 0.0)) ** 2).mean()))
                    mu_1_errs.append(np.sqrt(((test_data_dict['mu1'] - mu_posterior.get_posterior_predictive(test_data_dict, 1.0)) ** 2).mean()))
                    prop_errs.append(np.sqrt(((prop_posterior.get_posterior_predictive(test_data_dict) - test_data_dict['prop']) ** 2).mean()))
            else:  # vectorized
                mu_0_errs = np.sqrt(((mu_posterior_sample.get_posterior_predictive(test_data_dict, 0.0, n_samples=args.exp.num_post_samples) - test_data_dict['mu0'][:, None]) ** 2).mean(0))
                mu_1_errs = np.sqrt(((mu_posterior_sample.get_posterior_predictive(test_data_dict, 1.0, n_samples=args.exp.num_post_samples) - test_data_dict['mu1'][:, None]) ** 2).mean(0))
                prop_errs = np.sqrt(((prop_posterior_sample.get_posterior_predictive(test_data_dict, n_samples=args.exp.num_post_samples) - test_data_dict['prop'][:, None]) ** 2).mean(0))

            bayes_results = {'scaled_bayes_mse_mu0': max(mu_0_errs) * n_1_4,
                             'scaled_bayes_mse_mu1': max(mu_1_errs) * n_1_4,
                             'scaled_bayes_mse_prop': max(prop_errs) * n_1_4,
                             'scaled_bayes_r2': (max(mu_0_errs) + max(mu_1_errs)) * max(prop_errs) * (n_1_4 ** 2)}
            mlflow_logger.log_metrics(bayes_results) if args.exp.logging else None
            logger.info(f'Bayesian results: {bayes_results}')

            mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return freq_results, bayes_results

if __name__ == "__main__":
    main()