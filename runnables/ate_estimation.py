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
from src.models.ate_estimators import AIPTW, PosteriorATE
from scipy.stats import wasserstein_distance, Normal
import sys

from src.models.utils import subset_by_indices, ecdf, tv_hist, tv_kde

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

            args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
            logger.info(f'Fitting dataset {args.dataset.dataset_ix}')

            # Mlflow init
            if args.mp_wrapper is None:
                experiment_name = f'ate/{args.dataset.name}/{args.outcome_pfn.backbone}'
            else:
                experiment_name = f'ate/{args.dataset.name}/{args.outcome_pfn.backbone}/{args.mp_wrapper.name}'

            mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                         tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

            mlflow_logger.log_hyperparams(args) if args.exp.logging else None

            # Ground truth
            ate_gt = (np.concatenate([train_data_dict['mu1'], test_data_dict['mu1']]) -
                      np.concatenate([train_data_dict['mu0'], test_data_dict['mu0']])).mean()

            # ============================= Frequentist consistency =============================
            logger.info('Frequentist ATE estimation')

            # Initialisation of model
            mu_pfn = instantiate(args.outcome_pfn, args, _recursive_=False).fit(train_data_dict)
            prop_pfn = instantiate(args.prop_pfn, args, _recursive_=False).fit(train_data_dict)

            prop_pred = prop_pfn.get_posterior_predictive(test_data_dict)

            # Ground-truth frequentist asymptotic distribution
            aiptw = AIPTW(args.ate_estimation.q_trunc)
            ate_asymp_mean, ate_asymp_var = aiptw.get_mean_var(test_data_dict, test_data_dict['mu0'], test_data_dict['mu1'],
                                                               test_data_dict['prop'] if 'prop' in test_data_dict else prop_pfn.get_posterior_predictive(test_data_dict))
            ate_asymp_se = np.sqrt(ate_asymp_var / test_data_dict['mu1'].shape[0])


            if mu_pfn.kind == 'cate_pfn':
                # plugin estimator
                tau_pred = mu_pfn.get_posterior_predictive(test_data_dict)
                freq_results = {'ate_gt': ate_gt,
                                'ate_asymp': ate_asymp_mean,
                                'ate_asymp_mae': np.abs(ate_gt - ate_asymp_mean),
                                'ate_asymp_width_95':  2 * 1.96 * ate_asymp_se,
                                'ate_freq': tau_pred.mean(),
                                'ate_freq_mae': np.abs(ate_gt - tau_pred.mean())}
            else:
                # AIPTW estimator
                mu0_pred = mu_pfn.get_posterior_predictive(test_data_dict, 0.0)
                mu1_pred = mu_pfn.get_posterior_predictive(test_data_dict, 1.0)

                ate_aiptw_mean, ate_aiptw_var = aiptw.get_mean_var(test_data_dict, mu0_pred, mu1_pred, prop_pred)
                ate_aiptw_se = np.sqrt(ate_aiptw_var / test_data_dict['mu1'].shape[0])

                freq_results = {'ate_gt': ate_gt,
                                'ate_asymp': ate_asymp_mean,
                                'ate_asymp_mae': np.abs(ate_gt - ate_asymp_mean),
                                'ate_asymp_width_95':  2 * 1.96 * ate_asymp_se,
                                'ate_freq': ate_aiptw_mean,
                                'ate_freq_mae': np.abs(ate_gt - ate_aiptw_mean),
                                'ate_freq_width_95': 2 * 1.96 * ate_aiptw_se,
                                'ate_freq_gt_cdf': Normal(mu=ate_aiptw_mean, sigma=ate_aiptw_se).cdf(ate_gt)
                                }

            mlflow_logger.log_metrics(freq_results) if args.exp.logging else None
            logger.info(f'Frequentist results: {freq_results}')

            # ============================= Bayesian consistency =============================
            logger.info('Bayesian ATE estimation')
            bb_est = PosteriorATE(args.ate_estimation.q_trunc)

            # ============================= Naive sampling from posterior w/ marginal uncertainty =============================
            if args.mp_wrapper is None:

                if mu_pfn.kind == 'cate_pfn':
                    rand_q = np.random.uniform(size=args.exp.num_post_samples)
                    tau_pred = mu_pfn.get_posterior_predictive(test_data_dict, output_type='quantiles', quantiles=rand_q, kind='sampling')

                    ate_posterior_marg = bb_est.get_plugin_posterior_tau(tau_pred)

                else:
                    rand_q = np.random.uniform(size=args.exp.num_post_samples)
                    mu0_pred = mu_pfn.get_posterior_predictive(test_data_dict, 0.0, output_type='quantiles', quantiles=rand_q, kind='sampling')

                    rand_q = np.random.uniform(size=args.exp.num_post_samples)
                    mu1_pred = mu_pfn.get_posterior_predictive(test_data_dict, 1.0, output_type='quantiles', quantiles=rand_q, kind='sampling')

                    mu0_pred, mu1_pred = np.array(mu0_pred), np.array(mu1_pred)

                    ate_posterior_marg = bb_est.get_plugin_posterior(mu0_pred, mu1_pred)

                bayes_results = {'ate_bayes_marg_mean': ate_posterior_marg.mean(),
                                 'ate_bayes_marg_mae': np.abs(ate_posterior_marg.mean() - ate_gt),
                                 'ate_bayes_marg_width_95': np.quantile(ate_posterior_marg, 0.975) -
                                                              np.quantile(ate_posterior_marg, 0.025),
                                 'ate_bayes_marg_cdf_gt': ecdf(ate_posterior_marg)(ate_gt)}

            # ============================= Proper sample from posterior w/ martingale posteriors =============================
            else:
                mp_mu_pfn = instantiate(args.mp_wrapper, args, pfn_config=args.outcome_pfn, _recursive_=False)
                prop_mu_pfn = instantiate(args.mp_wrapper, args, pfn_config=args.prop_pfn, _recursive_=False)

                mu_posterior_sample, iterable = mp_mu_pfn.get_posterior_paths(train_data_dict, num_paths=args.exp.num_post_samples, nuisance='mu')
                prop_posterior_sample, iterable = prop_mu_pfn.get_posterior_paths(train_data_dict, num_paths=args.exp.num_post_samples, nuisance='prop')

                if iterable:
                    mu0_pred, mu1_pred, prop_pred = [], [], []
                    for mu_posterior, prop_posterior in tqdm(zip(mu_posterior_sample, prop_posterior_sample),
                                                             total=args.exp.num_post_samples):
                        mu0_pred.append(mu_posterior.get_posterior_predictive(test_data_dict, 0.0))
                        mu1_pred.append(mu_posterior.get_posterior_predictive(test_data_dict, 1.0))
                        prop_pred.append(prop_posterior.get_posterior_predictive(test_data_dict))

                    mu0_pred, mu1_pred, prop_pred = np.array(mu0_pred), np.array(mu1_pred), np.array(prop_pred)

                else:  # vectorized
                    prop_pred = prop_posterior_sample.get_posterior_predictive(test_data_dict, n_samples=args.exp.num_post_samples).T
                    mu0_pred = mu_posterior_sample.get_posterior_predictive(test_data_dict, 0.0, n_samples=args.exp.num_post_samples).T
                    mu1_pred = mu_posterior_sample.get_posterior_predictive(test_data_dict, 1.0, n_samples=args.exp.num_post_samples).T

                ate_posterior_plugin = bb_est.get_plugin_posterior(mu0_pred, mu1_pred)
                ate_posterior_one_step = bb_est.get_one_step_posterior(test_data_dict, mu0_pred, mu1_pred, prop_pred)

                bayes_results = {'ate_bayes_plugin_mean': ate_posterior_plugin.mean(),
                                 'ate_bayes_plugin_mae': np.abs(ate_posterior_plugin.mean() - ate_gt),
                                 'ate_bayes_plugin_width_95': np.quantile(ate_posterior_plugin, 0.975) -
                                                              np.quantile(ate_posterior_plugin, 0.025),
                                 'ate_bayes_plugin_cdf_gt': ecdf(ate_posterior_plugin)(ate_gt),
                                 'ate_bayes_one_step_mean': ate_posterior_one_step.mean(),
                                 'ate_bayes_one_step_mae': np.abs(ate_posterior_one_step.mean() - ate_gt),
                                 'ate_bayes_one_step_width_95': np.quantile(ate_posterior_one_step, 0.975) -
                                                              np.quantile(ate_posterior_one_step, 0.025),
                                 'ate_bayes_one_step_cdf_gt': ecdf(ate_posterior_one_step)(ate_gt)
                                 }

            mlflow_logger.log_metrics(bayes_results) if args.exp.logging else None
            logger.info(f'Bayesian results: {bayes_results}')


            # ============================= Bernstein von-Mises =============================
            ate_asymp_sample = np.random.normal(loc=ate_gt, scale=ate_asymp_se, size=args.exp.num_post_samples)

            if args.mp_wrapper is None:
                assert ate_posterior_marg.shape == ate_asymp_sample.shape

                if mu_pfn.kind == 'outcome_pfn':
                    ate_aiptw_sample = np.random.normal(loc=ate_aiptw_mean, scale=ate_aiptw_se, size=args.exp.num_post_samples)
                    results = {'bvm_wass_marg': wasserstein_distance(ate_asymp_sample, ate_posterior_marg),
                               'bvm_tv_marg': tv_kde(ate_asymp_sample, ate_posterior_marg),
                               'bvm_wass_aiptw': wasserstein_distance(ate_asymp_sample, ate_aiptw_sample),
                               'bvm_tv_aiptw': tv_kde(ate_asymp_sample, ate_aiptw_sample),
                               }
                else:
                    results = {'bvm_wass_marg': wasserstein_distance(ate_asymp_sample, ate_posterior_marg),
                               'bvm_tv_marg': tv_kde(ate_asymp_sample, ate_posterior_marg)
                               }
            else:
                assert ate_posterior_one_step.shape == ate_posterior_plugin.shape == ate_asymp_sample.shape
                results = {'bvm_wass_plugin': wasserstein_distance(ate_asymp_sample, ate_posterior_plugin),
                           'bvm_wass_one_step': wasserstein_distance(ate_asymp_sample, ate_posterior_one_step),
                           'bvm_tv_plugin': tv_kde(ate_asymp_sample, ate_posterior_plugin),
                           'bvm_tv_one_step': tv_kde(ate_asymp_sample, ate_posterior_one_step)}

            mlflow_logger.log_metrics(results) if args.exp.logging else None
            logger.info(f'Bernstein von-Mises results: {results}')

            mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return freq_results, bayes_results, results

if __name__ == "__main__":
    main()