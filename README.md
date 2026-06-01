Frequentist Consistency of PFNs for Causal Inference 
==============================

[![arXiv](https://img.shields.io/badge/arXiv-2603.12037-b31b1b.svg)](https://arxiv.org/abs/2603.12037)

Restoring frequentist consistency of PFNs with one-step posterior corrections & martingale posteriors

<img width="2196" height="599" alt="image" src="https://github.com/user-attachments/assets/63bcd4cf-56e7-49bb-92f3-7eae791628d1" />

The project is built with the following Python libraries:
1. [PyTorch](https://pytorch.org/)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

## Setup

### Installations
First, one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### MlFlow Setup / Connection
To start an experiment server, run: 

`mlflow server --port=5000 --gunicorn-opts "--timeout 280"`

To access the MLFlow web UI with all the experiments, connect via SSH:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

### Semi-synthetic datasets setup

Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition): to `data/acic2016/`
```
 ── data/acic_2016
    ├── synth_outcomes
    |   ├── zymu_<id0>.csv   
    |   ├── ... 
    │   └── zymu_<id14>.csv 
    ├── ids.csv
    └── x.csv 
```

## Experiments

There are two main scripts for different PFNs and datasets (`runnables/check_convergence.py` and `runnables/ate_estimation.py`), and each requires some mandatory arguments. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `config/` folder.

Generic scripts with logging and fixed random seed are as follows:
```console
PYTHONPATH=.  python3 runnables/check_convergence.py +dataset=<dataset> +mp_wrapper=<mp_wrapper> outcome_pfn.backbone=<pfn> exp.seed=10
```
or
```console
PYTHONPATH=.  python3 runnables/ate_estimation.py +dataset=<dataset> +mp_wrapper=<mp_wrapper> outcome_pfn.backbone=<pfn> exp.seed=10
```

### Datasets
One needs to specify a dataset/dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=1000`, or a subset index for ACIC 2016 data `dataset.dataset_ix=0`):
- Synthetic data (adapted from https://proceedings.mlr.press/v130/curth21a/curth21a.pdf): `+dataset=synthetic`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp` 
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition) dataset: `+dataset=acic2016`

### Prior-data fitted networks
We employed the following PFNs for the outcome model:
- [TabPFN](https://arxiv.org/abs/2207.01848): `outcome_pfn.backbone=tabpfn` with two varaints (S-learner `outcome_pfn.learner_type=s` and T-learner `outcome_pfn.learner_type=t`)
- [CausalPFN](https://arxiv.org/abs/2506.07918): `outcome_pfn.backbone=causalpfn`
- [CausalFM](https://arxiv.org/abs/2506.10914): `outcome_pfn.backbone=causalfm`
For the propensity model, only the TabPFN can be used: `prop_pfn.backbone=tabpfn`.

### MP wrapper
The following MP-wrappers are available:
- **PFN-only MPs** (https://arxiv.org/abs/2510.25154): `+mp_wrapper=pfns`
- **Copula-based MPs** (https://arxiv.org/abs/2505.11325): `+mp_wrapper=copulas` with different variants:
    - x-independent: `mp_wrapper.mode=independent`
    - x-parallel: `mp_wrapper.mode=parallel`
    - smooth: `mp_wrapper.mode=natural`

### Examples
Example of $L_2$-concentration check for the synthetic data with TabPFN + Copula-based MPs:
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/check_convergence.py -m +dataset=synthetic +mp_wrapper=copulas dataset.n_samples_train=100 dataset.n_samples_test=1000 exp.seed=10
```

Example of an ATE estimation experiment for the IHDP dataset with CausalPFN:
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/ate_estimation.py -m +dataset=ihdp outcome_pfn.backbone=causalpfn exp.seed=10 outcome_pfn.temp=0.5,0.1
```
