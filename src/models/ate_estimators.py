import numpy as np


def get_iptw(treat_f, prop_pred, q_trunc):
    ipwt0 = ((treat_f == 0.0) & ((1 - prop_pred) >= q_trunc)).astype('float') / (1 - prop_pred + 1e-9)
    ipwt1 = ((treat_f == 1.0) & (prop_pred >= q_trunc)).astype('float') / (prop_pred + 1e-9)
    return ipwt0, ipwt1


class AIPTW(object):
    def __init__(self, q_trunc):
        self.q_trunc = q_trunc

    def get_mean_var(self, test_data_dict, mu0_pred, mu1_pred, prop_pred):
        treat_f, out_f = test_data_dict['treat_f'], test_data_dict['out_f']
        ipwt0, ipwt1 = get_iptw(treat_f, prop_pred, self.q_trunc)
        pseudo_out = ipwt1 * (out_f - mu1_pred) + mu1_pred - (ipwt0 * (out_f - mu0_pred) + mu0_pred)
        return pseudo_out.mean(), pseudo_out.var()


class PosteriorATE(object):
    def __init__(self, q_trunc):
        self.q_trunc = q_trunc

    def get_plugin_posterior(self, mu0_pred, mu1_pred):
        tau_pred = mu1_pred - mu0_pred
        w = np.random.dirichlet(np.ones((tau_pred.shape[1],)), tau_pred.shape[0])
        return (w * tau_pred).sum(1)

    def get_plugin_posterior_tau(self, tau_pred):
        w = np.random.dirichlet(np.ones((tau_pred.shape[1],)), tau_pred.shape[0])
        return (w * tau_pred).sum(1)

    def get_one_step_posterior(self, test_data_dict, mu0_pred, mu1_pred, prop_pred):
        treat_f, out_f = test_data_dict['treat_f'], test_data_dict['out_f']
        ipwt0, ipwt1 = get_iptw(treat_f, prop_pred, self.q_trunc)
        pseudo_out = ipwt1 * (out_f - mu1_pred) + mu1_pred - (ipwt0 * (out_f - mu0_pred) + mu0_pred)

        w = np.random.dirichlet(np.ones((mu0_pred.shape[1],)), mu0_pred.shape[0])
        return (w * pseudo_out).sum(1)

