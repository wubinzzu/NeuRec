"""
Paper: Variational Autoencoders for Collaborative Filtering
Author: Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MultVAE"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pytorch import l2_loss
from reckit import DataIterator
from util.pytorch import get_initializer


class _MultVAE(nn.Module):
    def __init__(self, q_dims, p_dims, keep_prob):
        super(_MultVAE, self).__init__()

        # user and item embeddings
        self.layers_q = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(q_dims[:-1], q_dims[1:])):
            if i == len(q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance, respectively
                d_out *= 2
            self.layers_q.append(nn.Linear(d_in, d_out, bias=True))

        self.layers_p = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(p_dims[:-1], p_dims[1:])):
            self.layers_p.append(nn.Linear(d_in, d_out, bias=True))

        self.dropout = nn.Dropout(1-keep_prob)

        # weight initialization

        self.reg_params = [layer.weight for layer in self.layers_q] + \
                          [layer.weight for layer in self.layers_p]
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)

        for layer in self.layers_q:
            init(layer.weight)
            init(layer.bias)

        for layer in self.layers_p:
            init(layer.weight)
            init(layer.bias)

    def q_graph(self, input_x):
        mu_q, std_q, kl_dist = None, None, None
        h = F.normalize(input_x, p=2, dim=1)
        h = self.dropout(h)

        for i, layer in enumerate(self.layers_q):
            h = layer(h)
            if i != len(self.layers_q) - 1:
                h = F.tanh(h)
            else:
                size = int(h.shape[1] / 2)
                mu_q, logvar_q = torch.split(h, size, dim=1)
                std_q = torch.exp(0.5 * logvar_q)
                kl_dist = torch.sum(0.5*(-logvar_q + logvar_q.exp() + mu_q.pow(2) - 1), dim=1).mean()

        return mu_q, std_q, kl_dist

    def p_graph(self, z):
        h = z

        for i, layer in enumerate(self.layers_p):
            h = layer(h)
            if i != len(self.layers_p) - 1:
                h = F.tanh(h)

        return h

    def forward(self, input_x):
        # q-network
        mu_q, std_q, kl_dist = self.q_graph(input_x)
        epsilon = std_q.new_empty(std_q.shape)
        epsilon.normal_()
        sampled_z = mu_q + float(self.training)*epsilon*std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return logits, kl_dist

    def predict(self, input_x):
        ratings, _ = self.forward(input_x)

        return ratings


class MultVAE(AbstractRecommender):
    def __init__(self, config):
        super(MultVAE, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.keep_prob = config["keep_prob"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.anneal_cap = config["anneal_cap"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]

        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        p_dims = config["p_dims"]
        self.p_dims = p_dims + [self.num_items]
        if ("q_dims" not in config) or (config["q_dims"] is None):
            self.q_dims = self.p_dims[::-1]
        else:
            q_dims = config["q_dims"]
            q_dims = [self.num_items] + q_dims
            assert q_dims[0] == self.p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.multvae = _MultVAE(self.q_dims, self.p_dims, self.keep_prob).to(self.device)
        self.multvae.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.multvae.parameters(), lr=self.lr)

    def train_model(self):
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = DataIterator(train_users, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())

        update_count = 0.0
        for epoch in range(self.epochs):
            self.multvae.train()
            for bat_users in user_iter:
                bat_input = self.train_csr_mat[bat_users].toarray()
                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1.*update_count/self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap

                bat_input = torch.from_numpy(bat_input).float().to(self.device)

                logits, kl_dist = self.multvae(bat_input)
                log_softmax_var = F.log_softmax(logits, dim=-1)
                neg_ll = -torch.mul(log_softmax_var, bat_input).sum(dim=-1).mean()

                # apply regularization to weights
                reg_var = l2_loss(*self.multvae.reg_params)
                reg_var *= self.reg

                # l2 regularization multiply 0.5 to the l2 norm
                # multiply 2 so that it is back in the same scale
                neg_elbo = neg_ll + anneal*kl_dist + 2*reg_var

                self.optimizer.zero_grad()
                neg_elbo.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.multvae.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        bat_input = self.train_csr_mat[users].toarray()
        bat_input = torch.from_numpy(bat_input).float().to(self.device)
        ratings = self.multvae.predict(bat_input).cpu().detach().numpy()
        return ratings
