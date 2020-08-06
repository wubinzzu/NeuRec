"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["MF"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
from util.pytorch import inner_product, l2_loss
from util.pytorch import pairwise_loss, pointwise_loss
from util.common import Reduction
from data import PairwiseSampler, PointwiseSampler
import numpy as np
from util.pytorch import get_initializer


class _MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_MF, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)
        zero_init(self.item_biases.weight)

    def forward(self, user_ids, item_ids):
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        item_bias = self.item_biases(item_ids)
        ratings = inner_product(user_embs, item_embs) + torch.squeeze(item_bias)
        return ratings

    def predict(self, user_ids):
        user_embs = self.user_embeddings(user_ids)
        ratings = torch.matmul(user_embs, self.item_embeddings.weight.T)
        ratings += torch.squeeze(self.item_biases.weight)
        return ratings


class MF(AbstractRecommender):
    def __init__(self, config):
        super(MF, self).__init__(config)
        self.emb_size = config["embedding_size"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mf = _MF(self.num_users, self.num_items, self.emb_size).to(self.device)
        self.mf.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.mf.parameters(), lr=self.lr)

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.mf.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.mf(bat_users, bat_pos_items)
                yuj = self.mf(bat_users, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.mf.user_embeddings(bat_users),
                                   self.mf.item_embeddings(bat_pos_items),
                                   self.mf.item_embeddings(bat_neg_items),
                                   self.mf.item_biases(bat_pos_items),
                                   self.mf.item_biases(bat_neg_items))
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = PointwiseSampler(self.dataset.train_data, num_neg=1,
                                     batch_size=self.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.mf.train()
            for bat_users, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.mf(bat_users, bat_items)
                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.mf.user_embeddings(bat_users),
                                   self.mf.item_embeddings(bat_items),
                                   self.mf.item_biases(bat_items))
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.mf.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.mf.predict(users).cpu().detach().numpy()
