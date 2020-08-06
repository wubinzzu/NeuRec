"""
Paper: FISM: Factored Item Similarity Models for Top-N Recommender Systems
Author: Santosh Kabbur, Xia Ning, and George Karypis
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["FISM"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
from util.pytorch import inner_product, l2_loss
from util.pytorch import pairwise_loss, pointwise_loss
from util.common import Reduction
from data import FISMPairwiseSampler, FISMPointwiseSampler
import numpy as np
from util.pytorch import get_initializer
from reckit import pad_sequences


class _FISM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, alpha, item_pad_idx=None):
        super(_FISM, self).__init__()

        # user and item embeddings
        self.alpha = alpha
        self._pad_idx = item_pad_idx
        self.his_embeddings = nn.Embedding(num_items, embed_dim, padding_idx=item_pad_idx)
        self.item_embeddings = nn.Embedding(num_items, embed_dim, padding_idx=item_pad_idx)

        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        init(self.his_embeddings.weight)
        init(self.item_embeddings.weight)
        if self._pad_idx is not None:
            zero_init(self.his_embeddings.weight[self._pad_idx])
            zero_init(self.item_embeddings.weight[self._pad_idx])

        zero_init(self.user_biases.weight)
        zero_init(self.item_biases.weight)

    def _forward_user(self, his_items, his_lens):
        scale = his_lens.pow(-self.alpha).unsqueeze(dim=1)
        his_embs = self.his_embeddings(his_items)
        user_emb = his_embs.sum(dim=1)
        return torch.mul(user_emb, scale)

    def forward(self, users, his_items, his_lens, pre_items):
        user_embs = self._forward_user(his_items, his_lens)
        user_bias = self.user_biases(users)

        item_embs = self.item_embeddings(pre_items)
        item_bias = self.item_biases(pre_items)

        ratings = inner_product(user_embs, item_embs) + user_bias.squeeze() + item_bias.squeeze()
        return ratings

    def predict(self, his_items, his_lens):
        user_embs = self._forward_user(his_items, his_lens)

        ratings = torch.matmul(user_embs, self.item_embeddings.weight.T)
        ratings += self.item_biases.weight.squeeze()
        return ratings


class FISM(AbstractRecommender):
    def __init__(self, config):
        super(FISM, self).__init__(config)
        self.emb_size = config["embedding_size"]
        self.lr = config["lr"]
        self.alpha = config["alpha"]
        self.reg_lambda = config["reg_lambda"]
        self.reg_gamma = config["reg_gamma"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.pad_idx = self.num_items
        self.num_items += 1
        self.user_train_dict = self.dataset.train_data.to_user_dict()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.fism = _FISM(self.num_users, self.num_items, self.emb_size, self.alpha, self.pad_idx).to(self.device)
        self.fism.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.fism.parameters(), lr=self.lr)

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = FISMPairwiseSampler(self.dataset.train_data, pad=self.pad_idx,
                                        batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.fism.train()
            for bat_users, bat_his_items, bat_his_len, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_his_items = torch.from_numpy(bat_his_items).long().to(self.device)
                bat_his_len = torch.from_numpy(bat_his_len).float().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                yui = self.fism(bat_users, bat_his_items, bat_his_len, bat_pos_items)
                yuj = self.fism(bat_users, bat_his_items, bat_his_len, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                emb_reg = l2_loss(self.fism.his_embeddings(bat_his_items),
                                  self.fism.item_embeddings(bat_pos_items),
                                  self.fism.item_embeddings(bat_neg_items)
                                  )
                bias_reg = l2_loss(self.fism.item_biases(bat_pos_items),
                                   self.fism.item_biases(bat_neg_items)
                                   )

                loss += self.reg_lambda*emb_reg + self.reg_gamma*bias_reg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = FISMPointwiseSampler(self.dataset.train_data, pad=self.pad_idx,
                                         batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.fism.train()
            for bat_users, bat_his_items, bat_his_len, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_his_items = torch.from_numpy(bat_his_items).long().to(self.device)
                bat_his_len = torch.from_numpy(bat_his_len).float().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)

                yui = self.fism(bat_users, bat_his_items, bat_his_len, bat_items)
                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                emb_reg = l2_loss(self.fism.his_embeddings(bat_his_items),
                                  self.fism.item_embeddings(bat_items)
                                  )
                bias_reg = l2_loss(self.fism.item_biases(bat_items),
                                   self.fism.user_biases(bat_users))

                loss += self.reg_lambda*emb_reg + self.reg_gamma*bias_reg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.fism.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        his_items = [self.user_train_dict[u] for u in users]
        his_len = [len(self.user_train_dict[u]) for u in users]
        his_items = pad_sequences(his_items, value=self.pad_idx, max_len=None,
                                  padding='post', truncating='post', dtype=np.int32)

        his_items = torch.from_numpy(np.asarray(his_items)).long().to(self.device)
        his_len = torch.from_numpy(np.asarray(his_len)).float().to(self.device)

        return self.fism.predict(his_items, his_len).cpu().detach().numpy()
