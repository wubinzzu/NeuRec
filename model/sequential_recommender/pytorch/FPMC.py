"""
Paper: Factorizing Personalized Markov Chains for Next-Basket Recommendation
Author: Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["FPMC"]


from model.base import AbstractRecommender
import torch
import torch.nn as nn
import numpy as np
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import inner_product, l2_loss
from util.common import Reduction
from util.pytorch import get_initializer
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler


class _FPMC(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_FPMC, self).__init__()

        # user and item embeddings
        self.UI_embeddings = nn.Embedding(num_users, embed_dim)
        self.IU_embeddings = nn.Embedding(num_items, embed_dim)
        self.IL_embeddings = nn.Embedding(num_items, embed_dim)
        self.LI_embeddings = nn.Embedding(num_items, embed_dim)

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.UI_embeddings.weight)
        init(self.IU_embeddings.weight)
        init(self.IL_embeddings.weight)
        init(self.LI_embeddings.weight)

    def forward(self, user_ids, last_items, pre_items):
        ui_emb = self.UI_embeddings(user_ids)  # b*d
        pre_iu_emb = self.IU_embeddings(pre_items)  # b*d
        pre_il_emb = self.IL_embeddings(pre_items)  # b*d
        last_emb = self.LI_embeddings(last_items)  # b*d

        hat_y = inner_product(ui_emb, pre_iu_emb) + inner_product(last_emb, pre_il_emb)

        return hat_y

    def predict(self, user_ids, last_items):
        ui_emb = self.UI_embeddings(user_ids)  # b*d
        last_emb = self.LI_embeddings(last_items)  # b*d
        ratings = torch.matmul(ui_emb, self.IU_embeddings.weight.T) + \
                  torch.matmul(last_emb, self.IL_embeddings.weight.T)

        return ratings


class FPMC(AbstractRecommender):
    def __init__(self, config):
        super(FPMC, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.emb_size = config["embedding_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_dict = self.dataset.train_data.to_user_dict(by_time=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fpmc = _FPMC(self.num_users, self.num_items, self.emb_size).to(self.device)
        self.fpmc.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.fpmc.parameters(), lr=self.lr)

    def train_model(self):
        if self.is_pairwise:
            self._train_pairwise()
        else:
            self._train_pointwise()

    def _train_pairwise(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data,
                                             len_seqs=1, len_next=1, num_neg=1,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.fpmc.train()
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                yui = self.fpmc(bat_users, bat_last_items, bat_pos_items)
                yuj = self.fpmc(bat_users, bat_last_items, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.fpmc.UI_embeddings(bat_users),
                                   self.fpmc.LI_embeddings(bat_last_items),
                                   self.fpmc.IU_embeddings(bat_pos_items),
                                   self.fpmc.IU_embeddings(bat_neg_items),
                                   self.fpmc.IL_embeddings(bat_pos_items),
                                   self.fpmc.IL_embeddings(bat_neg_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _train_pointwise(self):
        data_iter = TimeOrderPointwiseSampler(self.dataset.train_data,
                                              len_seqs=1, len_next=1, num_neg=1,
                                              batch_size=self.batch_size,
                                              shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.fpmc.train()
            for bat_users, bat_last_items, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_last_items = torch.from_numpy(bat_last_items).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.fpmc(bat_users, bat_last_items, bat_items)

                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.fpmc.UI_embeddings(bat_users),
                                   self.fpmc.LI_embeddings(bat_last_items),
                                   self.fpmc.IU_embeddings(bat_items),
                                   self.fpmc.IL_embeddings(bat_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.fpmc.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        last_items = torch.from_numpy(np.asarray(last_items)).long().to(self.device)
        return self.fpmc.predict(users, last_items).cpu().detach().numpy()
