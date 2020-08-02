"""
Paper: Translation-based Recommendation
Author: Ruining He, Wang-Cheng Kang, and Julian McAuley
Reference: https://drive.google.com/file/d/0B9Ck8jw-TZUEVmdROWZKTy1fcEE/view?usp=sharing
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["TransRec"]


from model.base import AbstractRecommender
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import l2_distance, l2_loss
from util.common import Reduction
from util.pytorch import get_initializer
from data import TimeOrderPairwiseSampler, TimeOrderPointwiseSampler


class _TransRec(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super(_TransRec, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.global_transition = Parameter(torch.Tensor(1, embed_dim))

        self.item_biases = nn.Embedding(num_items, 1)

        # weight initialization
        self.reset_parameters("uniform")

    def reset_parameters(self, init_method):
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        zero_init(self.user_embeddings.weight)
        init(self.global_transition)
        init(self.item_embeddings.weight)
        zero_init(self.item_biases.weight)

    def forward(self, user_ids, last_items, pre_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        pre_item_embs = self.item_embeddings(pre_items)
        pre_item_bias = self.item_biases(pre_items)

        transed_emb = user_embs + self.global_transition + last_item_embs
        hat_y = -l2_distance(transed_emb, pre_item_embs) + torch.squeeze(pre_item_bias)

        return hat_y

    def predict(self, user_ids, last_items):
        user_embs = self.user_embeddings(user_ids)
        last_item_embs = self.item_embeddings(last_items)
        transed_emb = user_embs + self.global_transition + last_item_embs
        ratings = -l2_distance(transed_emb.unsqueeze(dim=1), self.item_embeddings.weight)

        ratings += torch.squeeze(self.item_biases.weight)
        return ratings


class TransRec(AbstractRecommender):
    def __init__(self, config):
        super(TransRec, self).__init__(config)
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
        self.transrec = _TransRec(self.num_users, self.num_items, self.emb_size).to(self.device)
        self.transrec.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.transrec.parameters(), lr=self.lr)

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
            self.transrec.train()
            for bat_users, bat_last_items, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(np.array(bat_users)).long().to(self.device)
                bat_last_items = torch.from_numpy(np.array(bat_last_items)).long().to(self.device)
                bat_pos_items = torch.from_numpy(np.array(bat_pos_items)).long().to(self.device)
                bat_neg_items = torch.from_numpy(np.array(bat_neg_items)).long().to(self.device)
                yui = self.transrec(bat_users, bat_last_items, bat_pos_items)
                yuj = self.transrec(bat_users, bat_last_items, bat_neg_items)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.transrec.user_embeddings(bat_users),
                                   self.transrec.global_transition,
                                   self.transrec.item_embeddings(bat_last_items),
                                   self.transrec.item_embeddings(bat_pos_items),
                                   self.transrec.item_embeddings(bat_neg_items),
                                   self.transrec.item_biases(bat_pos_items),
                                   self.transrec.item_biases(bat_neg_items)
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
            self.transrec.train()
            for bat_users, bat_last_items, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(np.array(bat_users)).long().to(self.device)
                bat_last_items = torch.from_numpy(np.array(bat_last_items)).long().to(self.device)
                bat_items = torch.from_numpy(np.array(bat_items)).long().to(self.device)
                bat_labels = torch.from_numpy(np.array(bat_labels)).float().to(self.device)
                yui = self.transrec(bat_users, bat_last_items, bat_items)

                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.SUM)
                reg_loss = l2_loss(self.transrec.user_embeddings(bat_users),
                                   self.transrec.global_transition,
                                   self.transrec.item_embeddings(bat_last_items),
                                   self.transrec.item_embeddings(bat_items),
                                   self.transrec.item_biases(bat_items)
                                   )
                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.transrec.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users, neg_items=None):
        last_items = [self.user_pos_dict[u][-1] for u in users]
        users = torch.from_numpy(np.array(users)).long().to(self.device)
        last_items = torch.from_numpy(np.array(last_items)).long().to(self.device)
        return self.transrec.predict(users, last_items).cpu().detach().numpy()
