"""
Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
Author: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["BPR"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
from util.pytorch_util import log_loss, inner_product, l2_loss
from data import PairwiseSampler
import numpy as np


class MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, device):
        super(MF, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embed_dim).to(device)
        self.item_embeddings = nn.Embedding(num_items, embed_dim).to(device)

        self.item_biases = nn.Embedding(num_items, 1).to(device)

        # weight initialization
        nn.init.uniform_(self.user_embeddings.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.item_embeddings.weight, a=-0.05, b=0.05)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user_ids, item_ids):
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        item_bias = self.item_biases(item_ids)
        ratings = inner_product(user_embs, item_embs) + torch.squeeze(item_bias)
        return ratings

    def predict(self, user_ids):
        user_embs = self.user_embeddings(user_ids)
        ratings = torch.matmul(user_embs, self.item_embeddings.weight.data.T)
        ratings += torch.squeeze(self.item_biases.weight.data)
        return ratings


class BPR(AbstractRecommender):
    def __init__(self, config):
        super(BPR, self).__init__(config)
        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mf = MF(self.num_users, self.num_items, self.factors_num, self.device)
        self.optimizer = torch.optim.Adam(self.mf.parameters(), lr=self.lr)

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.mf.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(np.array(bat_users)).type(torch.LongTensor).to(self.device)
                bat_pos_items = torch.from_numpy(np.array(bat_pos_items)).type(torch.LongTensor).to(self.device)
                bat_neg_items = torch.from_numpy(np.array(bat_neg_items)).type(torch.LongTensor).to(self.device)
                yui = self.mf(bat_users, bat_pos_items)
                yuj = self.mf(bat_users, bat_neg_items)
                loss = torch.sum(log_loss(yui-yuj))
                reg_loss = l2_loss(self.mf.user_embeddings(bat_users),
                                   self.mf.item_embeddings(bat_pos_items),
                                   self.mf.item_embeddings(bat_neg_items))
                loss += self.reg*reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.mf.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users, neg_items=None):
        users = torch.from_numpy(np.array(users)).type(torch.LongTensor).to(self.device)
        return self.mf.predict(users).cpu().detach().numpy()
