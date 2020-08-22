"""
Paper: Collaborative Denoising Auto-Encoder for Top-N Recommender Systems
Author: Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["CDAE"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
from util.pytorch import l2_loss
from util.pytorch import pointwise_loss
from util.common import Reduction
from reckit import DataIterator, randint_choice
import numpy as np
from util.pytorch import get_initializer


class _CDAE(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, dropout, hidden_act):
        super(_CDAE, self).__init__()

        # user and item embeddings
        self.en_embeddings = nn.Embedding(num_items, embed_dim)
        self.en_offset = nn.Parameter(torch.Tensor(embed_dim))
        self.de_embeddings = nn.Embedding(num_items, embed_dim)
        self.de_bias = nn.Embedding(num_items, 1)
        self.user_embeddings = nn.Embedding(num_users, embed_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_act = hidden_act

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        init(self.en_embeddings.weight)
        zero_init(self.en_offset)

        init(self.de_embeddings.weight)
        zero_init(self.de_bias.weight)

        init(self.user_embeddings.weight)

    def forward(self, user_id, item_ids):
        hidden, reg_loss = self._encoding(user_id, item_ids)  # (1,d)

        # decoding
        de_item_embs = self.de_embeddings(item_ids)  # (l,d)
        ratings = hidden.matmul(de_item_embs.T).flatten()  # (1,d)x(d,l)->(1,l)->(l)

        de_bias = self.de_bias(item_ids).flatten()
        ratings += de_bias  # add bias

        reg_loss += l2_loss(de_bias, de_item_embs)

        return ratings, reg_loss

    def _encoding(self, user_id, item_ids):
        ones = item_ids.new_ones(item_ids.shape, dtype=torch.float)
        corruption = self.dropout(ones).view([-1, 1])  # (l,1), noise

        en_item_embs = self.en_embeddings(item_ids)  # (l,d)
        hidden = en_item_embs.T.matmul(corruption).flatten()  # (d,l)x(l,1)->(d,1)->(d,)

        user_embs = self.user_embeddings(user_id).flatten()  # (d,)
        hidden += user_embs  # add user vector
        hidden += self.en_offset  # add bias
        hidden = self.hidden_act(hidden)  # hidden activate, z_u

        # reg_loss
        weights = torch.mul(corruption.bool().float(), en_item_embs)
        reg_loss = l2_loss(weights, self.en_offset)

        return hidden.view([1, -1]), reg_loss  # (1,d)

    def predict(self, user_id, item_ids):
        user_emb, _ = self._encoding(user_id, item_ids)
        ratings = user_emb.matmul(self.de_embeddings.weight.T).flatten()
        ratings += self.de_bias.weight.flatten()
        return ratings


class CDAE(AbstractRecommender):
    def __init__(self, config):
        super(CDAE, self).__init__(config)
        self.emb_size = config["hidden_dim"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.dropout = config["dropout"]
        self.hidden_act = config["hidden_act"]
        self.loss_func = config["loss_func"]
        self.num_neg = config["num_neg"]
        self.epochs = config["epochs"]
        self.param_init = config["param_init"]

        self.user_pos_train = self.dataset.train_data.to_user_dict()
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.hidden_act == "identity":
            hidden_act = nn.Identity()
        elif self.hidden_act == "sigmoid":
            hidden_act = nn.Sigmoid()
        else:
            raise ValueError(f"hidden activate function {self.hidden_act} is invalid.")

        self.cdae = _CDAE(self.num_users, self.num_items, self.emb_size, self.dropout, hidden_act).to(self.device)
        self.cdae.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.cdae.parameters(), lr=self.lr)

    def train_model(self):
        # result = self.evaluate_model()
        train_users = list(self.user_pos_train.keys())
        user_iter = DataIterator(train_users, batch_size=1, shuffle=True)

        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.cdae.train()
            for user in user_iter:
                user = user[0]
                pos_items = self.user_pos_train[user]
                neg_items = randint_choice(self.num_items, size=len(pos_items)*self.num_neg,
                                           replace=True, exclusion=pos_items)
                pos_items = torch.from_numpy(pos_items).long().to(self.device)
                neg_items = torch.from_numpy(neg_items).long().to(self.device)
                pos_labels = torch.ones_like(pos_items).float().to(self.device)
                neg_labels = torch.zeros_like(neg_items).float().to(self.device)

                bat_items = torch.cat([pos_items, neg_items])
                bat_labels = torch.cat([pos_labels, neg_labels])
                user = torch.scalar_tensor(user).long().to(self.device)
                hat_y, reg_loss = self.cdae(user, bat_items)

                loss = pointwise_loss(self.loss_func, hat_y, bat_labels, reduction=Reduction.SUM)

                loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.cdae.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        all_ratings = []
        for user in users:
            pos_items = self.user_pos_train[user]
            pos_items = torch.from_numpy(pos_items).long().to(self.device)
            user = torch.scalar_tensor(user).long().to(self.device)
            ratings = self.cdae.predict(user, pos_items).cpu().detach().numpy()
            all_ratings.append(ratings)
        return np.asarray(all_ratings)
