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
import torch.nn.functional as F
import torch.sparse as torch_sp
from util.pytorch import l2_loss, inner_product
from util.pytorch import pointwise_loss
from util.common import Reduction
from reckit import DataIterator, randint_choice
import numpy as np
from util.pytorch import get_initializer
from util.pytorch import sp_mat_to_sp_tensor, dropout_sparse


class _CDAE(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, dropout, hidden_act):
        super(_CDAE, self).__init__()

        # user and item embeddings
        self.en_embeddings = nn.Embedding(num_items, embed_dim)
        self.en_offset = nn.Parameter(torch.Tensor(embed_dim))
        self.de_embeddings = nn.Embedding(num_items, embed_dim)
        self.de_bias = nn.Embedding(num_items, 1)
        self.user_embeddings = nn.Embedding(num_users, embed_dim)

        self.dropout = dropout
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

    def forward(self, user_ids, bat_idx, sp_item_mat, bat_items):
        hidden = self._encoding(user_ids, sp_item_mat)  # (b,d)

        # decoding
        de_item_embs = self.de_embeddings(bat_items)  # (l,d)
        de_bias = self.de_bias(bat_items).squeeze()
        hidden = F.embedding(bat_idx, hidden)  # (l,d)

        ratings = inner_product(hidden, de_item_embs) + de_bias

        # reg loss
        bat_items = torch.unique(bat_items, sorted=False)
        reg_loss = l2_loss(self.en_embeddings(bat_items), self.en_offset,
                           self.user_embeddings(user_ids),
                           self.de_embeddings(bat_items), self.de_bias(bat_items))

        return ratings, reg_loss

    def _encoding(self, user_ids, sp_item_mat):

        corruption = dropout_sparse(sp_item_mat, 1-self.dropout, self.training)

        en_item_embs = self.en_embeddings.weight  # (n,d)
        hidden = torch_sp.mm(corruption, en_item_embs)  # (b,n)x(n,d)->(b,d)

        user_embs = self.user_embeddings(user_ids)  # (b,d)
        hidden += user_embs  # add user vector
        hidden += self.en_offset.view([1, -1])  # add bias
        hidden = self.hidden_act(hidden)  # hidden activate, z_u
        return hidden  # (b,d)

    def predict(self, user_ids, sp_item_mat):
        user_emb = self._encoding(user_ids, sp_item_mat)  # (b,d)
        ratings = user_emb.matmul(self.de_embeddings.weight.T)  # (b,d)x(d,n)->(b,n)
        ratings += self.de_bias.weight.view([1, -1])
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
        self.batch_size = config["batch_size"]
        self.param_init = config["param_init"]

        self.train_csr_mat = self.dataset.train_data.to_csr_matrix()
        self.train_csr_mat.data[:] = 1.0
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
        train_users = [user for user in range(self.num_users) if self.train_csr_mat[user].nnz]
        user_iter = DataIterator(train_users, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.cdae.train()
            for bat_users in user_iter:
                bat_sp_mat = self.train_csr_mat[bat_users]
                bat_items = []
                bat_labels = []
                bat_idx = []  # used to decoder
                for idx, _ in enumerate(bat_users):
                    pos_items = bat_sp_mat[idx].indices
                    neg_items = randint_choice(self.num_items, size=bat_sp_mat[idx].nnz*self.num_neg,
                                               replace=True, exclusion=pos_items)
                    neg_items = np.unique(neg_items)
                    bat_sp_mat[idx, neg_items] = 1

                    bat_items.append(pos_items)
                    bat_labels.append(np.ones_like(pos_items, dtype=np.float32))
                    bat_items.append(neg_items)
                    bat_labels.append(np.zeros_like(neg_items, dtype=np.float32))
                    bat_idx.append(np.full(len(pos_items)+len(neg_items), idx, dtype=np.int32))

                bat_items = np.concatenate(bat_items)
                bat_labels = np.concatenate(bat_labels)
                bat_idx = np.concatenate(bat_idx)
                bat_users = np.asarray(bat_users)

                bat_sp_mat = sp_mat_to_sp_tensor(bat_sp_mat).to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)

                bat_idx = torch.from_numpy(bat_idx).long().to(self.device)
                bat_users = torch.from_numpy(bat_users).long().to(self.device)

                hat_y, reg_loss = self.cdae(bat_users, bat_idx, bat_sp_mat, bat_items)

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
        user_ids = torch.from_numpy(np.asarray(users)).long().to(self.device)
        sp_item_mat = sp_mat_to_sp_tensor(self.train_csr_mat[users]).to(self.device)
        ratings = self.cdae.predict(user_ids, sp_item_mat).cpu().detach().numpy()
        return ratings
