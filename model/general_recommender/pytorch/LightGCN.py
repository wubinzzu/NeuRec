"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["LightGCN"]

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSampler, PairwiseSampler
import numpy as np
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix
from util.pytorch import sp_mat_to_sp_tensor


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

    def forward(self, users, items):
        user_embeddings, item_embeddings = self._forward_gcn()
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        ratings = inner_product(user_embs, item_embs)
        return ratings

    def _forward_gcn(self):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn()


class LightGCN(AbstractRecommender):
    def __init__(self, config):
        super(LightGCN, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.adj_type = config["adj_type"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat(self.adj_type)
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        self.lightgcn.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, adj_type):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalize_adj_matrix(adj_mat + sp.eye(adj_mat.shape[0]), norm_method="left")
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="left")
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalize_adj_matrix(adj_mat, norm_method="left")
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

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
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                _bat_users = torch.cat([bat_users, bat_users], dim=0)
                _bat_items = torch.cat([bat_pos_items, bat_neg_items], dim=0)

                hat_y = self.lightgcn(_bat_users, _bat_items)
                yui, yuj = torch.split(hat_y, [len(bat_pos_items), len(bat_neg_items)], dim=0)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.MEAN)
                reg_loss = l2_loss(self.lightgcn.user_embeddings(bat_users),
                                   self.lightgcn.item_embeddings(bat_pos_items),
                                   self.lightgcn.item_embeddings(bat_neg_items)
                                   )
                loss += self.reg * reg_loss/self.batch_size
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
            self.lightgcn.train()
            for bat_users, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.lightgcn(bat_users, bat_items)
                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.MEAN)
                reg_loss = l2_loss(self.lightgcn.user_embeddings(bat_users),
                                   self.lightgcn.item_embeddings(bat_items)
                                   )
                loss += self.reg * reg_loss/self.emb_size
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    # @timer
    def evaluate_model(self):
        self.lightgcn.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
