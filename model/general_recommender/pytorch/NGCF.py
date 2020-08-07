"""
Paper: Neural Graph Collaborative Filtering
Author: Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua
Reference: https://github.com/xiangwang1223/neural_graph_collaborative_filtering
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["NGCF"]

import torch
import torch.sparse as torch_sp
import torch.nn as nn
from torch.nn import Parameter
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


class _NGCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, layer_size,
                 node_dropout_flag, node_dropout, mess_dropout):
        super(_NGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.layer_size = layer_size
        self.n_layers = len(layer_size)
        self.node_dropout_flag = node_dropout_flag
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.user_embeddings_final = None
        self.item_embeddings_final = None

        self._create_parameters()
        # weight initialization
        self.reset_parameters()

    def _create_parameters(self):
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.weights = dict()
        w_sizes = [self.embed_dim] + self.layer_size
        for k in range(self.n_layers):
            self.weights['W_gc_%d' % k] = Parameter(torch.Tensor(w_sizes[k], w_sizes[k + 1]))
            self.weights['b_gc_%d' % k] = Parameter(torch.Tensor(1, w_sizes[k + 1]))

            self.weights['W_bi_%d' % k] = Parameter(torch.Tensor(w_sizes[k], w_sizes[k + 1]))
            self.weights['b_bi_%d' % k] = Parameter(torch.Tensor(1, w_sizes[k + 1]))

            self.weights['W_mlp_%d' % k] = Parameter(torch.Tensor(w_sizes[k], w_sizes[k + 1]))
            self.weights['b_mlp_%d' % k] = Parameter(torch.Tensor(1, w_sizes[k + 1]))

        for name, param in self.weights.items():
            self.register_parameter(name=name, param=param)

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)
        for key in self.weights.keys():
            init(self.weights[key])

    def forward(self, users, items):
        # user_embeddings, item_embeddings = self._forward_gcn()
        self.user_embeddings_final, self.item_embeddings_final = self._forward_gcn()
        user_embs = F.embedding(users, self.user_embeddings_final)
        item_embs = F.embedding(items, self.item_embeddings_final)
        ratings = inner_product(user_embs, item_embs)
        return ratings

    def _forward_gcn(self):
        norm_adj = self.norm_adj
        if self.node_dropout_flag is True:
            norm_adj = torch.dropout(norm_adj, self.node_dropout, self.training)

        ego_embeddings = torch.cat([self.user_embeddings.weight,
                                    self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            # transformed sum messages of neighbors.
            sum_embeddings = F.leaky_relu(side_embeddings.mm(self.weights['W_gc_%d' % k]) +
                                          self.weights['b_gc_%d' % k], negative_slope=0.2)

            # bi messages of neighbors.
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)

            # transformed bi messages of neighbors.
            bi_embeddings = F.leaky_relu(bi_embeddings.mm(self.weights['W_bi_%d' % k]) +
                                         self.weights['b_bi_%d' % k], negative_slope=0.2)
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = torch.dropout(ego_embeddings, 1-self.mess_dropout[k], self.training)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self.user_embeddings_final is None or self.item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self.user_embeddings_final)
        ratings = torch.matmul(user_embs, self.item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_NGCF, self).eval()
        self.user_embeddings_final, self.item_embeddings_final = self._forward_gcn()


class NGCF(AbstractRecommender):
    def __init__(self, config):
        super(NGCF, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.layer_size = config["layer_size"]

        self.n_layers = len(self.layer_size)
        self.adj_type = config["adj_type"]
        self.node_dropout_flag = config["node_dropout_flag"]
        self.node_dropout = config["node_dropout"]
        self.mess_dropout = config["mess_dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config['batch_size']
        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat(self.adj_type)
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.ngcf = _NGCF(self.num_users, self.num_items, self.emb_size,
                          adj_matrix, self.layer_size,  self.node_dropout_flag,
                          self.node_dropout, self.mess_dropout).to(self.device)
        self.ngcf.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.ngcf.parameters(), lr=self.lr)

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
            self.ngcf.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                _bat_users = torch.cat([bat_users, bat_users], dim=0)
                _bat_items = torch.cat([bat_pos_items, bat_neg_items], dim=0)

                hat_y = self.ngcf(_bat_users, _bat_items)
                yui, yuj = torch.split(hat_y, [len(bat_pos_items), len(bat_neg_items)], dim=0)

                loss = pairwise_loss(self.loss_func, yui-yuj, reduction=Reduction.MEAN)
                reg_loss = l2_loss(F.embedding(bat_users, self.ngcf.user_embeddings_final),
                                   F.embedding(bat_pos_items, self.ngcf.item_embeddings_final),
                                   F.embedding(bat_neg_items, self.ngcf.item_embeddings_final)
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
            self.ngcf.train()
            for bat_users, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.ngcf(bat_users, bat_items)
                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.MEAN)
                reg_loss = l2_loss(F.embedding(bat_users, self.ngcf.user_embeddings_final),
                                   F.embedding(bat_items, self.ngcf.item_embeddings_final)
                                   )
                loss += self.reg * reg_loss/self.emb_size
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    # @timer
    def evaluate_model(self):
        self.ngcf.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.ngcf.predict(users).cpu().detach().numpy()
