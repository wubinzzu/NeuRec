"""
Paper: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
Author: Jiaxi Tang, and Ke Wang
Reference: https://github.com/graytowne/caser_pytorch
"""

__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Caser"]

from model.base import AbstractRecommender
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from util.common import Reduction
from util.pytorch import sigmoid_cross_entropy
from util.pytorch import get_initializer
from data import TimeOrderPairwiseSampler


class _Caser(nn.Module):
    def __init__(self, num_users, num_items, dims, config, item_pad_idx=None):
        super(_Caser, self).__init__()
        self.args = config
        self._pad_idx = item_pad_idx

        # init args
        L = config["seq_L"]
        self.n_h = config["nh"]
        self.n_v = config["nv"]
        self.ac_conv = F.relu
        self.ac_fc = F.relu

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims, padding_idx=item_pad_idx)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims, padding_idx=item_pad_idx)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=item_pad_idx)
        # dropout
        self.dropout = nn.Dropout(config["dropout"])

        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        # weight initialization
        init = get_initializer(init_method)
        zero_init = get_initializer("zeros")

        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)
        init(self.W2.weight)
        zero_init(self.b2.weight)
        if self._pad_idx is not None:
            zero_init(self.item_embeddings.weight[self._pad_idx])
            zero_init(self.W2.weight[self._pad_idx])

    def _forward_user(self, user_var, seq_var):
        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        return x

    def forward(self, user_var, seq_var, item_var):
        x = self._forward_user(user_var, seq_var)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res

    def predict(self, user_var, seq_var):
        x = self._forward_user(user_var, seq_var)
        res = torch.matmul(x, self.W2.weight.T) + self.b2.weight.squeeze()
        return res


class Caser(AbstractRecommender):
    def __init__(self, config):
        super(Caser, self).__init__(config)

        self.lr = config["lr"]
        self.l2_reg = config["l2_reg"]
        self.emb_size = config["emb_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.seq_L = config["seq_L"]
        self.seq_T = config["seq_T"]
        self.neg_samples = config["neg_samples"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.pad_idx = self.num_items
        self.num_items += 1

        self.user_truncated_seq = self.dataset.train_data.to_truncated_seq_dict(self.seq_L,
                                                                                pad_value=self.pad_idx,
                                                                                padding='pre', truncating='pre')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.caser = _Caser(self.num_users, self.num_items, self.emb_size, config).to(self.device)
        self.caser.reset_parameters(config["param_init"])
        self.optimizer = torch.optim.Adam(self.caser.parameters(), weight_decay=self.l2_reg, lr=self.lr)

    def train_model(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.seq_L,
                                             len_next=self.seq_T, pad=self.pad_idx,
                                             num_neg=self.neg_samples,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            self.caser.train()
            for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_item_seqs = torch.from_numpy(bat_item_seqs).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                bat_items = torch.cat([bat_pos_items, bat_neg_items], dim=1)
                bat_ratings = self.caser(bat_users.unsqueeze(dim=1), bat_item_seqs, bat_items)

                yui, yuj = torch.split(bat_ratings, [self.seq_T, self.neg_samples], dim=1)
                ones = yui.new_ones(yui.size())
                zeros = yuj.new_zeros(yuj.size())
                loss = sigmoid_cross_entropy(yui, ones, reduction=Reduction.MEAN) + \
                       sigmoid_cross_entropy(yuj, zeros, reduction=Reduction.MEAN)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.caser.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        bat_seq = [self.user_truncated_seq[u] for u in users]
        bat_seq = torch.from_numpy(np.asarray(bat_seq)).long().to(self.device)
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        all_ratings = self.caser.predict(users, bat_seq)
        return all_ratings.cpu().detach().numpy()
