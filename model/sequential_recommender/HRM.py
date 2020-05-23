"""
Reference: Pengfei Wang et al., "Learning Hierarchical Representation Model for NextBasket Recommendation." in SIGIR 2015.
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import timer
from util import learner, tool
from util.tool import csr_to_user_dict_bytime
from model.AbstractRecommender import SeqAbstractRecommender
from util import l2_loss
from data import TimeOrderPointwiseSampler


class HRM(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(HRM, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.pre_agg = conf["pre_agg"]
        self.loss_function = conf["loss_function"]
        self.session_agg = conf["session_agg"]
        self.batch_size = conf["batch_size"]
        self.high_order = conf["high_order"]
        self.verbose = conf["verbose"]
        self.num_negatives = conf["num_neg"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.train_dict = csr_to_user_dict_bytime(dataset.time_matrix, dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, ], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, ], name="item_input_pos")
            self.item_input_recent = tf.placeholder(tf.int32, shape=[None, None], name = "item_input_recent")
            self.labels = tf.placeholder(tf.float32, shape=[None, ], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)
            self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                               name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)
    
    def avg_pooling(self, input_embedding):
        output = tf.reduce_mean(input_embedding, axis=1)
        return output

    def max_pooling(self, input_embedding):
        output = tf.reduce_max(input_embedding, axis=1)
        return output
    
    def _create_inference(self):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            item_embedding_recent = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_recent)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
            if self.high_order > 1:
                if self.session_agg == "max":
                    item_embedding_short = self.max_pooling(item_embedding_recent)  
                else :
                    item_embedding_short = self.avg_pooling(item_embedding_recent) 
                concat_user_item = tf.concat([tf.expand_dims(user_embedding, 1),
                                              tf.expand_dims(item_embedding_short, 1)], axis=1)
            else:
                concat_user_item = tf.concat([tf.expand_dims(user_embedding, 1),
                                              tf.expand_dims(item_embedding_recent, 1)], axis=1)
            if self.pre_agg == "max":
                hybrid_user_embedding = self.max_pooling(concat_user_item)
            else:
                hybrid_user_embedding = self.avg_pooling(concat_user_item)
            predict_vector = tf.multiply(hybrid_user_embedding, item_embedding)
            predict = tf.reduce_sum(predict_vector, 1)
            return user_embedding, item_embedding, item_embedding_recent, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            p1, q1, r1, self.output = self._create_inference()
            self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                        self.reg_mf * l2_loss(p1, r1, q1)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
                
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        data_iter = TimeOrderPointwiseSampler(self.dataset, high_order=self.high_order,
                                              neg_num=self.num_negatives,
                                              batch_size=self.batch_size, shuffle=True)

        for epoch in  range(1, self.num_epochs+1):
            num_training_instances = len(data_iter)
            total_loss = 0.0
            training_start_time = time()
      
            for bat_users, bat_items_recent, bat_items, bat_labels in data_iter:
                    
                    feed_dict = {self.user_input: bat_users,
                                 self.item_input: bat_items,
                                 self.item_input_recent: bat_items_recent,
                                 self.labels: bat_labels}
    
                    loss, _ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                    total_loss += loss
                    
            self.logger.info("[iter %d : loss : %f, time: %f]" %
                             (epoch, total_loss/num_training_instances, time()-training_start_time))
            
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)
    
    def predict(self, user_ids, candidate_items_user_ids):
        ratings = []
        if candidate_items_user_ids is None:
            all_items = np.arange(self.num_items)
            for user_id in user_ids:
                cand_items = self.train_dict[user_id]
                item_recent = []
    
                for _ in range(self.num_items):
                    item_recent.append(cand_items[len(cand_items)-self.high_order:])
                users = np.full(self.num_items, user_id, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: all_items}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))  
        
        else:
            for user_id, items_by_user_id in zip(user_ids, candidate_items_user_ids):
                cand_items = self.train_dict[user_id]
                item_recent = []
    
                for _ in range(len(items_by_user_id)):
                    item_recent.append(cand_items[len(cand_items)-self.high_order:])
                users = np.full(len(items_by_user_id), user_id, dtype=np.int32)
                feed_dict = {self.user_input: users,
                             self.item_input_recent: item_recent,
                             self.item_input: items_by_user_id}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings
