'''
Reference: Ruining He et al., "Fusing similarity models with Markov chains for sparse sequential recommendation." in ICDM 2016.
@author: wubin
'''

from neurec.model.AbstractRecommender import SeqAbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from neurec.util import learner, tool

from neurec.util.tool import csr_to_user_dict_bytime, randint_choice, timer
from neurec.util.tool import l2_loss


class Fossil(SeqAbstractRecommender):
    properties = [
        "verbose",
        "batch_size",
        "epochs",
        "embedding_size",
        "regs",
        "alpha",
        "num_neg",
        "learning_rate",
        "learner",
        "loss_function",
        "is_pairwise",
        "high_order",
        "num_neg",
        "init_method",
        "stddev"
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        
        self.verbose = self.conf["verbose"]
        self.batch_size = self.conf["batch_size"]
        self.num_epochs = self.conf["epochs"]
        self.embedding_size = self.conf["embedding_size"]
        regs = self.conf["regs"]
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.reg_eta = regs[2]
        self.alpha = self.conf["alpha"]
        self.num_negatives = self.conf["num_neg"]
        self.learning_rate = self.conf["learning_rate"]
        self.learner = self.conf["learner"]
        self.loss_function = self.conf["loss_function"]
        self.is_pairwise = self.conf["is_pairwise"]
        self.high_order = self.conf["high_order"]
        self.num_negatives = self.conf["num_neg"]
        self.init_method = self.conf["init_method"]
        self.stddev = self.conf["stddev"]
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        
        self.train_matrix = self.dataset.train_matrix
        self.train_dict = csr_to_user_dict_bytime(self.dataset.time_matrix,self.train_matrix)
        

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input_id = tf.placeholder(tf.int32, shape=[None,], name = "user_input_id")    #the index of users
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name = "user_input")    #the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None,],name = "num_idx")    #the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None,],name = "item_input_pos")      #the index of items
            self.item_input_recents = tf.placeholder(tf.int32, shape = [None,None], name = "item_input_recent")
            if self.is_pairwise == True:
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name = "user_input_neg")    
                self.item_input_neg = tf.placeholder(tf.int32, shape = [None,], name = "item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None,],name = "num_idx_neg")
            else :
                self.lables = tf.placeholder(tf.float32, shape=[None,],name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            initializer = tool.get_initializer(self.init_method, self.stddev)
            
            self.c1 = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                                 name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2' )
            self.embedding_P = tf.concat([self.c1,self.c2], 0, name='embedding_P')
            self.embedding_Q = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                                                name='embedding_Q', dtype=tf.float32)
            
            self.eta = tf.Variable(initializer([self.num_users, self.high_order]),name='eta')
            self.eta_bias = tf.Variable(initializer([1,self.high_order]),name='eta_bias')
            
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias')

    def _create_inference(self,user_input,item_input,num_idx):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, user_input), 1)
            embedding_eta_u = tf.nn.embedding_lookup(self.eta, self.user_input_id)
            batch_size = tf.shape(embedding_eta_u)[0]
            eta = tf.expand_dims(tf.tile(self.eta_bias, tf.stack([batch_size,1])) + embedding_eta_u,-1)
            embeddings_short = tf.nn.embedding_lookup(self.embedding_P, self.item_input_recents)
            embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(tf.multiply(embedding_p,embedding_q), 1)+ tf.reduce_sum(tf.multiply(tf.reduce_sum(eta*embeddings_short,1),embedding_q),1)+ bias_i
        return embedding_p, embedding_q,embedding_eta_u,embeddings_short,output
    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, eta_u,short, self.output = self._create_inference(self.user_input,self.item_input,self.num_idx)
            if self.is_pairwise == True:
                _, q2,_,_,output_neg = self._create_inference(self.user_input_neg,self.item_input_neg,self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function,self.result) + \
                            self.lambda_bilinear*l2_loss(p1) +\
                            self.gamma_bilinear*l2_loss(q2, q1, short) + \
                            self.reg_eta*l2_loss(eta_u, self.eta_bias)
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.lables,self.output)+ \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q1, short) + \
                            self.reg_eta * l2_loss(eta_u, self.eta_bias)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
              
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.evaluate()
        for epoch in range(1, self.num_epochs+1):
            if self.is_pairwise == True:
                user_input_id,user_input,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,item_input_recents = \
                self._get_pairwise_all_likefossil_data()
            else :
                user_input_id,user_input,num_idx,item_input,item_input_recents,lables = self._get_pointwise_all_likefossil_data()
           
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            for num_batch in np.arange(int(num_training_instances/self.batch_size)):
                if self.is_pairwise == True:
                    bat_user_input_id,bat_users_pos,bat_users_neg, bat_idx_pos, bat_idx_neg, \
                        bat_items_pos,bat_items_neg,bat_item_input_recents= \
                        self._get_pairwise_batch_likefossil_data(user_input_id,user_input,\
                        user_input_neg,self.dataset.num_items, num_idx_pos, num_idx_neg, item_input_pos,\
                        item_input_neg,item_input_recents, num_batch, self.batch_size) 
                    feed_dict = {self.user_input_id:bat_user_input_id,self.user_input:bat_users_pos,self.user_input_neg:bat_users_neg,\
                                self.num_idx:bat_idx_pos,self.num_idx_neg:bat_idx_neg,
                                self.item_input:bat_items_pos,self.item_input_neg:bat_items_neg,self.item_input_recents:bat_item_input_recents}
                else :
                    bat_user_input_id,bat_users,bat_idx,bat_items,bat_item_input_recents,bat_lables =\
                        self._get_pointwise_batch_likefossil_data(user_input_id,user_input,self.dataset.num_items,
                        num_idx,item_input,item_input_recents,lables, num_batch, self.batch_size)
                    feed_dict = {self.user_input_id:bat_user_input_id,self.user_input:bat_users,self.num_idx:bat_idx, self.item_input:bat_items,
                                self.item_input_recents:bat_item_input_recents,self.lables:bat_lables}
    
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
                
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)                

    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is None:
            allitems = np.arange(self.num_items)
            for userid in user_ids:
                items_by_userid = self.train_dict[userid]
                num_idx = len(items_by_userid)
                # Get prediction scores
                item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                item_recents = []
                user_input = []
                user_input.extend([items_by_userid]*self.num_items)
                for _ in range(self.num_items):
                    item_recents.append(items_by_userid[len(items_by_userid)-self.high_order:])
                users = np.full(self.num_items, userid, dtype=np.int32)
                feed_dict = {self.user_input_id: users,
                             self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: allitems,
                             self.item_input_recents: item_recents}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
                
        else:
            for userid, eval_items_by_userid in zip(user_ids, candidate_items_userids):    
                items_by_userid = self.train_dict[userid]
                num_idx = len(items_by_userid)
                # Get prediction scores
                item_idx = np.full(len(eval_items_by_userid), num_idx, dtype=np.int32)
                user_input = []
                user_input.extend([items_by_userid]*len(eval_items_by_userid))
                item_recents = []
                for _ in range(len(eval_items_by_userid)):
                    item_recents.append(items_by_userid[len(items_by_userid)-self.high_order:])
                users = np.full(len(eval_items_by_userid), userid, dtype=np.int32)
                feed_dict = {self.user_input_id: users,
                             self.user_input: user_input,
                             self.num_idx: item_idx, 
                             self.item_input: eval_items_by_userid,
                             self.item_input_recents: item_recents}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings

    def _get_pairwise_all_likefossil_data(self):
        user_input_id,user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,item_input_recents = [],[], [], [],[],[],[],[]
        for u in range(self.num_users):
            items_by_user = self.train_dict[u].copy()
            num_items_by_u = len(items_by_user)
            if  num_items_by_u > self.high_order: 
                negative_items = randint_choice(self.num_items, num_items_by_u, replace=True, exclusion = items_by_user)
                for idx in range(self.high_order,len(self.train_dict[u])):
                    i = self.train_dict[u][idx] # item id 
                    item_input_recent = []
                    for t in range(1,self.high_order+1):
                        item_input_recent.append(self.train_dict[u][idx-t])
                    item_input_recents.append(item_input_recent)
                    j = negative_items[idx]
                    user_input_neg.append(items_by_user)
                    num_idx_neg.append(num_items_by_u)
                    item_input_neg.append(j)
                    
                    items_by_user.remove(i)
                    user_input_id.append(u)
                    user_input_pos.append(items_by_user)
                    num_idx_pos.append(num_items_by_u-1)
                    item_input_pos.append(i)
        user_input_id =  np.array(user_input_id)
        user_input_pos = np.array(user_input_pos)
        user_input_neg = np.array(user_input_neg)
        num_idx_pos = np.array(num_idx_pos,dtype=np.int32)
        num_idx_neg = np.array(num_idx_neg,dtype=np.int32)
        item_input_pos = np.array(item_input_pos, dtype=np.int32)
        item_input_neg = np.array(item_input_neg, dtype=np.int32)
        item_input_recents = np.array(item_input_recents)
        num_training_instances = len(user_input_pos)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_input_id = user_input_id[shuffle_index]
        user_input_pos = user_input_pos[shuffle_index]
        item_input_recents = item_input_recents[shuffle_index]
        user_input_neg = user_input_neg[shuffle_index]
        num_idx_pos = num_idx_pos[shuffle_index]
        num_idx_neg = num_idx_neg[shuffle_index]
        item_input_pos = item_input_pos[shuffle_index]
        item_input_neg = item_input_neg[shuffle_index]    
        return user_input_id,user_input_pos,user_input_neg, num_idx_pos, num_idx_neg, item_input_pos,item_input_neg,item_input_recents

    def _get_pointwise_all_likefossil_data(self):
        user_input_id,user_input,num_idx,item_input,item_input_recents,lables = [],[],[],[],[],[]
        for u in range(self.num_users):
            items_by_user = self.train_dict[u].copy()
            size = len(items_by_user)   
            for idx in range(self.high_order,len(self.train_dict[u])):
                i = self.train_dict[u][idx] # item id 
                item_input_recent = []
                for t in range(1,self.high_order+1):
                    item_input_recent.append(self.train_dict[u][idx-t])
                # negative instances
                for _ in range(self.num_negatives):
                    j = np.random.randint(self.num_items)
                    while (u,j) in self.trainMatrix.keys():
                        j = np.random.randint(self.num_items)
                    user_input_id.append(u)
                    user_input.append(items_by_user)
                    item_input_recents.append(item_input_recent)
                    item_input.append(j)
                    num_idx.append(size)
                    lables.append(0)
                items_by_user.remove(i)
                user_input.append(items_by_user)
                user_input_id.append(u)
                item_input_recents.append(item_input_recent)
                item_input.append(i)
                num_idx.append(size-1)
                lables.append(1)
        user_input_id = np.array(user_input_id)
        user_input = np.array(user_input)
        item_input_recents = np.array(item_input_recents)
        num_idx = np.array(num_idx, dtype=np.int32)
        item_input = np.array(item_input, dtype=np.int32)
        lables = np.array(lables, dtype=np.float32)
        num_training_instances = len(user_input)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_input_id = user_input_id[shuffle_index]
        user_input = user_input[shuffle_index]
        item_input_recents = item_input_recents[shuffle_index]
        num_idx = num_idx[shuffle_index]
        item_input = item_input[shuffle_index]
        lables = lables[shuffle_index]
        return user_input_id,user_input,num_idx,item_input,item_input_recents,lables
    
    def _get_pairwise_batch_likefossil_data(self,user_input_id,user_input_pos,user_input_neg,num_items, num_idx_pos,\
         num_idx_neg, item_input_pos,item_input_neg,item_input_recents,num_batch,batch_size):
        num_training_instances = len(user_input_pos)
        id_start = num_batch * batch_size
        id_end = (num_batch + 1) * batch_size
        if id_end>num_training_instances:
            id_end=num_training_instances
        bat_idx_pos = num_idx_pos[id_start:id_end]
        bat_idx_neg = num_idx_neg[id_start:id_end]
        max_pos = max(bat_idx_pos)  
        max_neg = max(bat_idx_neg) 
        bat_users_pos = user_input_pos[id_start:id_end].tolist()
        bat_users_neg = user_input_neg[id_start:id_end].tolist()
        for i in range(len(bat_users_pos)):
            bat_users_pos[i] = bat_users_pos[i] + \
            [num_items] * (max_pos - len(bat_users_pos[i]))
            
        for i in range(len(bat_users_neg)):
            bat_users_neg[i] = bat_users_neg[i] + \
            [num_items] * (max_neg - len(bat_users_neg[i]))  
        bat_user_input_id = user_input_id[id_start:id_end]
        bat_items_pos = item_input_pos[id_start:id_end]
        bat_items_neg = item_input_neg[id_start:id_end]
        bat_item_input_recents = item_input_recents[id_start:id_end]
        return bat_user_input_id,bat_users_pos,bat_users_neg,bat_idx_pos,bat_idx_neg,bat_items_pos,bat_items_neg,bat_item_input_recents
    
    def _get_pointwise_batch_likefossil_data(self,user_input_id,user_input,num_items,num_idx,item_input,item_input_recents,lables,num_batch,batch_size):
        num_training_instances =len(user_input) 
        id_start = num_batch * batch_size
        id_end = (num_batch + 1) * batch_size
        if id_end>num_training_instances:
            id_end=num_training_instances
        bat_users = user_input[id_start:id_end].tolist()
        bat_idx = num_idx[id_start:id_end]
        max_by_user = max(bat_idx)
        for i in range(len(bat_users)):
            bat_users[i] = bat_users[i] + \
            [num_items] * (max_by_user - len(bat_users[i]))
        bat_user_input_id = user_input_id[id_start:id_end]
        bat_items = item_input[id_start:id_end]
        bat_item_input_recents = item_input_recents[id_start:id_end]
        bat_lables = lables[id_start:id_end]
        return bat_user_input_id,bat_users,bat_idx,bat_items,bat_item_input_recents,bat_lables
