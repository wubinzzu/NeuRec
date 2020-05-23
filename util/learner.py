import tensorflow as tf
def optimizer(learner,loss,learning_rate,momentum=0.9):
    optimizer=None
    if learner.lower() == "adagrad": 
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,\
                     initial_accumulator_value=1e-8).minimize(loss)
    elif learner.lower() == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "gd" :
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  
    elif learner.lower() == "momentum" :
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)  
    else :
        raise ValueError("please select a suitable optimizer")  
    return optimizer

def pairwise_loss(loss_function,y,margin=1):
    loss=None
    if loss_function.lower() == "bpr":
        loss = -tf.reduce_sum(tf.log_sigmoid(y))
    elif loss_function.lower() == "hinge":
        loss = tf.reduce_sum(tf.maximum(y+margin, 0))
    elif loss_function.lower() == "square":  
        loss = tf.reduce_sum(tf.square(1-y))   
    else:
        raise Exception("please choose a suitable loss function")
    return loss

def pointwise_loss(loss_function,y_rea,y_pre):
    loss=None
    if loss_function.lower() == "cross_entropy":
        loss = tf.losses.sigmoid_cross_entropy(y_rea,y_pre)
#         loss = - tf.reduce_sum(
#             y_rea * tf.log(y_pre) + (1 - y_rea) * tf.log(1 - y_pre)) 
    elif loss_function.lower() == "square":  
        loss = tf.reduce_sum(tf.square(y_rea-y_pre))  
    else:
        raise Exception("please choose a suitable loss function") 
    return loss
