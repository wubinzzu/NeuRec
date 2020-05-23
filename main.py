import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool


np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    conf = Configurator("NeuRec.properties", default_section="hyperparameters")
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    # num_thread = int(conf["rec.number.thread"])

    # if Tool.get_available_gpus(gpu_id):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
