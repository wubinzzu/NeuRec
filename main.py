import numpy as np
import tensorflow as tf
from data.Dataset import Dataset
from model.item_ranking.MF import MF
from model.seq_ranking.FPMC import FPMC
from model.seq_ranking.FPMCplus import FPMCplus
from model.seq_ranking.TransRec import TransRec
import configparser
from evaluation import Evaluate
from model.item_ranking.FISM import FISM
from model.item_ranking.NAIS import NAIS
from model.item_ranking.MLP import MLP
from model.item_ranking.NeuMF import NeuMF
from model.item_ranking.APR import APR
from model.seq_ranking.HRM import HRM

from model.item_ranking.DMF import DMF 
from model.item_ranking.ConvNCF import ConvNCF
np.random.seed(2018)
tf.random.set_random_seed(2017)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("tensorrec.properties")
    conf=dict(config.items("default")) 
    data_input_path = conf["data.input.path"]
    dataset_name = conf["data.input.dataset"]
    splitter = conf["data.splitter"]
    separator = eval(conf["data.convert.separator"])
    recommender = str(conf["recommender"])
    evaluate_neg = int(conf["rec.evaluate.neg"])
    isgivenTest = str(conf["isgiventest"])
    #splitterRatio=list(eval(conf["data.splitterratio"]))
    splitterRatio= [0.7,0.1,0.2]
    dataset = Dataset(data_input_path+dataset_name,splitter,separator,evaluate_neg,dataset_name,isgivenTest,splitterRatio) 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if recommender.lower() == "mf" : 
            model = MF(sess,dataset)    
                       
        elif recommender.lower() == "fpmc":
            model = FPMC(sess,dataset) 
            
        elif recommender.lower() == "fpmcplus":
            model = FPMCplus(sess,dataset) 
            
        elif recommender.lower() == "fism":
            model = FISM(sess,dataset) 
            
        elif recommender.lower() == "apr":
            model = APR(sess,dataset) 
            
        elif recommender.lower() == "nais":
            model = NAIS(sess,dataset) 
            
        elif recommender.lower() == "mlp":
            model = MLP(sess,dataset) 
        
        elif recommender.lower() == "hrm":
            model = HRM(sess,dataset) 
        
        elif recommender.lower() == "dmf":
            model = DMF(sess,dataset) 
            
        elif recommender.lower() == "neumf":
            model = NeuMF(sess,dataset) 
        
               
        elif recommender.lower() == "convncf":
            model = ConvNCF(sess,dataset) 
            
        elif recommender.lower() == "transrec":
            model = TransRec(sess,dataset)  
            
            
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        Evaluate.test_model(model,dataset)