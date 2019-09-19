import numpy as np
import tensorflow as tf
from neurec.data.Dataset import Dataset
from neurec.model.item_ranking.MF import MF
from neurec.model.seq_ranking.FPMC import FPMC
from neurec.model.seq_ranking.FPMCplus import FPMCplus
from neurec.model.seq_ranking.TransRec import TransRec
import configparser
from neurec.evaluation import Evaluate
from neurec.model.item_ranking.FISM import FISM
from neurec.model.item_ranking.NAIS import NAIS
from neurec.model.item_ranking.MLP import MLP
from neurec.model.item_ranking.NeuMF import NeuMF
from neurec.model.item_ranking.APR import APR
from neurec.model.seq_ranking.HRM import HRM
from neurec.model.item_ranking.DMF import DMF
from neurec.model.item_ranking.ConvNCF import ConvNCF
from neurec.model.item_ranking.CDAE import CDAE
from neurec.model.item_ranking.DAE import DAE
from neurec.model.seq_ranking.NPE import NPE
from neurec.model.item_ranking.IRGAN import IRGAN
from neurec.model.item_ranking.MultiDAE import MultiDAE
from neurec.model.item_ranking.MultiVAE import MultiVAE
from neurec.model.item_ranking.JCA import JCA
from neurec.model.item_ranking.CFGAN import CFGAN
from neurec.model.item_ranking.SBPR import SBPR
from neurec.model.item_ranking.WRMF import WRMF
from neurec.model.item_ranking.SpectralCF import SpectralCF
from neurec.util import reader

conf = reader.config("NeuRec.properties", "default")
data_input_path = conf["data.input.path"]
dataset_name = conf["data.input.dataset"]
splitter = conf["data.splitter"]
dataset_format = conf["data.column.format"]
separator = eval(conf["data.convert.separator"])
threshold = float(conf["data.convert.binarize.threshold"])
recommender = str(conf["recommender"])
evaluate_neg = int(conf["rec.evaluate.neg"])
splitterRatio=list(eval(conf["data.splitterratio"]))

dataset = Dataset(data_input_path,dataset_name,splitter,separator,threshold,evaluate_neg,splitterRatio)
num_thread = int(conf["rec.number.thread"])
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def setup(numpy_seed=2018, tensorflow_seed=2017):
    """Setups initial values for neurec.

    numpy_seed -- seed value for numpy random (default 2018)
    tensorflow_seed -- seed value for tensorflow random (default 2017)
    """
    np.random.seed(numpy_seed)
    tf.compat.v1.set_random_seed(tensorflow_seed)

def run():
    """Trains and evaluates a model."""
    with tf.compat.v1.Session(config=config) as sess:
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

        elif recommender.lower() == "cdae":
            model = CDAE(sess,dataset)

        elif recommender.lower() == "dae":
            model = DAE(sess,dataset)

        elif recommender.lower() == "npe":
            model = NPE(sess,dataset)

        elif recommender.lower() == "multidae":
            model = MultiDAE(sess,dataset)

        elif recommender.lower() == "multivae":
            model = MultiVAE(sess,dataset)

        elif recommender.lower() == "irgan":
            model = IRGAN(sess,dataset)

        elif recommender.lower() == "cfgan":
            model = CFGAN(sess,dataset)

        elif recommender.lower() == "jca":
            model = JCA(sess,dataset)

        elif recommender.lower() == "sbpr":
            model = SBPR(sess,dataset)

        elif recommender.lower() == "spectralcf":
            model = SpectralCF(sess,dataset)

        elif recommender.lower() == "wrmf":
            model = WRMF(dataset)

        model.build_graph()
        sess.run(tf.compat.v1.global_variables_initializer())
        model.train_model()
        Evaluate.test_model(model,dataset,num_thread)
