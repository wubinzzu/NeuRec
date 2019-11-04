from datetime import datetime
import logging
from neurec.data.properties import types
from neurec.data.models import models
from neurec.data.Dataset import Dataset
from neurec.evaluation import Evaluate
from neurec.util.properties import Properties
from neurec.util import reader
from neurec.data.models import models
import numpy as np
import tensorflow as tf

properties = Properties()
dataset = {}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

ch = logging.FileHandler('neurec.log', mode='a')
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def setup(properties_path, properties_section="DEFAULT", numpy_seed=2018, tensorflow_seed=2017):
    """Setups initial values for neurec.

    properties_path -- path to properties file
    properties_section -- section inside the properties files to read (default "DEFAULT")
    numpy_seed -- seed value for numpy random (default 2018)
    tensorflow_seed -- seed value for tensorflow random (default 2017)
    """
    np.random.seed(numpy_seed)
    tf.set_random_seed(tensorflow_seed)

    properties.setSection(properties_section)
    properties.setProperties(properties_path)

    data_input_path = properties.getProperty("data.input.path")
    dataset_name = properties.getProperty("data.input.dataset")
    splitter = properties.getProperty("data.splitter")
    separator = properties.getProperty("data.convert.separator")
    threshold = properties.getProperty("data.convert.binarize.threshold")
    evaluate_neg = properties.getProperty("rec.evaluate.neg")
    dataset_format = properties.getProperty("data.column.format")
    splitter_ratio = properties.getProperty("data.splitterratio")

    global dataset
    dataset = Dataset(data_input_path, dataset_name, dataset_format, splitter, separator, threshold, evaluate_neg, splitter_ratio)

def run():
    """Trains and evaluates a model."""
    logger = logging.getLogger(__name__)

    if not isinstance(dataset, Dataset):
       raise RuntimeError("Dataset not set. Call setup() function and pass a properties file to set the dataset")

    recommender = properties.getProperty("recommender")

    if not recommender in models:
        raise KeyError("Recommender " + str(recommender) + " not recognised. Add recommender to neurec.util.models")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    num_thread = properties.getProperty("rec.number.thread")

    with tf.Session(config=config) as sess:
        model = models[recommender](sess=sess)
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        Evaluate.test_model(model, dataset, num_thread)

def listModels():
    """Returns a list of available models."""
    return models

def listProperties(model):
    """Returns a list of properties used by the model.

    name -- name of a model
    """
    model_properties = models[model].properties
    list = []

    for property in model_properties:
        list.append({
            "name": property,
            "type": types[property].__name__
        })

    return list
