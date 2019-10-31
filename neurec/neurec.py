import logging
from neurec.data.Dataset import Dataset
from neurec.data.properties import types
from neurec.data.models import models
from neurec.util import tool
from neurec.util.properties import Properties
import numpy as np
import os
import random
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def setup(properties_path, properties_section="DEFAULT", numpy_seed=2018, random_seed=2018, tensorflow_seed=2018):
    """Sets up initial values for neurec and loads the dataset.

    properties_path -- path to properties file
    properties_section -- section inside the properties files to read (default "DEFAULT")
    numpy_seed -- seed value for numpy random (default 2018)
    random_seed -- seed value for random (default 2018)
    tensorflow_seed -- seed value for tensorflow random (default 2018)
    """
    np.random.seed(numpy_seed)
    random.seed(random_seed)
    tf.set_random_seed(tensorflow_seed)

    Properties().setSection(properties_section)
    Properties().setProperties(properties_path)

    properties = Properties().getProperties([
        "data.input.path",
        "data.input.dataset",
        "data.splitter",
        "data.convert.separator",
        "data.convert.binarize.threshold",
        "rec.evaluate.neg",
        "data.column.format",
        "data.splitterratio",
        "gpu_id"
    ])

    Dataset(
        properties["data.input.path"],
        properties["data.input.dataset"],
        properties["data.column.format"],
        properties["data.splitter"],
        properties["data.convert.separator"],
        properties["data.convert.binarize.threshold"],
        properties["rec.evaluate.neg"],
        properties["data.splitterratio"]
    )

    if tool.get_available_gpus(properties["gpu_id"]):
        os.environ["CUDA_VISIBLE_DEVICES"] = properties["gpu_id"]

def run():
    """Trains and evaluates a model."""
    logger = logging.getLogger(__name__)

    recommender = Properties().getProperty("recommender")    

    if not recommender in models:
        raise KeyError("Recommender " + str(recommender) + " not recognised. Add recommender to neurec.util.models")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    num_thread = Properties().getProperty("rec.number.thread")

    with tf.Session(config=config) as sess:
        model = models[recommender](sess=sess)
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()

def listModels():
    """Returns a list of available models."""
    return models

def listProperties(model):
    """Returns a list of properties used by the model.

    model -- name of a model
    """
    model_properties = models[model].properties
    list = []

    for property in model_properties:
        list.append({
            "name": property,
            "type": types[property].__name__
        })

    return list
