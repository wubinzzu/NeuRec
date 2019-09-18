from configparser import ConfigParser
from collections import OrderedDict
import logging
from neurec.util import reader
import time
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.logger = logging.getLogger("NeuRec")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on screen
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)

        # add two Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def _create_logger():
    lib_config = reader.config("NeuRec.properties", "default")

    model_name = lib_config["recommender"].upper() + ".properties"
    model_config = reader.config(model_name, "hyperparameters")

    data_name = lib_config["data.input.dataset"]

    log_dir = os.path.join("log", data_name, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger_name = '_'.join(["{}={}".format(arg, value) for arg, value in model_config.items()
                            if len(value) < 20])
    special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t'}
    logger_name = [c if c not in special_char else '_' for c in logger_name]
    logger_name = ''.join(logger_name)
    timestamp = time.time()

    logger_name = logger_name[:200]
    # data name, model name, param, timestamp
    logger_name = "%s_%s_%s_%d.log" % (data_name, model_name, logger_name, timestamp)
    logger_name = os.path.join(log_dir, logger_name)
    logger = Logger(logger_name)
    logger.info("Recommender:%s" % model_name)
    logger.info("Dataset name:%s" % data_name)
    argument = '\n'.join(["{}={}".format(arg, value) for arg, value in model_config.items()])
    logger.info("\nHyperparameters:\n%s " % argument)

    return logger


logger = _create_logger()


if __name__ == '__main__':
    log = Logger('NeuRec_test.log')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.error('error')
    log.critical('critical')
