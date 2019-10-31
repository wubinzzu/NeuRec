"""
@author: Zhongchuan Sun
"""

import os
import sys
from configparser import ConfigParser
from collections import OrderedDict

class Configurer(object):
    def __init__(self):
        self.lib_arg = None
        self.alg_arg = None

        # get argument
        self._load_lib_arg()
        self._load_alg_arg()
        self._load_cmd_arg()

    def _load_lib_arg(self):
        lib_file = "NeuRec.properties"
        config = ConfigParser()
        config.optionxform = str
        config.read(lib_file)
        self.lib_arg = OrderedDict(config["default"].items())
        self.lib_arg["data_name"] = self.lib_arg["data.input.dataset"]

    def _load_alg_arg(self):
        alg_file = os.path.join("./conf", self.lib_arg["recommender"] + ".properties")
        config = ConfigParser()
        config.optionxform = str
        config.read(alg_file)
        self.alg_arg = OrderedDict(config["hyperparameters"].items())

    def _load_cmd_arg(self):
        for arg in sys.argv[1:]:
            arg_name, arg_value = arg.split("=")
            arg_name = arg_name[2:]
            if arg_name in self.lib_arg:
                self.lib_arg[arg_name] = arg_value
            if arg_name in self.alg_arg:
                self.alg_arg[arg_name] = arg_value

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.lib_arg:
            param = self.lib_arg[item]
        elif item in self.alg_arg:
            param = self.alg_arg[item]
        else:
            raise NameError("There are not the parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple)):
                value = param
        except:
            value = param

        return value

    def __str__(self):
        lib_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.lib_arg.items()])
        alg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items()])
        info = "\nNeuRec hyperparameters:\n%s\n\nRecommender's hyperparameters:\n%s" % (lib_info, alg_info)
        return info

    def __repr__(self):
        return self.__str__()
