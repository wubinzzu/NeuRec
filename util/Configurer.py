"""
@author: Zhongchuan Sun
"""

import os
import sys
from configparser import ConfigParser
from collections import OrderedDict


class Configurer(object):
    def __init__(self):
        # get argument
        self.cmd_arg = self._load_cmd_arg()
        self.lib_arg = self._load_lib_arg()
        self.alg_arg = self._load_alg_arg()

        self._set_info()

    def _set_info(self):
        # parse the train file name to get dataset name
        self.lib_arg["data_name"] = self.lib_arg["data.input.dataset"]

    def _load_lib_arg(self):
        lib_file = "NeuRec.properties"
        config = ConfigParser()
        config.optionxform = str
        config.read(lib_file, encoding="utf-8")
        lib_arg = OrderedDict(config["default"].items())
        for arg in self.cmd_arg:
            if arg in lib_arg:
                lib_arg[arg] = self.cmd_arg[arg]

        return lib_arg

    def _load_alg_arg(self):
        alg_file = os.path.join("./conf", self.lib_arg["recommender"] + ".properties")
        config = ConfigParser()
        config.optionxform = str
        config.read(alg_file, encoding="utf-8")
        alg_arg = OrderedDict(config["hyperparameters"].items())
        for arg in self.cmd_arg:
            if arg in alg_arg:
                alg_arg[arg] = self.cmd_arg[arg]

        return alg_arg

    def _load_cmd_arg(self):
        cmd_arg = OrderedDict()
        for arg in sys.argv[1:]:
            arg_name, arg_value = arg.split("=")
            cmd_arg[arg_name[2:]] = arg_value

        return cmd_arg

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
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except:
            value = param

        return value

    def __str__(self):
        lib_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.lib_arg.items()])
        alg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items()])
        info = "\n\nNeuRec hyperparameters:\n%s\n\n%s's hyperparameters:\n%s" % (lib_info, self["recommender"], alg_info)
        return info

    def __repr__(self):
        return self.__str__()
