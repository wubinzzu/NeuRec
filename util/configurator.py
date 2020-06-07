"""
@author: Zhongchuan Sun
"""

import os
import sys
from configparser import ConfigParser
from collections import OrderedDict


class Configurator(object):
    """A configurator class.

    This class can read arguments from ini-style configuration file and parse
    arguments from command line simultaneously. This class can also convert
    the argument value from `str` to `int`, `float`, `bool`, `list` and `None`
    automatically. The priority of arguments from command line is higher than
    that from configuration file. That is, if there are same argument name in
    configuration file and command line, the value in the former will be
    overwritten by that in the latter. Moreover:

    * Command line: The format of arguments is ``--arg_name=arg_value``,
      there cannot be any space in the inner of an argument string.
      For example::

        python main.py --model=Pop --num_thread=128 --group_view=[10,30,50,100]

    * Configuration file: This file must be ini-style. If there is only one
      section and whatever the name is, this class will read arguments from
      that section. If there are more than one sections, this class will read
      arguments from the section named `default_section`.

    After initialization successful, the objective of this class can be used as
    a dictionary::

        config = Configurator("./NeuRec.properties")
        num_thread = config["num_thread"]
        group_view = config["group_view"]

    Here, the types of `num_thread` and `group_view` are `int` and `list`,
    respectively.
    """

    def __init__(self, config_file, default_section="default"):
        """Initializes a new `Configurator` instance.

        Args:
            config_file (str): The path of ini-style configuration file.
            default_section (str): The default section if there are more than
                one sections in configuration file.

        Raises:
             FileNotFoundError: If `config_file` is not existing.
             SyntaxError: If the format of arguments in commend line is invalid.
             ValueError: If there is more than one section but no one section
                named `default_section` in ini-style file.
        """
        if not os.path.isfile(config_file):
            raise FileNotFoundError("There is not config file named '%s'!" % config_file)

        self._default_section = default_section
        self.cmd_arg = self._read_cmd_arg()
        self.lib_arg = self._read_config_file(config_file)
        config_dir = self.lib_arg["config_dir"]
        model_name = self.lib_arg["recommender"]
        arg_file = os.path.join(config_dir, model_name+'.properties')
        self.alg_arg = self._read_config_file(arg_file)

    def _read_cmd_arg(self):
        cmd_arg = OrderedDict()
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--"):
                    raise SyntaxError("Commend arg must start with '--', but '%s' is not!" % arg)
                arg_name, arg_value = arg[2:].split("=")
                cmd_arg[arg_name] = arg_value

        return cmd_arg

    def _read_config_file(self, filename):
        config = ConfigParser()
        config.optionxform = str
        config.read(filename, encoding="utf-8")
        sections = config.sections()

        if len(sections) == 0:
            raise ValueError("'%s' is empty!" % filename)
        elif len(sections) == 1:
            config_sec = sections[0]
        elif self._default_section in sections:
            config_sec = self._default_section
        else:
            raise ValueError("'%s' has more than one sections but there is no "
                             "section named '%s'" % filename, self._default_section)

        config_arg = OrderedDict(config[config_sec].items())
        for arg in self.cmd_arg:
            if arg in config_arg:
                config_arg[arg] = self.cmd_arg[arg]

        return config_arg

    def params_str(self):
        """Get a summary of parameters.

        Returns:
            str: A string summary of parameters.
        """
        params_id = '_'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items() if len(value) < 20])
        special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t'}
        params_id = [c if c not in special_char else '_' for c in params_id]
        params_id = ''.join(params_id)
        params_id = "%s_%s" % (self["recommender"], params_id)
        return params_id

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.lib_arg:
            param = self.lib_arg[item]
        elif item in self.alg_arg:
            param = self.alg_arg[item]
        elif item in self.cmd_arg:
            param = self.cmd_arg[item]
        else:
            raise KeyError("There are not the parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except:
            if param.lower() == "true":
                value = True
            elif param.lower() == "false":
                value = False
            else:
                value = param

        return value

    def __getattr__(self, item):
        return self[item]

    def __contains__(self, o):
        return o in self.lib_arg or o in self.alg_arg or o in self.cmd_arg

    def __str__(self):
        lib_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.lib_arg.items()])
        alg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.alg_arg.items()])
        info = "\n\nNeuRec hyperparameters:\n%s\n\n%s's hyperparameters:\n%s\n" % (lib_info, self["recommender"], alg_info)
        return info

    def __repr__(self):
        return self.__str__()
