"""Functions to handle reading files in the package."""
from configparser import ConfigParser, MissingSectionHeaderError
import pickle
from importlib import util
import os
import pandas

def file(path):
    """Returns the configuration settings for a file.

    path -- path to the file
    """
    parser = ConfigParser()

    try:
        with open(path) as file:
            parser.read_file(file)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find file " + str(path) + ". Make sure the file path is correct.")
    except MissingSectionHeaderError:
        raise RuntimeError("Could not find a section header in " + str(path) + '. Added [DEFAULT] to line 1 of this file')

    return parser

def lines(file_path):
    """Returns all lines from a file.

    file_path -- path of the file to read
    """
    try:
        with open(file_path) as file:
            return file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(str(file_path) + " could not be found. Check the file path is correct.")

def load_pickle(file_path):
    """Returns a pickle file.

    file_path -- path to pickle file
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file, encoding="latin")
    except FileNotFoundError:
        raise FileNotFoundError(str(file_path) + " could not be found. Check the file path is correct.")

def load_csv(file_path, separator='\t', header=None, names=[]):
    """Returns a pandas dataframe from reaing a csv.

    file_path -- path to a csv file
    """
    try:
        return pandas.read_csv(file_path, sep=separator, header=header, names=names)
    except FileNotFoundError:
        raise FileNotFoundError(str(file_path) + " could not be found. Check the file path is correct.")

def load_pretrained(file_path):
    """Returns a pretrain file.

    file_path -- path to the pretrained pickle file, place 'neurec' at the beginning for a pretrain file
    """
    file_path = _check_file_path(file_path, 'pretrain')
    return load_pickle(file_path)

def load_social_file(file_path, separator, header, names):
    """Returns a pandas dataframe from a csv file.

    file_path -- path to csv file
    """
    file_path = _check_file_path(file_path, 'dataset')
    return load_csv(file_path, separator=separator, header=header, names=names)

def _check_file_path(file_path, potential_folder):
    """Checks the file path for neurec package files
    and converts the file path if necessary.

    file_path -- path to csv file
    potential_folder -- potential folder which the file exists in the neurec package
    """
    if (file_path.split('/')[0] == 'neurec'):
        neurec_path = util.find_spec('neurec', package='neurec').submodule_search_locations[0]
        file_path = os.path.join(neurec_path, potential_folder, file_path.split('/')[1])

    return file_path