"""Functions to handle reading files in the package."""
from configparser import ConfigParser
from importlib_resources import open_text, is_resource
import logging

logger = logging.getLogger('neurec.util.reader')

def file(path):
    """Returns the configuration settings for a file.

    path -- path to the file
    """
    parser = ConfigParser()

    try:
        with open(path) as file:
            parser.read_file(file)
    except FileNotFoundError:
        logger.error("Could not find file " + path + ". Make sure the file path is correct.")

        raise

    return parser

def lines(file_path, file_name):
    """Returns all lines from a file.

    file_path -- location of the file
    file_name -- name of the file to read
    """
    try:
        if is_resource(file_path, file_name):
            with open_text(file_path, file_name) as file:
                return file.readlines()
    except TypeError:
        logger.info(str(file_path) + str(file_name) + " is not a package dataset. Looking for a local dataset.")

    with open(file_name) as file:
        return file.readlines()
