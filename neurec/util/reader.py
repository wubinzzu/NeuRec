"""Functions to handle reading files in the package."""
from configparser import ConfigParser
from importlib_resources import open_text
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

def lines(file_name, file_path="neurec.dataset"):
    """Returns all lines from a file.

    file_name -- name of the file to read
    file_path -- location of the file (default neurec.dataset) [optional]
    """
    with open_text(file_path, file_name) as file:
        return file.readlines()
