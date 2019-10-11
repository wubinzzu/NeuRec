"""Functions to handle reading files in the package."""
from configparser import ConfigParser, MissingSectionHeaderError
import logging

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
