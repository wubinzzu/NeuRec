"""Functions to handle reading files in the package."""
from configparser import ConfigParser
from importlib_resources import read_text, open_text

def config(config_name, item, config_path="neurec.conf"):
    """Returns the configuration settings for a file.

    config_name -- name of the file to read
    item -- section in the file to read
    config_path -- location of the configuration file (default neurec.conf) [optional]
    """
    parser = ConfigParser()

    config = read_text(config_path, config_name)
    parser.read_string(config)

    return dict(parser.items(item))

def lines(file_name, file_path="neurec.dataset"):
    """Returns all lines from a file.

    file_name -- name of the file to read
    file_path -- location of the file (default neurec.dataset) [optional]
    """
    with open_text(file_path, file_name) as file:
        return file.readlines()
