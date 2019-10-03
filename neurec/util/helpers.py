import numpy
import logging

logger = logging.getLogger("neurec.util.helpers")

def to_list(string):
    """Returns a list from a string.

    string -- string in format [0.5,0.5]|type
    """
    pipe_split = string.split("|")

    try:
        cast = pipe_split[1]
        array = pipe_split[0]
        array_values = array[1:-1].split(",")
    except IndexError:
        logger.error("list index out of range for " + str(string) + ". Make sure the list is the form [0,0]|type")
        raise

    return numpy.array(array_values, dtype=cast)

def to_bool(string):
    """Returns a boolean from a string.

    string -- string in format yes|no|true|false
    """
    if string.lower() in ("yes", "y", "true", "1"):
        return True
    elif string.lower() in ("no", "n", "false", "0"):
        return False

    raise ValueError(str(string) + ' not in ["yes", "y", "true", "1", "no", "n", "false", "0"].')
