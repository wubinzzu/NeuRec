def to_list(string, cast=float):
    """Returns a list from a string.

    string -- string in format [0.5,0.5]
    """
    list = string[1:-1].split(",")

    return [cast(item) for item in list]
