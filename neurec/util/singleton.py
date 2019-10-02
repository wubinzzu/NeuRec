class Singleton(type):
    """Singleton class to always have one instance of a class."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Returns an instance of the class if available."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
