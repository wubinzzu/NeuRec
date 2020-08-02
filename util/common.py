__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Reduction", "InitArg"]


class Reduction(object):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"

    @classmethod
    def all(cls):
        return (cls.NONE,
                cls.SUM,
                cls.MEAN)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            key_list = ', '.join(cls.all())
            raise ValueError(f"{key} is an invalid Reduction Key, which must be one of '{key_list}'.")


class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05
