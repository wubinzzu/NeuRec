"""
@author: Zhongchuan Sun
"""
import sys
import os
import logging
from util import Configurator


class Logger(object):
    """`Logger` is a simple encapsulation of python logger.

    This class can show a message on standard output and write it into the
    file named `filename` simultaneously. This is convenient for observing
    and saving training results.
    """

    def __init__(self, filename):
        """Initializes a new `Logger` instance.

        Args:
            filename (str): File name to create. The directory component of this
                file will be created automatically if it is not existing.
        """
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()


if __name__ == '__main__':
    log = Logger('NeuRec_test.log')
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.error('error')
    log.critical('critical')

