from abc import ABC, abstractmethod
from neurec.data.Dataset import Dataset
from neurec.util.properties import Properties
import logging

class AbstractRecommender(ABC):
    """Abstract class for building a Recommender class."""
    @property
    @abstractmethod
    def properties(self):
        pass

    def __init__(self, sess):
        """Sets up the model with properties, dataset, and session."""
        self.logger = logging.getLogger(__name__)
        self.conf = Properties().getProperties(self.properties)
        self.dataset = Dataset()
        self.sess = sess

        self.logger.info("Arguments: %s " %(self.conf))

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass
