from abc import ABC, abstractmethod

class AbstractRecommender(ABC):
    """Abstract class for building a Recommender class."""
    @property
    @abstractmethod
    def properties(self):
        pass

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass
